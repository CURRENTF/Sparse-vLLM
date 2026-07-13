import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sparsevllm.platforms as platforms
from sparsevllm.config import Config, RuntimeLayout
from sparsevllm.engine.cache_manager.base import LayerBatchStates
from sparsevllm.engine.cache_manager.quest import QuestCacheManager
from sparsevllm.engine.cache_manager.snapkv import SnapKVCacheManager
from sparsevllm.engine.cache_manager.standard import StandardCacheManager
from sparsevllm.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex
from sparsevllm.engine.prefix_cache_coordinator import MixedPrefixBlockPayload, PrefixCacheCoordinator
from sparsevllm.engine.recurrent_state_manager import RecurrentStateManager, RecurrentStateSpec
from sparsevllm.engine.runtime_state import RuntimeState
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.sparse_controller import LayerBatchSparseState, SparseController
from sparsevllm.models.qwen3_5 import (
    Qwen35ForCausalLM,
    Qwen35LinearConv1D,
    Qwen35RMSNorm,
    _get_rotary_dim,
)
from sparsevllm.platforms.cpu import CpuPlatform
from sparsevllm.utils.loader import _target_weight_name_for_model, _validate_all_quantized_weights_loaded


def _qwen35_outer_config(*, num_layers: int = 64, full_layers: tuple[int, ...] | None = None):
    if full_layers is None:
        full_layers = tuple(range(0, num_layers, 4))
    full = set(full_layers)
    layer_types = ["full_attention" if idx in full else "linear_attention" for idx in range(num_layers)]
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        num_hidden_layers=num_layers,
        num_kv_layers=len(full_layers),
        layer_types=layer_types,
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        rms_norm_eps=1.0e-6,
        hidden_act="silu",
        attention_bias=False,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        max_position_embeddings=131072,
        torch_dtype=torch.float16,
        tie_word_embeddings=False,
        quantization_config={
            "quant_method": "fp8",
            "weight_dtype": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )
    return SimpleNamespace(model_type="qwen3_5", text_config=text_config)


def _make_config(tmp_path, **kwargs):
    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_qwen35_outer_config()):
        return Config(model=str(tmp_path), **kwargs)


def test_runtime_layout_maps_qwen35_full_layers_to_compact_kv_indices():
    layout = RuntimeLayout.from_config(_qwen35_outer_config().text_config, require_mixed=True)

    assert layout.num_layers == 64
    assert layout.num_kv_layers == 16
    assert layout.full_attention_layer_indices[:4] == (0, 4, 8, 12)
    assert layout.linear_attention_layer_indices[:3] == (1, 2, 3)
    assert layout.kv_layer_index(0) == 0
    assert layout.kv_layer_index(4) == 1
    with pytest.raises(RuntimeError, match="linear_attention"):
        layout.kv_layer_index(1)


def test_omnikv_observation_layers_follow_compact_qwen_kv_order(tmp_path):
    outer_config = _qwen35_outer_config(full_layers=tuple(range(3, 64, 4)))
    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(
            model=str(tmp_path),
            vllm_sparse_method="omnikv",
            full_attn_layers="3,11,23,31,35,47,59",
        )

    assert cfg.obs_layer_ids == [3, 11, 23, 35, 47, 59]


def test_qwen35_pyramidkv_ratios_follow_compact_kv_order(tmp_path):
    outer_config = _qwen35_outer_config(full_layers=tuple(range(3, 64, 4)))
    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(
            model=str(tmp_path),
            vllm_sparse_method="pyramidkv",
            pyramidkv_start_ratio=0.5,
            pyramidkv_least_ratio=0.1,
        )

    assert len(cfg.pyramid_layer_ratios) == cfg.runtime_layout.num_kv_layers == 16
    assert cfg.pyramid_layer_ratios[0] == pytest.approx(0.5)
    assert cfg.pyramid_layer_ratios[-1] == pytest.approx(0.1)

    controller = SparseController(cfg, SimpleNamespace())
    first_kv_layer = cfg.runtime_layout.kv_idx_to_layer_idx[0]
    last_kv_layer = cfg.runtime_layout.kv_idx_to_layer_idx[-1]
    assert controller._get_layer_budget(first_kv_layer, is_prefill=True) == 4672
    assert controller._get_layer_budget(last_kv_layer, is_prefill=True) == 1395


def test_qwen35_pyramidkv_allocates_slots_only_for_kv_layers(tmp_path):
    outer_config = _qwen35_outer_config(num_layers=8, full_layers=(3, 7))
    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(
            model=str(tmp_path),
            vllm_sparse_method="pyramidkv",
            pyramidkv_start_ratio=0.5,
            pyramidkv_least_ratio=0.1,
        )

    manager = object.__new__(SnapKVCacheManager)
    manager.config = cfg
    manager.hf_config = cfg.hf_config
    manager.runtime_layout = cfg.runtime_layout
    manager.num_layers = cfg.runtime_layout.num_layers
    manager.num_kv_layers = cfg.runtime_layout.num_kv_layers
    manager.num_kv_heads = 1
    manager.head_dim = 1
    manager.device = torch.device("cpu")
    manager.pyramidkv_prefill_staging_num_slots = 0
    manager.pyramidkv_prefill_staging_kv_cache = None
    manager._get_available_slots_info = lambda: (1_000, 8)
    manager._pyramidkv_can_use_full_prefill_staging = lambda: False

    SnapKVCacheManager.allocate_kv_cache(manager)

    assert len(manager.kv_cache) == 2
    assert [cfg.num_kvcache_slots[idx] for idx in (0, 1, 2, 4, 5, 6)] == [0] * 6
    assert cfg.num_kvcache_slots[3] > cfg.num_kvcache_slots[7] > 0


def test_qwen35_snapkv_initializes_compact_kv_metadata(tmp_path):
    outer_config = _qwen35_outer_config(num_layers=8, full_layers=(3, 7))
    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(model=str(tmp_path), vllm_sparse_method="snapkv")
    cfg.max_model_len = 8
    cfg.max_num_seqs_in_batch = 2

    def allocate_small_cache(manager):
        manager.config.num_kvcache_slots = 16
        manager.kv_cache = torch.empty(0)

    with (
        patch.object(platforms, "_current_platform", CpuPlatform()),
        patch.object(SnapKVCacheManager, "allocate_kv_cache", allocate_small_cache),
    ):
        manager = SnapKVCacheManager(cfg, rank=0, world_size=1)

    assert manager.buffer_req_to_token_slots_tensor.shape == (2, 6, 8)
    assert manager.free_slots_stack_tensor.shape == (2, 16)
    assert (
        manager.buffer_req_to_token_slots[3].data_ptr()
        == manager.buffer_req_to_token_slots_tensor[0].data_ptr()
    )
    assert (
        manager.buffer_req_to_token_slots[7].data_ptr()
        == manager.buffer_req_to_token_slots_tensor[1].data_ptr()
    )
    assert manager.buffer_req_to_token_slots[0] is None
    assert manager.free_slots_stack[0] is None

    seq_a = Sequence([1, 2])
    seq_b = Sequence([3, 4])
    seq_a.seq_id = 10
    seq_b.seq_id = 11
    seq_a.current_chunk_size = 2
    seq_b.current_chunk_size = 2
    layers_slot_mapping = torch.full((8, 4), -1, dtype=torch.int32)

    assert manager._allocate_prefill_batch_same_size_all_layers(
        [seq_a, seq_b],
        layers_slot_mapping,
    )
    assert manager._num_free_slots[3] == manager._num_free_slots[7] == 12
    assert manager._num_free_slots[0] == 0
    assert manager.buffer_req_to_token_slots_tensor[:, :, :2].tolist() == [
        [[14, 15], [12, 13], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[14, 15], [12, 13], [0, 0], [0, 0], [0, 0], [0, 0]],
    ]
    assert layers_slot_mapping[3].tolist() == [14, 15, 12, 13]
    assert layers_slot_mapping[7].tolist() == [14, 15, 12, 13]
    assert layers_slot_mapping[0].tolist() == [-1, -1, -1, -1]

    selected_slots, next_lens, _, wrote_slot_output = manager._allocate_decode_batch_all_layers(
        [seq_a.seq_id, seq_b.seq_id]
    )
    assert not wrote_slot_output
    assert selected_slots[3].tolist() == [10, 11]
    assert selected_slots[7].tolist() == [10, 11]
    assert selected_slots[0].tolist() == [-1, -1]
    assert next_lens[3].tolist() == [3, 3]
    assert next_lens[7].tolist() == [3, 3]
    assert manager.buffer_req_to_token_slots_tensor[:, :2, 2].tolist() == [
        [10, 11],
        [10, 11],
    ]


def test_qwen35_pyramidkv_projects_legacy_transformer_layer_ratios(tmp_path):
    outer_config = _qwen35_outer_config(num_layers=8, full_layers=(3, 7))
    legacy_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=outer_config):
        cfg = Config(
            model=str(tmp_path),
            vllm_sparse_method="pyramidkv",
            pyramid_layer_ratios=legacy_ratios,
        )

    assert cfg.pyramid_layer_ratios == [legacy_ratios[3], legacy_ratios[7]]


def test_qwen35_prefix_block_defaults_to_4096_and_rejects_unaligned(tmp_path):
    cfg = _make_config(tmp_path, enable_prefix_caching=True, prefix_cache_block_size=None)
    assert cfg.prefix_cache_block_size == 4096

    with pytest.raises(ValueError, match="4096\\*N"):
        _make_config(tmp_path, enable_prefix_caching=True, prefix_cache_block_size=2048)
    with pytest.raises(ValueError, match="4096\\*N"):
        _make_config(tmp_path, enable_prefix_caching=True, prefix_cache_block_size=4097)


def test_qwen35_quest_prefix_block_may_span_multiple_pages(tmp_path):
    cfg = _make_config(
        tmp_path,
        vllm_sparse_method="quest",
        enable_prefix_caching=True,
        quest_chunk_size=16,
    )

    assert cfg.prefix_cache_block_size == 4096


def test_qwen35_deltakv_requires_compatible_checkpoint_even_when_missing_allowed(tmp_path):
    with pytest.raises(ValueError, match="DeltaKV for qwen3_5 requires"):
        _make_config(
            tmp_path,
            vllm_sparse_method="deltakv",
            allow_missing_deltakv_path=True,
            kv_quant_bits=0,
        )


def test_qwen35_raw_config_fallback_when_transformers_autoconfig_is_unknown(tmp_path):
    outer_config = _qwen35_outer_config()
    text_config = dict(vars(outer_config.text_config))
    text_config["torch_dtype"] = "float16"
    with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "qwen3_5",
                "text_config": text_config,
            },
            f,
        )

    with patch("sparsevllm.config.AutoConfig.from_pretrained", side_effect=ValueError("unknown model")):
        cfg = Config(model=str(tmp_path))

    assert cfg.outer_hf_config.model_type == "qwen3_5"
    assert cfg.hf_config.model_type == "qwen3_5"
    assert cfg.runtime_layout.num_kv_layers == 16


def test_qwen35_linear_conv1d_matches_hf_biasless_checkpoint():
    with patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.get_world_size", return_value=1):
        conv = Qwen35LinearConv1D(conv_dim=16, kernel_size=4, qk_dim=4, v_dim=8)

    assert conv.bias is None
    assert "bias" not in dict(conv.named_parameters())


def test_qwen35_rotary_dim_uses_partial_rotary_factor():
    config = SimpleNamespace(rope_parameters={"partial_rotary_factor": 0.25})

    assert _get_rotary_dim(config, 256) == 64
    assert _get_rotary_dim(SimpleNamespace(partial_rotary_factor=0.5), 256) == 128
    assert _get_rotary_dim(SimpleNamespace(qk_rope_head_dim=96), 256) == 96


def test_qwen35_rmsnorm_uses_hf_offset_weight_semantics():
    norm = Qwen35RMSNorm(4, eps=1.0e-6)
    norm.weight.data.copy_(torch.tensor([0.5, -0.25, 0.0, 1.0]))
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)

    out = norm(x)

    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1.0e-6)
    expected = expected * (1.0 + norm.weight)
    assert torch.allclose(out, expected)


def test_qwen35_rmsnorm_residual_path_uses_hf_offset_weight_semantics():
    norm = Qwen35RMSNorm(4, eps=1.0e-6)
    norm.weight.data.copy_(torch.tensor([0.5, -0.25, 0.0, 1.0]))
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)
    residual_in = torch.tensor([[0.5, 0.5, -1.0, 2.0]], dtype=torch.float32)

    out, residual = norm(x, residual_in)

    merged = x + residual_in
    expected = merged * torch.rsqrt(merged.pow(2).mean(dim=-1, keepdim=True) + 1.0e-6)
    expected = expected * (1.0 + norm.weight)
    assert torch.allclose(out, expected)
    assert torch.allclose(residual, merged)


def test_qwen35_linear_attention_repeats_qk_to_value_heads():
    from sparsevllm.models.qwen3_5 import Qwen35LinearAttention

    attn = object.__new__(Qwen35LinearAttention)
    attn.num_k_heads = 2
    attn.num_v_heads = 6
    q = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
    k = q + 100

    q_rep, k_rep = attn._repeat_qk_for_value_heads(q, k)

    assert q_rep.shape == (1, 4, 6, 3)
    assert k_rep.shape == (1, 4, 6, 3)
    assert torch.equal(q_rep[:, :, 0], q[:, :, 0])
    assert torch.equal(q_rep[:, :, 1], q[:, :, 0])
    assert torch.equal(q_rep[:, :, 2], q[:, :, 0])
    assert torch.equal(q_rep[:, :, 3], q[:, :, 1])
    assert torch.equal(k_rep[:, :, 5], k[:, :, 1])


def test_qwen35_decode_state_padding_preserves_real_rows_for_static_batch():
    from sparsevllm.models.qwen3_5 import Qwen35LinearAttention

    conv = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    recurrent = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)

    padded_conv, padded_recurrent = Qwen35LinearAttention._pad_decode_states_for_static_batch(
        conv,
        recurrent,
        token_batch=4,
        real_batch=2,
    )

    assert padded_conv.shape == (4, 3, 4)
    assert padded_recurrent.shape == (4, 3, 2, 2)
    assert torch.equal(padded_conv[:2], conv)
    assert torch.equal(padded_recurrent[:2], recurrent)
    assert torch.equal(padded_conv[2], conv[0])
    assert torch.equal(padded_recurrent[3], recurrent[0])


def test_linear_layer_kv_hook_fails_fast_without_allocating_cache():
    manager = object.__new__(StandardCacheManager)
    manager.runtime_layout = RuntimeLayout.from_config(_qwen35_outer_config().text_config, require_mixed=True)

    with pytest.raises(RuntimeError, match="linear_attention"):
        manager.get_layer_buffer_req_to_token_slots(1)


def test_snapkv_prefill_context_lens_stays_dense_for_mixed_layout():
    manager = object.__new__(SnapKVCacheManager)
    manager.device = torch.device("cpu")
    manager.num_layers = 8
    manager.runtime_layout = RuntimeLayout.from_config(
        _qwen35_outer_config(num_layers=8, full_layers=(3, 7)).text_config,
        require_mixed=True,
    )
    manager.layer_batch_states = [LayerBatchStates() for _ in range(manager.num_layers)]
    manager.seq_id_to_row = [{} for _ in range(manager.num_layers)]
    for layer_idx in manager.kv_transformer_layer_indices():
        manager.seq_id_to_row[layer_idx] = {10: 0, 11: 1}
    manager._clear_prefill_attention_scores = lambda seq_id: None
    manager._should_use_pyramidkv_long_prefill_offload_staging = lambda seqs: False
    manager._should_use_pyramidkv_full_prefill_staging = lambda seqs: False

    def fake_batched_alloc(seqs, layers_slot_mapping):
        del seqs
        layers_slot_mapping[3].copy_(torch.arange(layers_slot_mapping.shape[1], dtype=torch.int32))
        layers_slot_mapping[7].copy_(torch.arange(layers_slot_mapping.shape[1], dtype=torch.int32) + 100)
        return True

    manager._allocate_prefill_batch_same_size_all_layers = fake_batched_alloc
    seqs = [
        SimpleNamespace(seq_id=10, current_chunk_size=5, num_prefilled_tokens=0, token_ids=[1, 2, 3, 4, 5]),
        SimpleNamespace(seq_id=11, current_chunk_size=5, num_prefilled_tokens=0, token_ids=[6, 7, 8, 9, 10]),
    ]

    input_ids, positions, cu_seqlens_q = SnapKVCacheManager._prepare_prefill(manager, seqs)

    assert input_ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert positions.tolist() == list(range(5)) + list(range(5))
    assert cu_seqlens_q.tolist() == [0, 5, 10]
    assert manager.layer_batch_states[3].context_lens.tolist() == [5, 5]
    assert manager.layer_batch_states[7].context_lens.tolist() == [5, 5]
    assert manager.layer_batch_states[0].context_lens is None


def test_omnikv_uses_first_target_kv_layer_for_slot_table_in_mixed_layout():
    class FakeCacheManager:
        def __init__(self):
            self.requested_layers = []

        def get_layer_buffer_req_to_token_slots(self, layer_idx):
            self.requested_layers.append(int(layer_idx))
            return torch.arange(20, dtype=torch.int32).reshape(1, 20)

    controller = object.__new__(SparseController)
    controller.sparse_method = "omnikv"
    controller.is_deltakv_family = False
    controller.debug_dynamic_selection = {}
    controller.debug_dynamic_selection_detail = False
    controller.dynamic_deltakv_topk_tiebreak = False
    controller.device = torch.device("cpu")
    controller.cache_manager = FakeCacheManager()
    controller.num_sink = 0
    controller.num_recent = 1
    controller.decode_keep_tokens = 2
    controller.layer_batch_sparse_states = {
        3: LayerBatchSparseState(),
        7: LayerBatchSparseState(),
    }
    obs_state = controller.layer_batch_sparse_states[3]
    obs_state.attn_score = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    obs_state.context_lens = torch.tensor([10], dtype=torch.int32)
    obs_state.req_indices = torch.tensor([0], dtype=torch.int32)

    def fake_build(topk_indices, topk_lens, hist_lens, recent_chunk_lens, slot_table, req_indices, num_sink, max_s):
        del topk_indices, topk_lens, hist_lens, recent_chunk_lens, slot_table, req_indices, num_sink
        keep = torch.zeros((1, max_s), dtype=torch.int32)
        slots = torch.zeros((1, max_s), dtype=torch.int32)
        lens = torch.tensor([max_s], dtype=torch.int32)
        return keep, slots, lens

    with patch("sparsevllm.engine.sparse_controller.get_context", return_value=SimpleNamespace(is_long_text=True, is_prefill=False)):
        with patch("sparsevllm.engine.sparse_controller.build_omnikv_keep_and_slots", side_effect=fake_build):
            SparseController._update_dynamic_omnikv_indices(controller, 3, [7])

    assert controller.cache_manager.requested_layers == [7]
    assert controller.layer_batch_sparse_states[7].active_slots is not None


def test_standard_mixed_prefix_payload_preserves_block_range():
    manager = object.__new__(StandardCacheManager)
    manager.seq_id_to_row = {7: 0}
    manager.row_seq_lens = [8192]
    manager.buffer_req_to_token_slots = torch.arange(8192, dtype=torch.int32).reshape(1, 8192)
    seq = SimpleNamespace(seq_id=7)

    payload = manager.build_prefix_kv_payload(seq, 4096, 8192)

    assert payload.block_start == 4096
    assert payload.block_end == 8192
    assert payload.token_slots[0].item() == 4096


class _BoundaryCacheManager:
    def prefill_step_free_slots_for(self, seq):
        del seq
        return 9999


class _BoundaryCoordinator:
    block_size = 4096

    def evictable_slots(self):
        return 0


def test_mixed_runtime_caps_prefill_chunks_at_recurrent_snapshot_boundary():
    runtime_state = RuntimeState(
        config=None,
        cache_manager=_BoundaryCacheManager(),
        prefix_cache_coordinator=_BoundaryCoordinator(),
    )
    seq = SimpleNamespace(num_prompt_tokens=5000, num_prefilled_tokens=0, prefix_cache_hit_len=0)

    assert runtime_state.prefill_step_free_slots_for(seq) == 4096

    seq.num_prefilled_tokens = 4096
    assert runtime_state.prefill_step_free_slots_for(seq) == 904


def test_mixed_runtime_uses_prefix_hit_as_virtual_prefill_boundary():
    runtime_state = RuntimeState(
        config=None,
        cache_manager=_BoundaryCacheManager(),
        prefix_cache_coordinator=_BoundaryCoordinator(),
    )
    seq = SimpleNamespace(num_prompt_tokens=5000, num_prefilled_tokens=0, prefix_cache_hit_len=4096)

    assert runtime_state.prefill_step_free_slots_for(seq) == 904


def test_mixed_prefix_rejects_prefill_chunk_crossing_recurrent_snapshot_boundary():
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.prefix_cache = object()
    coordinator.block_size = 4096
    coordinator.capacity_limited_seq_ids = set()
    seq = SimpleNamespace(
        seq_id=11,
        num_prefilled_tokens=0,
        current_chunk_size=5000,
        token_ids=list(range(5000)),
    )

    with pytest.raises(RuntimeError, match="snapshot boundaries"):
        coordinator.record_step_tokens([seq], is_prefill=True)


def test_runtime_warmup_reset_clears_coordinator_before_kv_allocator():
    calls = []
    cache_manager = SimpleNamespace(reset_after_warmup=lambda: calls.append("kv"))
    recurrent_manager = SimpleNamespace(reset_after_warmup=lambda: calls.append("recurrent"))
    coordinator = SimpleNamespace(reset_after_warmup=lambda: calls.append("coordinator"))
    runtime_state = RuntimeState(
        config=None,
        cache_manager=cache_manager,
        recurrent_state_manager=recurrent_manager,
        prefix_cache_coordinator=coordinator,
    )

    runtime_state.reset_after_warmup()

    assert calls == ["coordinator", "kv", "recurrent"]


def test_runtime_state_does_not_proxy_undeclared_cache_manager_api():
    runtime_state = RuntimeState(
        config=None,
        cache_manager=SimpleNamespace(method_specific_api=lambda: "leaked"),
    )

    with pytest.raises(AttributeError, match="method_specific_api"):
        runtime_state.method_specific_api()


def test_generic_loader_only_applies_model_declared_weight_rules():
    generic_model = SimpleNamespace()
    assert (
        _target_weight_name_for_model(generic_model, "model.language_model.layers.0.weight")
        == "model.language_model.layers.0.weight"
    )
    assert _target_weight_name_for_model(generic_model, "visual.encoder.weight") == "visual.encoder.weight"

    qwen35_model = object.__new__(Qwen35ForCausalLM)
    assert (
        _target_weight_name_for_model(qwen35_model, "model.language_model.layers.0.weight")
        == "model.layers.0.weight"
    )
    assert _target_weight_name_for_model(qwen35_model, "visual.encoder.weight") is None


def test_mixed_prefix_recurrent_byte_budget_evicts_freeable_block():
    freed = []
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.block_size = 4
    coordinator.max_recurrent_bytes = 8
    coordinator.cache_manager = SimpleNamespace(free_prefix_kv_payload=lambda payload: freed.append(("kv", payload)))
    coordinator.recurrent_state_manager = SimpleNamespace(
        free_prefix_recurrent_payload=lambda payload: freed.append(("recurrent", payload))
    )
    coordinator.prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"test")
    block_id = coordinator.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload="kv-0",
            recurrent_payload="state-0",
            token_count=4,
            accounting_bytes=6,
            recurrent_bytes=6,
        ),
        token_ids=(1, 2, 3, 4),
    )
    coordinator.prefix_cache.insert_block(block)

    assert coordinator._evict_for_insert(1, incoming_recurrent_bytes=6)

    assert len(coordinator.prefix_cache) == 0
    assert freed == [("kv", "kv-0"), ("recurrent", "state-0")]


def test_mixed_prefix_capacity_skips_insert_when_live_chain_is_referenced():
    coordinator = object.__new__(PrefixCacheCoordinator)
    coordinator.block_size = 4
    coordinator.max_recurrent_bytes = 8
    coordinator.cache_manager = SimpleNamespace(free_prefix_kv_payload=lambda payload: None)
    coordinator.recurrent_state_manager = SimpleNamespace(free_prefix_recurrent_payload=lambda payload: None)
    coordinator.prefix_cache = RadixPrefixIndex(block_size=4, fingerprint=b"test")
    block_id = coordinator.prefix_cache.stable_block_id([1, 2, 3, 4], None)
    block = PrefixCacheBlock(
        stable_block_id=block_id,
        parent_block_id=None,
        block_size=4,
        logical_block_idx=0,
        payload=MixedPrefixBlockPayload(
            kv_payload="kv-0",
            recurrent_payload="state-0",
            token_count=4,
            accounting_bytes=6,
            recurrent_bytes=6,
        ),
        token_ids=(1, 2, 3, 4),
        ref_count=1,
    )
    coordinator.prefix_cache.insert_block(block)

    assert not coordinator._evict_for_insert(1, incoming_recurrent_bytes=6)
    assert coordinator.prefix_cache.get_block(block_id) is block


def test_recurrent_state_manager_reuses_preallocated_rows_for_decode():
    config = SimpleNamespace(
        runtime_layout=None,
        max_num_seqs_in_batch=2,
        max_decoding_seqs=4,
        prefix_cache_block_size=4,
    )
    manager = RecurrentStateManager(
        config,
        rank=0,
        world_size=1,
        state_spec=RecurrentStateSpec(
            name="test recurrent",
            state_names=("conv_state", "recurrent_state"),
        ),
    )
    assert manager.row_capacity == 6
    seq = SimpleNamespace(seq_id=7)
    manager.prepare_step([seq], is_prefill=True)
    conv = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    recurrent = torch.arange(8, dtype=torch.float16).reshape(2, 2, 2)
    manager.set_layer_state(7, 1, {"conv_state": conv, "recurrent_state": recurrent})
    first_ptr = manager.get_layer_state(7, 1)["conv_state"].data_ptr()

    manager.set_layer_state(7, 1, {"conv_state": conv + 1, "recurrent_state": recurrent + 1})
    manager.prepare_decode_static([seq], token_batch=4, device=torch.device("cpu"))
    state_buffers, state_indices = manager.get_decode_layer_state(
        [seq],
        layer_idx=1,
        token_batch=4,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    conv_pool = state_buffers["conv_state"]
    recurrent_pool = state_buffers["recurrent_state"]

    assert manager.get_layer_state(7, 1)["conv_state"].data_ptr() == first_ptr
    assert conv_pool.shape == (7, 2, 3)
    assert recurrent_pool.shape == (7, 2, 2, 2)
    assert conv_pool.dtype == torch.float16
    assert recurrent_pool.dtype == torch.float16
    assert state_indices.tolist() == [0, 6, 6, 6]
    assert torch.equal(conv_pool[0], conv + 1)
    assert torch.count_nonzero(conv_pool[6]).item() == 0

    manager.free_seq(7)
    replacement = SimpleNamespace(seq_id=8)
    manager.prepare_step([replacement], is_prefill=True)
    manager.set_layer_state(8, 1, {"conv_state": conv + 2, "recurrent_state": recurrent + 2})
    manager.prepare_decode_static([replacement], token_batch=4, device=torch.device("cpu"))

    assert manager.decode_state_indices[4].data_ptr() == state_indices.data_ptr()
    assert manager.decode_state_indices[4].tolist() == [1, 6, 6, 6]

    manager.reset_after_warmup()

    assert manager.decode_state_indices[4].data_ptr() == state_indices.data_ptr()


def test_recurrent_state_manager_uses_model_declared_state_schema():
    config = SimpleNamespace(
        runtime_layout=None,
        max_num_seqs_in_batch=1,
        max_decoding_seqs=1,
        prefix_cache_block_size=4,
    )
    manager = RecurrentStateManager(
        config,
        rank=0,
        world_size=1,
        state_spec=RecurrentStateSpec(name="single-state model", state_names=("ssm_state",)),
    )
    seq = SimpleNamespace(seq_id=3)
    state = torch.arange(4, dtype=torch.float16)

    manager.set_layer_state(seq.seq_id, 0, {"ssm_state": state})
    manager.prepare_decode_static([seq], token_batch=1, device=torch.device("cpu"))
    state_buffers, state_indices = manager.get_decode_layer_state(
        [seq],
        layer_idx=0,
        token_batch=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )

    assert tuple(state_buffers) == ("ssm_state",)
    assert torch.equal(state_buffers["ssm_state"][0], state)
    assert state_indices.tolist() == [0]
    with pytest.raises(RuntimeError, match="schema mismatch"):
        manager.set_layer_state(seq.seq_id, 0, {"conv_state": state})


def test_quest_mixed_prefix_payload_spans_multiple_pages():
    manager = object.__new__(QuestCacheManager)
    manager.page_size = 2
    manager.device = torch.device("cpu")
    manager.num_pages = 8
    manager.page_offsets_i32 = torch.arange(2, dtype=torch.int32)
    manager.seq_id_to_row = {7: 0}
    manager.row_seq_lens = torch.tensor([4, 0], dtype=torch.int32).numpy()
    manager.buffer_req_to_token_slots = torch.zeros((2, 8), dtype=torch.int32)
    manager.buffer_req_to_token_slots[0, :4] = torch.tensor([0, 1, 4, 5], dtype=torch.int32)
    manager.buffer_req_to_page_slots = torch.full((2, 4), -1, dtype=torch.int32)
    manager.seq_id_to_cached_pages = {}
    manager.free_rows = deque([1])
    manager._num_free_pages = 6
    manager.free_pages_stack = torch.empty(8, dtype=torch.int32)

    payload = manager.build_prefix_kv_payload(SimpleNamespace(seq_id=7), 0, 4)
    manager.seq_id_to_row.pop(7)
    manager.attach_prefix_kv_payload(SimpleNamespace(seq_id=8), payload)

    assert payload.block_slot is None
    assert payload.block_slots.tolist() == [0, 2]
    assert manager.buffer_req_to_page_slots[1, :2].tolist() == [0, 2]
    assert manager.buffer_req_to_token_slots[1, :4].tolist() == [0, 1, 4, 5]
    assert manager.row_seq_lens[1] == 4

    manager.free_prefix_kv_payload(payload)

    assert manager._num_free_pages == 8
    assert manager.free_pages_stack[6:8].tolist() == [0, 2]


def test_quantized_loader_rejects_unloaded_fp8_modules():
    class FakeQuantizedLinear(torch.nn.Module):
        quantized = True
        _quantized_weight_loaded = False
        _quantized_loaded_ranges = [(0, 128)]

    model = torch.nn.Module()
    model.proj = FakeQuantizedLinear()

    with pytest.raises(ValueError, match="Missing FP8 weight loads"):
        _validate_all_quantized_weights_loaded(model)
