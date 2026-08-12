import copy
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest
from transformers import DeepseekV4Config, DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention as ReferenceDeepseekV4Attention,
    DeepseekV4RotaryEmbedding,
)

from sparsevllm.models.deepseek_v4 import DeepseekV4ForCausalLM
from sparsevllm.engine.model_runner import ModelRunner
from sparsevllm.models.deepseek_v4_native import (
    DeepseekV4GroupedFp8Linear,
    DeepseekV4HyperConnection,
    DeepseekV4PackedExperts,
    DeepseekV4Attention,
)
from sparsevllm.operators.moe import FlashInferCutlassFp4MoeProvider
from sparsevllm.utils.context import reset_context, set_context


def _config():
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        head_dim=32,
        q_lora_rank=32,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        mlp_layer_types=["hash_moe", "hash_moe", "hash_moe", "moe"],
        sliding_window=8,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        partial_rotary_factor=1.0,
        dtype=torch.float32,
    )
    config.sparsevllm_tiny_random = True
    return config


def _parallel_context():
    return SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        tp_all_reduce=lambda tensor: tensor,
        tp_gather=lambda tensor: [tensor],
    )


def test_tiny_reference_prefill_and_decode_match_full_sequence():
    with patch(
        "sparsevllm.layers.embed_head.get_parallel_context",
        return_value=_parallel_context(),
    ):
        torch.manual_seed(7)
        model = DeepseekV4ForCausalLM(_config()).eval()
    reference = copy.deepcopy(model.model).eval()
    input_ids = torch.arange(9, dtype=torch.long).remainder(model.config.vocab_size)
    positions = torch.arange(9, dtype=torch.long)
    seq = SimpleNamespace(seq_id=11)

    try:
        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, input_ids.numel()]),
            seqs=[seq],
        )
        with torch.inference_mode():
            actual_prefill = model(input_ids, positions)
            expected_prefill = reference(
                input_ids=input_ids.unsqueeze(0),
                position_ids=positions.unsqueeze(0),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state.squeeze(0)
        torch.testing.assert_close(actual_prefill, expected_prefill)

        set_context(False, seqs=[seq])
        with torch.inference_mode():
            actual_decode = model(torch.tensor([9]), torch.tensor([9]))
            expected_decode = reference(
                input_ids=torch.arange(10).unsqueeze(0),
                position_ids=torch.arange(10).unsqueeze(0),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1]
        torch.testing.assert_close(actual_decode, expected_decode, rtol=1e-5, atol=2e-6)
    finally:
        reset_context()


def test_formal_runtime_selects_native_model():
    config = _config()
    config.sparsevllm_tiny_random = False
    native = torch.nn.Module()
    with (
        patch(
            "sparsevllm.models.deepseek_v4.NativeDeepseekV4Model",
            return_value=native,
        ),
        patch(
            "sparsevllm.layers.embed_head.get_parallel_context",
            return_value=_parallel_context(),
        ),
    ):
        model = DeepseekV4ForCausalLM(config)

    assert model.model is native
    assert not model.tiny_random


def test_native_hyper_connection_matches_transformers_reference():
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4HyperConnection as ReferenceHyperConnection,
    )

    config = _config()
    torch.manual_seed(9)
    reference = ReferenceHyperConnection(config)
    torch.nn.init.normal_(reference.fn, std=0.02)
    torch.nn.init.zeros_(reference.base)
    torch.nn.init.ones_(reference.scale)
    native = DeepseekV4HyperConnection(config)
    native.load_state_dict(reference.state_dict())
    streams = torch.randn(2, 5, config.hc_mult, config.hidden_size)

    actual = native(streams)
    expected = reference(streams)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_native_grouped_linear_preserves_group_boundaries():
    config = _config()
    config.quantization_config = SimpleNamespace(enabled=False)
    parallel = _parallel_context()
    with patch(
        "sparsevllm.layers.linear.get_parallel_context",
        return_value=parallel,
    ):
        grouped = DeepseekV4GroupedFp8Linear(config)
    torch.manual_seed(4)
    grouped.weight.data.normal_()
    x = torch.randn(3, config.o_groups, config.num_attention_heads * config.head_dim // config.o_groups)
    weight = grouped.weight.view(config.o_groups, config.o_lora_rank, -1)

    actual = grouped(x)
    expected = torch.einsum("tgi,goi->tgo", x, weight)

    torch.testing.assert_close(actual, expected)


def test_native_fp4_expert_loader_preserves_checkpoint_bits():
    config = SimpleNamespace(
        n_routed_experts=4,
        num_experts_per_tok=2,
        hidden_size=256,
        moe_intermediate_size=128,
        dtype=torch.bfloat16,
        decode_cuda_graph=True,
        swiglu_limit=10.0,
    )
    parallel = SimpleNamespace(ep_rank=1, ep_size=2)
    with (
        patch(
            "sparsevllm.models.deepseek_v4_native.get_parallel_context",
            return_value=parallel,
        ),
        patch(
            "sparsevllm.models.deepseek_v4_native.resolve_moe_provider",
            return_value=FlashInferCutlassFp4MoeProvider(),
        ),
    ):
        experts = DeepseekV4PackedExperts(config)
    weight_bits = torch.arange(128 * 128, dtype=torch.int64).to(torch.uint8).view(128, 128)
    scale_bits = torch.arange(128 * 8, dtype=torch.int64).to(torch.uint8).view(128, 8)

    experts.load_expert_weight(
        2,
        "gate",
        weight_bits.view(torch.int8),
        scale_bits.view(torch.float8_e8m0fnu),
    )

    assert torch.equal(experts.w13_weight[0, 128:], weight_bits)
    assert torch.equal(experts.w13_scale_inv[0, 128:], scale_bits)


class _AttentionCache:
    def __init__(self, config):
        rows, max_len = 1, 256
        head_dim, index_dim = int(config.head_dim), int(config.index_head_dim)
        dtype = config.torch_dtype
        self.sliding_window = int(config.sliding_window)
        self.max_model_len = max_len
        self._decode_static_max_context_len = None
        self.raw_kv = torch.zeros(1, rows, self.sliding_window, head_dim, dtype=dtype)
        self.csa_kv = torch.zeros(1, rows, max_len // 4, head_dim, dtype=dtype)
        self.csa_index = torch.zeros(1, rows, max_len // 4, index_dim, dtype=dtype)
        self.hca_kv = torch.zeros(1, rows, max_len // 128, head_dim, dtype=dtype)
        self.csa_ring_kv = torch.zeros(1, rows, 4, 2 * head_dim, dtype=dtype)
        self.csa_ring_gate = torch.full_like(self.csa_ring_kv, float("-inf"))
        self.csa_overlap_kv = torch.zeros(1, rows, 4, head_dim, dtype=dtype)
        self.csa_overlap_gate = torch.full_like(self.csa_overlap_kv, float("-inf"))
        self.index_ring_kv = torch.zeros(1, rows, 4, 2 * index_dim, dtype=dtype)
        self.index_ring_gate = torch.full_like(self.index_ring_kv, float("-inf"))
        self.index_overlap_kv = torch.zeros(1, rows, 4, index_dim, dtype=dtype)
        self.index_overlap_gate = torch.full_like(self.index_overlap_kv, float("-inf"))
        self.hca_ring_kv = torch.zeros(1, rows, 128, head_dim, dtype=dtype)
        self.hca_ring_gate = torch.full_like(self.hca_ring_kv, float("-inf"))
        self.state = SimpleNamespace(req_indices=torch.tensor([0]))

    def csa_slot(self, layer_idx):
        return 0

    def hca_slot(self, layer_idx):
        return 0

    def compressed_capacity(self, ratio, positions):
        return (int(positions.max()) + int(ratio)) // int(ratio)

    def get_layer_batch_states(self, layer_idx):
        return self.state


@pytest.mark.parametrize(
    ("layer_type", "prefill_tokens"),
    [
        ("sliding_attention", 13),
        ("compressed_sparse_attention", 13),
        ("heavily_compressed_attention", 140),
    ],
)
def test_native_attention_prefill_and_decode_match_reference(layer_type, prefill_tokens):
    config = _config()
    config.num_hidden_layers = 1
    config.layer_types = [layer_type]
    config.mlp_layer_types = ["hash_moe"]
    config.quantization_config = SimpleNamespace(enabled=False)
    config._attn_implementation = "eager"
    parallel = _parallel_context()
    with patch(
        "sparsevllm.layers.linear.get_parallel_context",
        return_value=parallel,
    ):
        native = DeepseekV4Attention(config, 0).eval()
    reference = ReferenceDeepseekV4Attention(config, 0).eval()
    torch.manual_seed(29)
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    native.load_state_dict(reference.state_dict())
    rotary = DeepseekV4RotaryEmbedding(config)
    manager = _AttentionCache(config)
    cache = DynamicCache(config=config)
    torch.manual_seed(31)
    hidden_states = torch.randn(1, prefill_tokens, config.hidden_size)
    positions = torch.arange(prefill_tokens).unsqueeze(0)
    embeddings = {
        kind: rotary(hidden_states, positions, kind)
        for kind in ("main", "compress")
    }
    mask = create_sliding_window_causal_mask(
        config, hidden_states, None, cache, positions
    )

    with torch.inference_mode():
        expected = reference(hidden_states, embeddings, positions, mask, cache)[0]
    try:
        set_context(True, cache_manager=manager, seqs=[SimpleNamespace(seq_id=1)])
        with torch.inference_mode():
            actual = native(
                hidden_states,
                embeddings,
                positions,
                cache_rows=manager.state.req_indices,
            )[0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

        hidden_states = torch.randn(1, 1, config.hidden_size)
        positions = torch.tensor([[prefill_tokens]])
        embeddings = {
            kind: rotary(hidden_states, positions, kind)
            for kind in ("main", "compress")
        }
        with torch.inference_mode():
            expected = reference(hidden_states, embeddings, positions, None, cache)[0]
        set_context(False, cache_manager=manager, seqs=[SimpleNamespace(seq_id=1)])
        with torch.inference_mode():
            actual = native(
                hidden_states,
                embeddings,
                positions,
                cache_rows=manager.state.req_indices,
            )[0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    finally:
        reset_context()


def test_checkpoint_maps_hyper_head_parameters():
    config = _config()
    config.sparsevllm_tiny_random = False
    native = torch.nn.Module()
    with (
        patch("sparsevllm.models.deepseek_v4.NativeDeepseekV4Model", return_value=native),
        patch("sparsevllm.layers.embed_head.get_parallel_context", return_value=_parallel_context()),
    ):
        model = DeepseekV4ForCausalLM(config)

    assert model.map_weight_name("hc_head_fn") == "model.hc_head.fn"
    assert model.map_weight_name("hc_head_base") == "model.hc_head.base"
    assert model.map_weight_name("hc_head_scale") == "model.hc_head.scale"


def test_dpa_decode_partition_balances_stable_sequence_owners():
    seqs = [SimpleNamespace(seq_id=seq_id) for seq_id in (0, 4, 8)]
    for rank in range(4):
        runner = ModelRunner.__new__(ModelRunner)
        runner.parallel_context = SimpleNamespace(dp_size=4, dp_rank=rank)
        selected, owned = runner._deepseek_v4_decode_partition(seqs)

        assert len(selected) == 3
        assert owned == (3 if rank == 0 else 0)
        assert all(int(seq.seq_id) % 4 == rank for seq in selected[:owned])


def test_dpa_logits_are_restored_to_global_sequence_order():
    runner = ModelRunner.__new__(ModelRunner)
    gathered = torch.tensor([[10.0], [40.0], [20.0], [0.0]])
    runner.parallel_context = SimpleNamespace(
        dp_size=2,
        ep_all_gather_into_tensor=lambda tensor: gathered,
    )
    runner.rank = 0
    seqs = [SimpleNamespace(seq_id=0), SimpleNamespace(seq_id=1), SimpleNamespace(seq_id=4)]

    actual = runner._gather_deepseek_v4_logits(
        torch.tensor([[10.0], [40.0]]), seqs, local_owned=2
    )

    torch.testing.assert_close(actual, torch.tensor([[10.0], [20.0], [40.0]]))
