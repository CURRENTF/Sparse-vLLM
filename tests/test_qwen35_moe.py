from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sparsevllm.config import Config, QuantizationConfig
from sparsevllm.distributed import ParallelContext, ParallelGroup
from sparsevllm.engine.recurrent_state_manager import RecurrentStateManager
from sparsevllm.models.qwen3_5_moe import (
    Qwen35MoeForCausalLM,
    Qwen35MoePackedExperts,
    Qwen35MoeSparseMoeBlock,
)
from sparsevllm.models.qwen3_5 import Qwen35LinearAttention
from sparsevllm.operators.moe import TritonMoeProvider
from sparsevllm.operators.moe_router import (
    MoeRouterOpSpec,
    TorchMoeRouterProvider,
)


def _outer_config():
    layer_types = [
        "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
        for layer_idx in range(40)
    ]
    text_config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        layer_types=layer_types,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        num_experts_per_tok=8,
        hidden_act="silu",
        attn_output_gate=True,
        attention_bias=False,
        partial_rotary_factor=0.25,
        mamba_ssm_dtype="float32",
        tie_word_embeddings=False,
        rms_norm_eps=1.0e-6,
        max_position_embeddings=262144,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
    )
    return SimpleNamespace(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=text_config,
    )


def _make_config(tmp_path, **overrides):
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=_outer_config(),
    ):
        return Config(model=str(tmp_path), **overrides)


def _hybrid_context(world_rank: int) -> ParallelContext:
    world_ranks = (0, 1, 2, 3)
    moe_tp_ranks = (0, 1) if world_rank < 2 else (2, 3)
    moe_ep_ranks = (0, 2) if world_rank % 2 == 0 else (1, 3)
    return ParallelContext(
        world=ParallelGroup(None, world_ranks, world_rank, 4),
        tensor=ParallelGroup(None, world_ranks, world_rank, 4),
        expert=ParallelGroup(
            None, moe_ep_ranks, moe_ep_ranks.index(world_rank), 2
        ),
        data=ParallelGroup(None, (world_rank,), 0, 1),
        moe_tensor=ParallelGroup(
            None, moe_tp_ranks, moe_tp_ranks.index(world_rank), 2
        ),
    )


def _single_context() -> ParallelContext:
    group = ParallelGroup(None, (0,), 0, 1)
    return ParallelContext(group, group, group, group)


def _pure_tp_context(world_rank: int) -> ParallelContext:
    world = ParallelGroup(None, (0, 1), world_rank, 2)
    singleton = ParallelGroup(None, (world_rank,), 0, 1)
    return ParallelContext(
        world=world,
        tensor=world,
        expert=singleton,
        data=singleton,
        moe_tensor=world,
    )


def test_qwen36_moe_config_normalizes_text_runtime_and_topology(tmp_path):
    config = _make_config(
        tmp_path,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        decode_cuda_graph=True,
        enforce_eager=False,
    )

    assert config.hf_config.model_type == "qwen3_5_moe"
    assert config.uses_outer_tp_moe_layout is True
    assert config.world_size == 2
    assert config.moe_tensor_parallel_size == 1
    assert config.runtime_layout.full_attention_layer_indices == tuple(
        range(3, 40, 4)
    )
    assert config.runtime_layout.num_kv_layers == 10


def test_qwen36_moe_rejects_non_vanilla_sparse_method(tmp_path):
    with pytest.raises(ValueError, match="validated methods: 'vanilla'"):
        _make_config(tmp_path, vllm_sparse_method="quest")


def test_qwen36_moe_rejects_invalid_outer_tp_ep_topology(tmp_path):
    with pytest.raises(ValueError, match="must be divisible"):
        _make_config(
            tmp_path,
            tensor_parallel_size=2,
            expert_parallel_size=3,
        )


def test_qwen36_moe_rejects_non_bf16_checkpoint(tmp_path):
    outer = _outer_config()
    outer.text_config.torch_dtype = torch.float16
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=outer,
    ):
        with pytest.raises(NotImplementedError, match="requires BF16"):
            Config(model=str(tmp_path))


def test_qwen36_moe_recurrent_state_uses_attention_tp_and_fp32_state():
    config = _outer_config().text_config

    spec = Qwen35MoeForCausalLM.recurrent_state_spec(
        config, attention_tp_size=2
    )

    assert spec.tensor_specs[0].shape == (4096, 3)
    assert spec.tensor_specs[0].dtype == torch.bfloat16
    assert spec.tensor_specs[1].shape == (16, 128, 128)
    assert spec.tensor_specs[1].dtype == torch.float32


def test_qwen36_moe_torch_router_is_fp32_softmax_oracle():
    logits = torch.tensor(
        [[-80.0, -2.0, 0.0, 1.0, 3.0, 7.0, 8.0, 9.0, 10.0]],
        dtype=torch.bfloat16,
    )
    spec = MoeRouterOpSpec(
        num_experts=9,
        top_k=8,
        activation_dtype=torch.bfloat16,
        norm_topk_prob=True,
        cuda_graph=False,
    )

    weights, ids = TorchMoeRouterProvider().run(spec, logits)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    expected_weights, expected_ids = torch.topk(probabilities, 8, dim=-1)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    assert weights.dtype == torch.bfloat16
    assert ids.dtype == torch.int32
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(
        weights.float(), expected_weights, atol=4e-3, rtol=4e-3
    )


def test_qwen36_moe_linear_attention_uses_configured_recurrent_dtype():
    config = SimpleNamespace(
        hidden_size=8,
        hidden_act="silu",
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1.0e-6,
        mlp_chunk_size=16,
        torch_dtype=torch.bfloat16,
        runtime_recurrent_state_dtype=torch.float32,
        quantization_config=QuantizationConfig.disabled(),
    )
    context = _single_context()
    with (
        patch(
            "sparsevllm.models.qwen3_5.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparsevllm.layers.linear.get_parallel_context",
            return_value=context,
        ),
    ):
        attention = Qwen35LinearAttention(config)

    assert attention.recurrent_state_dtype == torch.float32


def test_recurrent_pool_accepts_model_declared_mixed_state_dtypes():
    runtime_config = SimpleNamespace(
        runtime_layout=SimpleNamespace(
            linear_attention_layer_indices=(0,),
            is_linear_attention=lambda layer_idx: int(layer_idx) == 0,
        ),
        enable_prefix_caching=False,
        max_num_seqs_in_batch=1,
        max_decoding_seqs=1,
        max_num_seqs_in_gpu=1,
        recurrent_state_max_bytes=None,
        prefix_cache_block_size=4,
    )
    state_spec = Qwen35MoeForCausalLM.recurrent_state_spec(
        _outer_config().text_config,
        attention_tp_size=1,
    )
    manager = RecurrentStateManager(
        runtime_config,
        _single_context(),
        device=torch.device("cpu"),
        state_spec=state_spec,
    )
    seq = SimpleNamespace(seq_id=1)
    manager.prepare_step([seq], is_prefill=False)
    manager.prepare_decode_static([seq], token_batch=1, device=torch.device("cpu"))

    state_buffers, _ = manager.get_decode_layer_state(
        [seq],
        layer_idx=0,
        token_batch=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert state_buffers["conv_state"].dtype == torch.bfloat16
    assert state_buffers["recurrent_state"].dtype == torch.float32


def test_packed_experts_slice_ep_before_moe_tp():
    context = _hybrid_context(world_rank=1)
    config = SimpleNamespace(
        num_experts=4,
        hidden_size=4,
        moe_intermediate_size=4,
        num_experts_per_tok=2,
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
        decode_cuda_graph=False,
        quantization_config=QuantizationConfig.disabled(),
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "sparsevllm.models.qwen3_moe.get_parallel_context",
                    return_value=context,
                )
            )
            stack.enter_context(
                patch(
                    "sparsevllm.models.qwen3_moe.resolve_moe_provider",
                    return_value=TritonMoeProvider(),
                )
            )
            experts = Qwen35MoePackedExperts(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    gate_up_global_shape = (4, 8, 4)
    down_global_shape = (4, 4, 4)
    assert experts.rank_local_weight_slice(
        gate_up_global_shape,
        loaded_shard_id="gate_up_proj",
    ) == (slice(0, 2), slice(None), slice(None))
    assert experts.rank_local_weight_slice(
        down_global_shape,
        loaded_shard_id="down_proj",
    ) == (slice(0, 2), slice(None), slice(None))

    gate_up = torch.arange(2 * 8 * 4, dtype=torch.bfloat16).view(2, 8, 4)
    down = torch.arange(2 * 4 * 4, dtype=torch.bfloat16).view(2, 4, 4)
    experts.load_packed_expert_weight("gate_up_proj", gate_up)
    experts.load_packed_expert_weight("down_proj", down)
    experts.validate_loaded_weights()

    expected_gate = gate_up[:, 2:4]
    expected_up = gate_up[:, 6:8]
    torch.testing.assert_close(experts.w13_weight[:, :2], expected_gate)
    torch.testing.assert_close(experts.w13_weight[:, 2:], expected_up)
    torch.testing.assert_close(experts.w2_weight, down[:, :, 2:4])


def test_packed_expert_pure_tp_shards_reconstruct_checkpoint():
    config = SimpleNamespace(
        num_experts=4,
        hidden_size=4,
        moe_intermediate_size=4,
        num_experts_per_tok=2,
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
        decode_cuda_graph=False,
        quantization_config=QuantizationConfig.disabled(),
    )
    gate_up = torch.arange(4 * 8 * 4, dtype=torch.bfloat16).view(4, 8, 4)
    down = torch.arange(4 * 4 * 4, dtype=torch.bfloat16).view(4, 4, 4)
    rank_experts = []
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for world_rank in (0, 1):
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "sparsevllm.models.qwen3_moe.get_parallel_context",
                        return_value=_pure_tp_context(world_rank),
                    )
                )
                stack.enter_context(
                    patch(
                        "sparsevllm.models.qwen3_moe.resolve_moe_provider",
                        return_value=TritonMoeProvider(),
                    )
                )
                experts = Qwen35MoePackedExperts(config)
                experts.load_packed_expert_weight("gate_up_proj", gate_up)
                experts.load_packed_expert_weight("down_proj", down)
                experts.validate_loaded_weights()
                rank_experts.append(experts)
    finally:
        torch.set_default_dtype(previous_dtype)

    gate = torch.cat(
        [experts.w13_weight[:, :2] for experts in rank_experts], dim=1
    )
    up = torch.cat(
        [experts.w13_weight[:, 2:] for experts in rank_experts], dim=1
    )
    reconstructed_down = torch.cat(
        [experts.w2_weight for experts in rank_experts], dim=2
    )
    torch.testing.assert_close(gate, gate_up[:, :4])
    torch.testing.assert_close(up, gate_up[:, 4:])
    torch.testing.assert_close(reconstructed_down, down)


def test_routed_output_reduces_without_reducing_shared_output_twice():
    class FixedRouter(torch.nn.Module):
        def forward(self, hidden_states):
            tokens = hidden_states.shape[0]
            return (
                torch.zeros(tokens, 4),
                torch.full((tokens, 2), 0.5),
                torch.zeros(tokens, 2, dtype=torch.int32),
            )

    class FixedExperts(torch.nn.Module):
        def forward(self, hidden_states, _topk_ids, _topk_weights):
            return torch.full_like(hidden_states, 2.0)

    class ZeroGate(torch.nn.Module):
        def forward(self, hidden_states):
            return torch.zeros(hidden_states.shape[0], 1)

    block = Qwen35MoeSparseMoeBlock.__new__(Qwen35MoeSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.shared_expert = torch.nn.Identity()
    block.shared_expert_gate = ZeroGate()
    block.gate = FixedRouter()
    block.experts = FixedExperts()
    block.parallel_context = SimpleNamespace(
        world_all_reduce=Mock(side_effect=lambda tensor: tensor * 3)
    )
    hidden_states = torch.full((2, 3), 4.0)

    output, *_ = block._forward_chunk(hidden_states)

    torch.testing.assert_close(output, torch.full_like(hidden_states, 8.0))
    block.parallel_context.world_all_reduce.assert_called_once()
    reduced_input = block.parallel_context.world_all_reduce.call_args.args[0]
    torch.testing.assert_close(reduced_input, torch.full_like(hidden_states, 2.0))
