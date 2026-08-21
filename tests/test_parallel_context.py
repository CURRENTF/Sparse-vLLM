from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist

import sparsevllm.platforms as platforms
from sparsevllm.config import Config, RuntimeLayout
from sparsevllm.distributed import (
    ParallelContext,
    ParallelGroup,
    ParallelMode,
    ParallelTopology,
    parallel_group_ranks,
    parallel_ranks_from_world_rank,
    world_rank_from_parallel_ranks,
)
from sparsevllm.distributed.parallel_context import (
    get_parallel_context,
    init_parallel_context,
    reset_parallel_context,
)
from sparsevllm.engine.cache_manager.base import CacheManager
from sparsevllm.layers.embed_head import VocabParallelEmbedding
from sparsevllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from sparsevllm.platforms.cpu import CpuPlatform


def _replicated_ep_context(world_rank: int = 2, world_size: int = 4) -> ParallelContext:
    return ParallelContext(
        world=ParallelGroup(None, tuple(range(world_size)), world_rank, world_size),
        tensor=ParallelGroup(None, (world_rank,), 0, 1),
        expert=ParallelGroup(None, tuple(range(world_size)), world_rank, world_size),
        data=ParallelGroup(None, (world_rank,), 0, 1),
    )


class _MinimalCacheManager(CacheManager):
    def allocate_kv_cache(self):
        raise NotImplementedError

    def get_layer_batch_states(self, layer_idx):
        raise NotImplementedError

    def get_layer_kv_cache(self, layer_idx):
        raise NotImplementedError

    def get_layer_store_view(self, layer_idx):
        raise NotImplementedError

    def get_layer_compute_tensors(self, layer_idx, selection=None):
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx):
        raise NotImplementedError

    @property
    def num_free_slots(self):
        return 0

    def free_seq(self, seq_id):
        raise NotImplementedError

    def free_part_slots(self, layer_idx, seq, keep_indices):
        raise NotImplementedError

    def _prepare_prefill(self, seqs):
        raise NotImplementedError

    def _prepare_decode(self, seqs):
        raise NotImplementedError


def _hf_config(model_type: str = "qwen3_moe", *, num_experts: int = 8):
    return SimpleNamespace(
        model_type=model_type,
        torch_dtype=torch.bfloat16,
        max_position_embeddings=32768,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        num_experts=num_experts,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


def test_world_rank_mapping_round_trips():
    topology = ParallelTopology(2, 3, 4)
    for world_rank in range(24):
        ranks = parallel_ranks_from_world_rank(topology, world_rank)
        assert world_rank_from_parallel_ranks(topology, *ranks) == world_rank


def test_parallel_group_members_follow_dp_ep_tp_layout():
    tensor_groups = ((0, 1), (2, 3), (4, 5), (6, 7))
    assert parallel_group_ranks(ParallelTopology(2, 2, 2)) == {
        "tensor": tensor_groups,
        "expert": ((0, 2), (1, 3), (4, 6), (5, 7)),
        "data": ((0, 4), (1, 5), (2, 6), (3, 7)),
        "moe_tensor": tensor_groups,
    }


def test_hybrid_moe_groups_split_outer_attention_world():
    topology = ParallelTopology(4, 2, 1, ParallelMode.OUTER_TP_MOE)
    assert parallel_group_ranks(topology) == {
        "tensor": ((0, 1, 2, 3),),
        "moe_tensor": ((0, 1), (2, 3)),
        "expert": ((0, 2), (1, 3)),
        "data": ((0,), (1,), (2,), (3,)),
    }


def test_parallel_topology_resolves_rank_local_sizes():
    standard = ParallelTopology(2, 4, 1)
    hybrid = ParallelTopology(4, 2, 1, ParallelMode.OUTER_TP_MOE)

    assert (standard.world_size, standard.attention_tp_size, standard.moe_tp_size) == (8, 2, 2)
    assert (hybrid.world_size, hybrid.attention_tp_size, hybrid.moe_tp_size) == (4, 4, 2)


@pytest.mark.parametrize(
    "topology",
    [
        (0, 1, 1, ParallelMode.STANDARD),
        (4, 3, 1, ParallelMode.OUTER_TP_MOE),
        (4, 2, 2, ParallelMode.OUTER_TP_MOE),
    ],
)
def test_parallel_topology_rejects_invalid_sizes(topology):
    with pytest.raises(ValueError):
        ParallelTopology(*topology)


def test_hybrid_moe_parallel_context_uses_explicit_groups():
    reset_parallel_context()
    with (
        patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=4),
            patch.object(dist, "get_rank", return_value=2),
            patch.object(dist, "get_backend", return_value=dist.Backend.GLOO),
            patch.object(dist, "new_group", side_effect=lambda _ranks: object()),
    ):
        context = init_parallel_context(
            topology=ParallelTopology(4, 2, 1, ParallelMode.OUTER_TP_MOE),
        )
    assert context.attention.ranks == (0, 1, 2, 3)
    assert context.attention_tp_rank == 2
    assert context.moe_tensor.ranks == (2, 3)
    assert context.moe_tp_rank == 0
    assert context.expert.ranks == (0, 2)
    assert context.ep_rank == 1
    reset_parallel_context()


def test_parallel_context_lifecycle_and_local_groups():
    reset_parallel_context()
    fake_groups = []

    def new_group(ranks):
        group = object()
        fake_groups.append((tuple(ranks), group))
        return group

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=4),
        patch.object(dist, "get_rank", return_value=2),
        patch.object(dist, "get_backend", return_value=dist.Backend.GLOO),
        patch.object(dist, "new_group", side_effect=new_group),
    ):
        topology = ParallelTopology(1, 2, 2)
        context = init_parallel_context(topology=topology)
        assert context.world_rank == 2
        assert context.tp_rank == 0
        assert context.tp_size == 1
        assert context.ep_rank == 0
        assert context.expert.ranks == (2, 3)
        assert context.dp_rank == 1
        assert context.data.ranks == (0, 2)
        assert get_parallel_context() is context
        with pytest.raises(RuntimeError, match="already initialized"):
            init_parallel_context(topology=topology)

    assert [ranks for ranks, _ in fake_groups] == [
        (0, 1),
        (2, 3),
        (0, 2),
        (1, 3),
    ]
    reset_parallel_context()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_parallel_context()


def test_parallel_context_rejects_world_size_mismatch():
    reset_parallel_context()
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=2),
        patch.object(dist, "get_rank", return_value=0),
    ):
        with pytest.raises(ValueError, match="does not match"):
            init_parallel_context(topology=ParallelTopology(1, 4, 1))


def test_ep_broadcast_uses_source_world_rank():
    context = _replicated_ep_context(world_rank=2, world_size=4)
    tensor = torch.tensor([1.0])

    with patch.object(dist, "broadcast", return_value=None) as broadcast:
        returned = context.ep_broadcast(tensor, src_ep_rank=1)

    assert returned is tensor
    broadcast.assert_called_once_with(
        tensor,
        src=1,
        group=context.expert.process_group,
    )


def test_ep_broadcast_rejects_invalid_source_rank():
    context = _replicated_ep_context()

    with pytest.raises(ValueError, match="EP broadcast source"):
        context.ep_broadcast(torch.tensor([1.0]), src_ep_rank=4)


@pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.MAX])
def test_parallel_context_collectives_are_always_in_place_torch_operations(op):
    world_group = object()
    context = ParallelContext(
        world=ParallelGroup(world_group, (0, 1, 2, 3), 0, 4),
        tensor=ParallelGroup(world_group, (0, 1, 2, 3), 0, 4),
        expert=ParallelGroup(None, (0,), 0, 1),
        data=ParallelGroup(None, (0,), 0, 1),
    )
    tensor = torch.ones(2, 3072, dtype=torch.bfloat16)

    with patch.object(dist, "all_reduce") as all_reduce:
        returned = context.world_all_reduce(tensor, op=op)

    assert returned is tensor
    all_reduce.assert_called_once_with(
        tensor,
        op=op,
        group=world_group,
    )


def test_qwen3_moe_parallel_config_validation(tmp_path):
    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        config = Config(model=str(tmp_path), expert_parallel_size=4)
    assert config.world_size == 4
    assert config.weight_loading_workers_per_rank == 1

    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        hybrid = Config(
            model=str(tmp_path), tensor_parallel_size=2, expert_parallel_size=2
        )
    assert hybrid.world_size == 2
    assert hybrid.attention_tensor_parallel_size == 2
    assert hybrid.moe_expert_parallel_size == 2
    assert hybrid.moe_tensor_parallel_size == 1

    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        config = Config(model=str(tmp_path), tensor_parallel_size=2)
    assert config.tensor_parallel_size == 2
    assert config.expert_parallel_size == 1

    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            Config(model=str(tmp_path), tensor_parallel_size=4)

    fp16 = _hf_config()
    fp16.torch_dtype = torch.float16
    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=fp16):
        with pytest.raises(NotImplementedError, match="outer TP supports BF16"):
            Config(model=str(tmp_path), tensor_parallel_size=2)

    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config()):
        with pytest.raises(ValueError, match="TP divisible by EP"):
            Config(model=str(tmp_path), tensor_parallel_size=3, expert_parallel_size=2)

    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config(num_experts=6)):
        with pytest.raises(ValueError, match="divisible"):
            Config(model=str(tmp_path), expert_parallel_size=4)

    invalid_layout = _hf_config()
    invalid_layout.decoder_sparse_step = 0
    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=invalid_layout):
        with pytest.raises(NotImplementedError, match="every decoder layer"):
            Config(model=str(tmp_path))

    invalid_dtype = _hf_config()
    invalid_dtype.torch_dtype = torch.float32
    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=invalid_dtype):
        with pytest.raises(NotImplementedError, match="BF16/FP16 checkpoints"):
            Config(model=str(tmp_path))


def test_qwen3_moe_fp8_config_validation(tmp_path):
    hf_config = _hf_config()
    hf_config.architectures = ["Qwen3MoeForCausalLM"]
    hf_config.hidden_size = 128
    hf_config.moe_intermediate_size = 128
    raw_quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "lm_head",
            "model.layers.0.mlp.gate",
            "model.layers.1.mlp.gate",
        ],
    }
    hf_config.quantization_config = raw_quantization_config
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        config = Config(model=str(tmp_path), expert_parallel_size=2)
    assert config.quantization_config.enabled
    assert config.quantization_config.model_name == "Qwen3MoE"

    hf_config.quantization_config = {
        **raw_quantization_config,
        "modules_to_not_convert": ["lm_head"],
    }
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        with pytest.raises(ValueError, match="router gate"):
            Config(model=str(tmp_path), expert_parallel_size=2)


def test_qwen3_dense_fp8_config_validation(tmp_path):
    hf_config = _hf_config("qwen3")
    hf_config.architectures = ["Qwen3ForCausalLM"]
    hf_config.hidden_size = 4096
    hf_config.intermediate_size = 12288
    hf_config.head_dim = 128
    hf_config.num_attention_heads = 32
    hf_config.num_key_value_heads = 8
    hf_config.quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        config = Config(model=str(tmp_path), tensor_parallel_size=8)
    assert config.quantization_config.enabled
    assert config.quantization_config.model_name == "Qwen3"
    assert config.quantization_config.activation_dtype == "bfloat16"

    hf_config.intermediate_size = 12160
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        with pytest.raises(ValueError, match="TP-local dense projection"):
            Config(model=str(tmp_path), tensor_parallel_size=8)


def test_qwen3_dense_fp8_rejects_wrong_architecture(tmp_path):
    hf_config = _hf_config("qwen3")
    hf_config.architectures = ["Qwen3MoeForCausalLM"]
    hf_config.quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        with pytest.raises(ValueError, match="Qwen3ForCausalLM"):
            Config(model=str(tmp_path))


def test_dense_config_rejects_expert_or_data_parallelism(tmp_path):
    with patch("sparsevllm.configs.runtime.AutoConfig.from_pretrained", return_value=_hf_config("qwen3")):
        with pytest.raises(ValueError, match="does not support expert parallelism"):
            Config(model=str(tmp_path), expert_parallel_size=2)


def test_dense_layers_use_tp_group_in_replicated_ep_topology():
    context = _replicated_ep_context()
    with (
        patch("sparsevllm.layers.linear.get_parallel_context", return_value=context),
        patch("sparsevllm.layers.embed_head.get_parallel_context", return_value=context),
    ):
        column = ColumnParallelLinear(8, 16)
        row = RowParallelLinear(8, 16)
        embedding = VocabParallelEmbedding(32, 8)

    assert column.weight.shape == (16, 8)
    assert row.weight.shape == (16, 8)
    assert embedding.weight.shape == (32, 8)


def test_vocab_parallel_embedding_reduces_results():
    reduced = torch.randn(2, 4)
    context = SimpleNamespace(
        tp_rank=0,
        tp_size=2,
        tp_all_reduce=Mock(return_value=reduced),
    )
    with patch(
        "sparsevllm.layers.embed_head.get_parallel_context",
        return_value=context,
    ):
        embedding = VocabParallelEmbedding(8, 4)

    output = embedding(torch.tensor([0, 5]))

    assert output is reduced
    context.tp_all_reduce.assert_called_once()


def test_cache_kv_heads_depend_on_tp_not_ep():
    context = _replicated_ep_context()
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=4,
            num_attention_heads=8,
            hidden_size=32,
            head_dim=4,
        ),
        runtime_layout=RuntimeLayout.dense(2),
        max_model_len=128,
        max_num_seqs_in_gpu=2,
        max_num_seqs_in_batch=2,
    )
    with patch.object(platforms, "_current_platform", CpuPlatform()):
        manager = _MinimalCacheManager(config, context)

    assert manager.world_size == 4
    assert manager.tp_size == 1
    assert manager.ep_size == 4
    assert manager.num_kv_heads == 4
