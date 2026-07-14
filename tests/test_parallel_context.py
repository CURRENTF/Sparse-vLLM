from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import sparsevllm.platforms as platforms
from sparsevllm.config import Config
from sparsevllm.distributed import ParallelContext, ParallelGroup
from sparsevllm.distributed.parallel_context import (
    get_parallel_context,
    init_parallel_context,
    parallel_group_ranks,
    parallel_ranks_from_world_rank,
    reset_parallel_context,
    world_rank_from_parallel_ranks,
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
        num_experts=num_experts,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


def test_world_rank_mapping_round_trips():
    for world_rank in range(24):
        ranks = parallel_ranks_from_world_rank(
            world_rank,
            tp_size=2,
            ep_size=3,
            dp_size=4,
        )
        assert world_rank_from_parallel_ranks(
            *ranks,
            tp_size=2,
            ep_size=3,
            dp_size=4,
        ) == world_rank


def test_parallel_group_members_follow_dp_ep_tp_layout():
    assert parallel_group_ranks(tp_size=2, ep_size=2, dp_size=2) == {
        "tensor": ((0, 1), (2, 3), (4, 5), (6, 7)),
        "expert": ((0, 2), (1, 3), (4, 6), (5, 7)),
        "data": ((0, 4), (1, 5), (2, 6), (3, 7)),
    }


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
        patch.object(dist, "new_group", side_effect=new_group),
    ):
        context = init_parallel_context(tp_size=1, ep_size=2, dp_size=2)
        assert context.world_rank == 2
        assert context.tp_rank == 0
        assert context.tp_size == 1
        assert context.ep_rank == 0
        assert context.expert.ranks == (2, 3)
        assert context.dp_rank == 1
        assert context.data.ranks == (0, 2)
        assert get_parallel_context() is context
        with pytest.raises(RuntimeError, match="already initialized"):
            init_parallel_context(tp_size=1, ep_size=2, dp_size=2)

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
            init_parallel_context(tp_size=1, ep_size=4, dp_size=1)


def test_qwen3_moe_parallel_config_validation(tmp_path):
    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config()):
        config = Config(model=str(tmp_path), expert_parallel_size=4)
    assert config.world_size == 4
    assert config.moe_backend == "triton"

    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config()):
        with pytest.raises(ValueError, match="moe_backend must be"):
            Config(model=str(tmp_path), moe_backend="automatic")

    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config()):
        with pytest.raises(ValueError, match="only supports TP=1 and DP=1"):
            Config(model=str(tmp_path), tensor_parallel_size=2, expert_parallel_size=2)

    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config(num_experts=6)):
        with pytest.raises(ValueError, match="divisible"):
            Config(model=str(tmp_path), expert_parallel_size=4)


def test_dense_config_rejects_expert_or_data_parallelism(tmp_path):
    with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config("qwen3")):
        with pytest.raises(ValueError, match="requires EP=1 and DP=1"):
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
        runtime_layout=None,
        max_model_len=128,
        max_num_seqs_in_batch=2,
    )
    with patch.object(platforms, "_current_platform", CpuPlatform()):
        manager = _MinimalCacheManager(config, context)

    assert manager.world_size == 4
    assert manager.tp_size == 1
    assert manager.ep_size == 4
    assert manager.num_kv_heads == 4
