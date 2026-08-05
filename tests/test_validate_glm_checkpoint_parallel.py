from __future__ import annotations

import pytest

from scripts.validation.validate_glm_checkpoint_parallel import (
    _validate_parallel_evidence,
)


def _graph_summary(enabled: bool = False) -> dict:
    return {
        "enabled": enabled,
        "capture_count": 1 if enabled else 0,
        "replay_count": 2 if enabled else 0,
        "eager_static_count": 0,
        "force_eager_count": 0,
    }


def _ep4_summary(world_rank: int, local_hit_count: int) -> dict:
    layer_key = "46"
    local_width = 16
    return {
        "world_rank": world_rank,
        "parallel": {
            "configured": {
                "tensor_parallel_size": 1,
                "expert_parallel_size": 4,
                "data_parallel_size": 1,
                "world_size": 4,
            },
            "effective": {
                "world": {
                    "rank": world_rank,
                    "size": 4,
                    "ranks": [0, 1, 2, 3],
                },
                "attention": {
                    "rank": 0,
                    "size": 1,
                    "ranks": [world_rank],
                },
                "expert": {
                    "rank": world_rank,
                    "size": 4,
                    "ranks": [0, 1, 2, 3],
                },
                "moe_tensor": {
                    "rank": 0,
                    "size": 1,
                    "ranks": [world_rank],
                },
                "data": {
                    "rank": 0,
                    "size": 1,
                    "ranks": [world_rank],
                },
            },
            "attention_replicated_for_ep": True,
        },
        "moe_local": {
            layer_key: {
                "local_expert_start": world_rank * local_width,
                "local_expert_end": (world_rank + 1) * local_width,
                "local_hit_count": local_hit_count,
            }
        },
        "replica_consistency": {
            "last_logits_comparison": "compared",
            "last_logits_tolerance_ratio": 0.0,
            "moe_layers": {
                layer_key: {
                    "topk_ids_mismatch": False,
                    "topk_weights_tolerance_ratio": 0.0,
                    "output_tolerance_ratio": 0.0,
                }
            },
        },
        "decode_cuda_graph": _graph_summary(),
    }


def test_ep_prefill_requires_every_local_expert_shard_to_execute():
    summaries = [
        _ep4_summary(rank, hit_count)
        for rank, hit_count in enumerate([0, 80, 84, 92])
    ]

    with pytest.raises(RuntimeError, match="did not execute a local routed expert"):
        _validate_parallel_evidence(
            summaries,
            tensor_parallel_size=1,
            expert_parallel_size=4,
            last_moe_layer=46,
        )


def test_ep_single_decode_keeps_zero_hit_as_evidence_without_rejecting_it():
    summaries = [
        _ep4_summary(rank, hit_count)
        for rank, hit_count in enumerate([0, 1, 1, 2])
    ]

    evidence = _validate_parallel_evidence(
        summaries,
        tensor_parallel_size=1,
        expert_parallel_size=4,
        last_moe_layer=46,
        require_all_ep_ranks_hit=False,
    )

    assert evidence["local_expert_hit_counts"] == [0, 1, 1, 2]
    assert evidence["all_ep_ranks_hit"] is False


def _hybrid_tp4_ep2_summary(world_rank: int) -> dict:
    ep_rank = 0 if world_rank < 2 else 1
    expert_ranks = [0, 2] if world_rank % 2 == 0 else [1, 3]
    moe_tensor_ranks = [0, 1] if world_rank < 2 else [2, 3]
    layer_key = "46"
    return {
        "world_rank": world_rank,
        "parallel": {
            "configured": {
                "tensor_parallel_size": 4,
                "expert_parallel_size": 2,
                "data_parallel_size": 1,
                "world_size": 4,
            },
            "effective": {
                "world": {
                    "rank": world_rank,
                    "size": 4,
                    "ranks": [0, 1, 2, 3],
                },
                "attention": {
                    "rank": world_rank,
                    "size": 4,
                    "ranks": [0, 1, 2, 3],
                },
                "expert": {
                    "rank": expert_ranks.index(world_rank),
                    "size": 2,
                    "ranks": expert_ranks,
                },
                "moe_tensor": {
                    "rank": moe_tensor_ranks.index(world_rank),
                    "size": 2,
                    "ranks": moe_tensor_ranks,
                },
                "data": {
                    "rank": 0,
                    "size": 1,
                    "ranks": [world_rank],
                },
            },
            "attention_replicated_for_ep": False,
        },
        "moe_local": {
            layer_key: {
                "local_expert_start": ep_rank * 32,
                "local_expert_end": (ep_rank + 1) * 32,
                "local_hit_count": 8,
            }
        },
        "replica_consistency": {
            "last_logits_comparison": "not_applicable_tp_vocab_sharded",
            "last_logits_tolerance_ratio": None,
            "moe_layers": {
                layer_key: {
                    "topk_ids_mismatch": False,
                    "topk_weights_tolerance_ratio": 0.0,
                    "output_tolerance_ratio": 0.0,
                }
            },
        },
        "decode_cuda_graph": _graph_summary(enabled=True),
    }


def test_hybrid_tp_ep_validation_checks_groups_and_graph_on_every_rank():
    evidence = _validate_parallel_evidence(
        [_hybrid_tp4_ep2_summary(rank) for rank in range(4)],
        tensor_parallel_size=4,
        expert_parallel_size=2,
        last_moe_layer=46,
        require_decode_cuda_graph=True,
        require_decode_cuda_graph_execution=True,
    )

    assert evidence["hybrid_moe"] is True
    assert evidence["local_expert_ranges"] == [
        [0, 32],
        [0, 32],
        [32, 64],
        [32, 64],
    ]
    assert evidence["expected_groups"]["moe_tensor"] == [[0, 1], [2, 3]]
