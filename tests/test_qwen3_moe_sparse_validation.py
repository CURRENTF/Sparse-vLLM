from types import SimpleNamespace

import torch

from scripts.validation.validate_qwen3_moe_sparse_ep import (
    _cache_max_row_len,
    _compare_reference,
    _rank_sync_error,
    _validate_method_trigger,
)


def _summary(rank: int, start: int, end: int):
    synced = {"sha256": "same"}
    return {
        "world_rank": rank,
        "ep_rank": rank,
        "state": synced,
        "last_logits": synced,
        "moe_synced": synced,
        "moe_local": {
            "0": {
                "local_expert_start": start,
                "local_expert_end": end,
            }
        },
        "replica_consistency": {
            "last_logits_max_abs": 0.0,
            "last_logits_tolerance_ratio": 0.0,
            "moe_layers": {
                "0": {
                    "topk_ids_mismatch": False,
                    "topk_weights_max_abs": 0.0,
                    "topk_weights_tolerance_ratio": 0.0,
                    "output_max_abs": 0.0,
                    "output_tolerance_ratio": 0.0,
                }
            },
        },
    }


def test_rank_sync_accepts_replicated_state_with_sharded_experts():
    assert _rank_sync_error([_summary(0, 0, 64), _summary(1, 64, 128)]) is None


def test_rank_sync_rejects_replicated_state_divergence():
    summaries = [_summary(0, 0, 64), _summary(1, 64, 128)]
    summaries[1]["state"] = {"sha256": "different"}

    assert "diverged" in _rank_sync_error(summaries)


def test_cache_row_length_reads_live_rows():
    summary = {
        "state": {
            "cache": {
                "live_rows": {
                    "0": [{"seq_id": 1, "row_idx": 0, "row_len": 52}]
                }
            }
        }
    }

    assert _cache_max_row_len(summary) == 52


def test_method_trigger_requires_profile_and_compression_boundary():
    args = SimpleNamespace(method="rkv")
    profiler_stats = {"rkv_decode_eviction": {"calls": 1}}
    per_step = [{"cache_max_row_len": 52}]

    assert _validate_method_trigger(args, profiler_stats, per_step) == []


def _raw_step():
    return {
        "case_name": "primary",
        "step_idx": 0,
        "stage": "prefill",
        "logits": torch.tensor([[1.0, 2.0]]),
        "hidden_states": {-1: torch.tensor([[0.5, 1.5]])},
        "moe_states": {
            0: {
                "input": torch.tensor([[0.25, 0.75]]),
                "topk_ids": torch.tensor([[1, 3]]),
                "topk_weights": torch.tensor([[0.6, 0.4]]),
                "output": torch.tensor([[0.1, 0.2]]),
            }
        },
        "sampled_token_outputs": [(0, [7])],
    }


def _request():
    return {
        "generated_token_ids": [[7]],
        "prefix_cache_hit_tokens": [0],
    }


def test_reference_comparison_enforces_all_layer_moe_state(tmp_path):
    reference = {"steps": [_raw_step()], "requests": [_request()]}
    path = tmp_path / "raw_outputs.pt"
    torch.save(reference, path)

    metrics = _compare_reference(
        path,
        raw_steps=[_raw_step()],
        requests=[_request()],
        atol=0.05,
        rtol=0.05,
    )

    assert metrics["status"] == "success"
    assert metrics["steps"][0]["first_topk_ids_mismatch"] is None


def test_reference_comparison_rejects_router_id_mismatch(tmp_path):
    reference = {"steps": [_raw_step()], "requests": [_request()]}
    path = tmp_path / "raw_outputs.pt"
    torch.save(reference, path)
    actual = _raw_step()
    actual["moe_states"][0]["topk_ids"][0, 0] = 2

    metrics = _compare_reference(
        path,
        raw_steps=[actual],
        requests=[_request()],
        atol=0.05,
        rtol=0.05,
    )

    assert metrics["status"] == "metric_failed"
    assert metrics["steps"][0]["first_topk_ids_mismatch"] == 0
