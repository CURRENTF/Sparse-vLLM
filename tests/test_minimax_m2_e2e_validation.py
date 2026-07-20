from unittest.mock import patch

import pytest
import torch

from scripts.validation import validate_minimax_m2_e2e as validation


def _raw_case():
    return {
        "generated_token_ids": [[1, 2]],
        "logits": torch.tensor([[1.0, 2.0]]),
        "hidden_states": {0: torch.tensor([[0.5, -0.5]])},
        "attention_states": {
            0: {
                "q_norm_rope": torch.ones(1, 1, 2),
                "k_raw": torch.ones(1, 1, 2),
                "k_norm_rope": torch.ones(1, 1, 2),
                "value": torch.ones(1, 1, 2),
            }
        },
        "moe_states": {
            0: {
                "input": torch.ones(1, 2),
                "router_logits": torch.ones(1, 4),
                "topk_ids": torch.tensor([[1, 3]]),
                "topk_weights": torch.tensor([[0.4, 0.6]]),
                "output": torch.ones(1, 2),
            }
        },
    }


def test_e2e_validator_compares_tokens_router_and_tensor_groups():
    actual = _raw_case()
    expected = _raw_case()

    metrics = validation._compare_case(
        torch,
        actual,
        expected,
        max_relative_l2=0.1,
        max_router_relative_l2=1.0e-4,
    )
    assert metrics["logits"]["relative_l2_error"] == 0.0

    actual["generated_token_ids"] = [[2, 1]]
    with pytest.raises(validation.MetricFailure, match="Greedy token mismatch"):
        validation._compare_case(
            torch,
            actual,
            expected,
            max_relative_l2=0.1,
            max_router_relative_l2=1.0e-4,
        )


def test_e2e_validator_checks_prefix_hits_and_graph_key_reuse():
    graph_state = {
        "key": {
            "method": "",
            "batch_size": 1,
            "context_capacity": 1024,
            "is_long_text": False,
            "capture_sampling": False,
        }
    }
    results = {
        "prefix_warm": {
            "prefix_cache_hit_lengths": [0],
            "graph_states": [graph_state],
        },
        "prefix_exact": {
            "prefix_cache_hit_lengths": [112],
            "graph_states": [graph_state],
        },
        "prefix_partial": {
            "prefix_cache_hit_lengths": [64],
            "graph_states": [graph_state],
        },
        "prefix_no_hit": {
            "prefix_cache_hit_lengths": [0],
            "graph_states": [graph_state],
        },
        "prefix_shared": {
            "prefix_cache_hit_lengths": [64, 64],
            "graph_states": [graph_state],
        },
    }
    summary = validation._validate_prefix_patterns(results, use_graph=True)
    assert summary["exact_hit_tokens"] == [112]

    results["prefix_partial"]["graph_states"][0] = {
        "key": {**graph_state["key"], "context_capacity": 2048}
    }
    with pytest.raises(validation.MetricFailure, match="graph key"):
        validation._validate_prefix_patterns(results, use_graph=True)


def test_e2e_validator_refuses_busy_or_incomplete_gpu_selection():
    devices = [
        {
            "index": index,
            "name": f"gpu-{index}",
            "memory_used_mib": 1 if index else 2048,
            "memory_total_mib": 96_000,
            "utilization_percent": 0,
        }
        for index in range(4)
    ]
    with patch.object(validation, "_query_gpus", return_value=devices):
        with pytest.raises(ValueError, match="exactly EP=4"):
            validation._validate_gpu_selection(
                [0, 1],
                ep_size=4,
                max_memory_used_mib=512,
                max_utilization_percent=5,
            )
        with pytest.raises(RuntimeError, match="busy"):
            validation._validate_gpu_selection(
                [0, 1, 2, 3],
                ep_size=4,
                max_memory_used_mib=512,
                max_utilization_percent=5,
            )


def test_e2e_validator_records_checkpoint_manifest(tmp_path):
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    manifest = validation._checkpoint_manifest(tmp_path)

    assert manifest["num_safetensor_shards"] == 1
    assert manifest["safetensor_bytes"] == len(b"weights")
    assert len(manifest["config_sha256"]) == 64


def test_e2e_validator_requires_ep_replica_consistency():
    summary = {
        "world_rank": 0,
        "replica_consistency": {
            "last_logits_tolerance_ratio": 0.0,
            "moe_layers": {
                "0": {
                    "topk_ids_mismatch": False,
                    "topk_weights_tolerance_ratio": 0.0,
                    "output_tolerance_ratio": 0.0,
                }
            },
        },
    }
    validation._validate_replica_consistency([summary])

    summary["replica_consistency"]["moe_layers"]["0"][
        "topk_ids_mismatch"
    ] = True
    with pytest.raises(validation.MetricFailure, match="router IDs"):
        validation._validate_replica_consistency([summary])
