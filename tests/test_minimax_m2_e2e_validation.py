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
    actual["hidden_states"][0] += 1.0
    with pytest.raises(
        validation.MetricFailure, match="Greedy token mismatch"
    ) as error:
        validation._compare_case(
            torch,
            actual,
            expected,
            max_relative_l2=0.1,
            max_router_relative_l2=1.0e-4,
        )
    assert error.value.metrics["greedy_tokens"]["exact"] is False
    assert error.value.metrics["hidden_states"]["0"]["relative_l2_error"] > 0.1
    assert error.value.metrics["moe_states"]["0"]["topk_ids"]["exact"] is True


def test_e2e_validator_checks_prefix_hits_and_graph_key_reuse():
    graph_state = {
        "key": {
            "method": "",
            "batch_size": 1,
            "context_capacity": 1024,
            "is_long_text": False,
            "capture_sampling": False,
        },
        "captured": True,
        "num_graph_states": 1,
    }
    graph_replays = [graph_state, graph_state]
    results = {
        "prefix_warm": {
            "prefix_cache_hit_lengths": [0],
            "graph_states": graph_replays,
        },
        "prefix_exact": {
            "prefix_cache_hit_lengths": [112],
            "graph_states": graph_replays,
        },
        "prefix_partial": {
            "prefix_cache_hit_lengths": [64],
            "graph_states": graph_replays,
        },
        "prefix_no_hit": {
            "prefix_cache_hit_lengths": [0],
            "graph_states": graph_replays,
        },
        "prefix_shared": {
            "prefix_cache_hit_lengths": [64, 64],
            "graph_states": graph_replays,
        },
    }
    summary = validation._validate_prefix_patterns(results, use_graph=True)
    assert summary["exact_hit_tokens"] == [112]

    results["prefix_partial"]["graph_states"] = [
        {
            **graph_state,
            "key": {**graph_state["key"], "context_capacity": 2048},
        },
        graph_state,
    ]
    with pytest.raises(validation.MetricFailure, match="graph key"):
        validation._validate_prefix_patterns(results, use_graph=True)

    results["prefix_partial"]["graph_states"] = [graph_state]
    with pytest.raises(validation.MetricFailure, match="fewer than two"):
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


def test_e2e_validator_enables_debug_for_spawned_workers(monkeypatch):
    for key in validation.DEBUG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    validation._enable_debug_runtime()

    assert {
        key: validation.os.environ[key] for key in validation.DEBUG_ENV_KEYS
    } == {
        "SPARSEVLLM_DEBUG_RUNTIME": "1",
        "SPARSEVLLM_DEBUG_HIDDEN_LAYERS": "0,30,61",
        "SPARSEVLLM_DEBUG_MOE": "1",
        "SPARSEVLLM_DEBUG_MINIMAX_M2": "1",
    }


def test_e2e_validator_records_raw_before_parse_failure():
    raw_cases = {}
    parsed_outputs = {}
    raw_case = _raw_case()

    with patch.object(
        validation,
        "_parsed_case",
        side_effect=ValueError("parse failed"),
    ):
        with pytest.raises(validation.ParseFailure, match="case-0") as error:
            validation._record_case_artifacts(
                "case-0",
                raw_case,
                raw_cases,
                parsed_outputs,
            )

    assert raw_cases == {"case-0": raw_case}
    assert parsed_outputs == {}
    assert isinstance(error.value.__cause__, ValueError)
