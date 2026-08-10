from pathlib import Path

import pytest

from scripts.validation.qwen36_compare_longbench import (
    _validate_worker_providers,
)


def test_fp8_longbench_accepts_registered_graph_providers():
    workers = [
        {
            "moe_expert_provider": "flashinfer_cutlass_fp8_sm90",
            "moe_router_provider": "triton",
            "moe_weight_dtype": "torch.float8_e4m3fn",
            "fp8_linear_provider": "flashinfer_sm90",
        },
        {
            "moe_expert_provider": "triton",
            "moe_router_provider": "triton",
            "moe_weight_dtype": "torch.float8_e4m3fn",
            "fp8_linear_provider": "flashinfer_sm90",
        },
    ]

    _validate_worker_providers(
        workers,
        precision="fp8",
        path=Path("fp8-run"),
    )


def test_fp8_longbench_rejects_missing_fp8_diagnostics():
    workers = [
        {
            "moe_expert_provider": "triton",
            "moe_router_provider": "triton",
        }
    ]

    with pytest.raises(RuntimeError, match="invalid FP8 providers"):
        _validate_worker_providers(
            workers,
            precision="fp8",
            path=Path("fp8-run"),
        )
