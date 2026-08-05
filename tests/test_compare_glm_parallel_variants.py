from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.validation.compare_glm_parallel_variants import compare_runs, main


def _write_run(
    directory: Path,
    *,
    logits: torch.Tensor,
    tokens: list[int],
    ep_size: int,
) -> None:
    directory.mkdir()
    summary = {
        "status": "success",
        "model_path": "/checkpoint",
        "seed": 7,
        "prompt_len": 4,
        "prompt_sha256": "prompt",
        "max_tokens": len(tokens),
        "sparse_method": "vanilla",
        "tensor_parallel_size": 2,
        "expert_parallel_size": ep_size,
        "generated_token_ids": tokens,
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    torch.save(
        {"prefill": logits, "final_decode": logits},
        directory / "logits.pt",
    )


def test_compare_parallel_runs_accepts_bounded_drift_and_reference_tie(tmp_path):
    reference = torch.tensor([[1.0, 3.0, 3.0, 0.0]])
    candidate = torch.tensor([[1.0, 2.5, 3.0, 0.0]])
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_run(reference_dir, logits=reference, tokens=[1, 3], ep_size=1)
    _write_run(candidate_dir, logits=candidate, tokens=[2, 3], ep_size=2)

    result = compare_runs(
        reference_dir,
        candidate_dir,
        max_abs_limit=0.5,
        mean_abs_limit=0.2,
        p99_abs_limit=0.5,
        allow_first_token_tie=True,
        tie_atol=0.0,
    )

    assert result["status"] == "success"
    assert result["generated_tokens"]["first_token_tie_accepted"] is True


def test_compare_parallel_runs_fails_when_drift_exceeds_limit(tmp_path):
    reference = torch.tensor([[1.0, 2.0]])
    candidate = torch.tensor([[1.0, 3.0]])
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_run(reference_dir, logits=reference, tokens=[1], ep_size=1)
    _write_run(candidate_dir, logits=candidate, tokens=[1], ep_size=2)

    result = compare_runs(
        reference_dir,
        candidate_dir,
        max_abs_limit=0.5,
        mean_abs_limit=1.0,
        p99_abs_limit=1.0,
        allow_first_token_tie=False,
        tie_atol=0.0,
    )

    assert result["status"] == "metric_failed"
    assert result["gates"]["prefill_logits"] is False


def test_cli_writes_invalid_input_status_before_failing(tmp_path, monkeypatch):
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_glm_parallel_variants.py",
            "--reference_dir",
            str(tmp_path / "missing-reference"),
            "--candidate_dir",
            str(tmp_path / "missing-candidate"),
            "--output",
            str(output),
            "--max_abs_diff",
            "1",
            "--mean_abs_diff",
            "1",
            "--p99_abs_diff",
            "1",
        ],
    )

    assert main() == 1
    assert json.loads(output.read_text())["status"] == "invalid_input"
