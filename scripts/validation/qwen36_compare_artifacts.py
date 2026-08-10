from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


THRESHOLDS = {
    "forced_logits_max_abs": 4.0,
    "forced_logits_mean_abs": 0.6,
    "near_tie_margin": 0.25,
    "decoder_layer_cosine": 0.75,
    "routing_layer_overlap": 0.90,
    "routing_mean_overlap": 0.95,
    "graph_eager_logits_max_abs": 0.0,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"Expected an object at {path}:{line_no}.")
        rows.append(row)
    return rows


def _load_tensor_list(path: Path) -> list[Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, list):
        raise TypeError(f"Expected a tensor list in {path}, got {type(value).__name__}.")
    return value


def _require_run(path: Path) -> None:
    required = (
        "run_info.json",
        "runtime_status.json",
        "per_sample_results.jsonl",
        "aggregate_metrics.json",
        "raw_logits.pt",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Validation run {path} is missing artifacts: {missing}.")
    aggregate = _read_json(path / "aggregate_metrics.json")
    if aggregate.get("status") != "success":
        raise RuntimeError(f"Validation run {path} is not successful: {aggregate!r}.")


def _tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    if actual.shape != reference.shape:
        raise ValueError(
            f"Tensor shape mismatch: actual={tuple(actual.shape)}, reference={tuple(reference.shape)}."
        )
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    difference = (actual_fp32 - reference_fp32).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "cosine": float(
            F.cosine_similarity(actual_fp32.flatten(), reference_fp32.flatten(), dim=0).item()
        ),
    }


def _output_ids(path: Path) -> list[list[int]]:
    rows = _read_jsonl(path / "per_sample_results.jsonl")
    if any(row.get("status") != "success" for row in rows):
        raise RuntimeError(f"Non-success sample found in {path}.")
    return [[int(token) for token in row["output_token_ids"]] for row in rows]


def _graph_records(
    label: str,
    eager_dir: Path,
    graph_dir: Path,
) -> list[dict[str, Any]]:
    eager_logits = _load_tensor_list(eager_dir / "raw_logits.pt")
    graph_logits = _load_tensor_list(graph_dir / "raw_logits.pt")
    if len(eager_logits) != len(graph_logits):
        raise ValueError(f"{label} eager/Graph sample counts differ.")
    max_abs = max(
        _tensor_metrics(graph, eager)["max_abs"]
        for eager, graph in zip(eager_logits, graph_logits)
    )
    token_ids_equal = _output_ids(eager_dir) == _output_ids(graph_dir)
    runtime = _read_json(graph_dir / "runtime_status.json")
    workers = runtime.get("worker_runtime_status")
    if not isinstance(workers, list) or not workers:
        raise RuntimeError(f"{graph_dir} has no worker_runtime_status records.")
    all_graph_active = all(
        worker.get("decode_cuda_graph_configured") is True
        and worker.get("decode_cuda_graph_active") is True
        for worker in workers
    )
    providers = {
        (worker.get("moe_expert_provider"), worker.get("moe_router_provider"))
        for worker in workers
    }
    status = (
        "success"
        if max_abs <= THRESHOLDS["graph_eager_logits_max_abs"]
        and token_ids_equal
        and all_graph_active
        and providers == {("triton", "triton")}
        else "metric_failed"
    )
    return [
        {
            "check": "graph_eager_equivalence",
            "topology": label,
            "status": status,
            "max_abs": max_abs,
            "token_ids_equal": token_ids_equal,
            "all_graph_active": all_graph_active,
            "worker_count": len(workers),
            "providers": sorted([list(pair) for pair in providers]),
        }
    ]


def _forced_reference_records(
    label: str,
    reference_dir: Path,
    actual_dir: Path,
) -> list[dict[str, Any]]:
    reference_logits = _load_tensor_list(reference_dir / "raw_logits.pt")
    actual_logits = _load_tensor_list(actual_dir / "raw_logits.pt")
    reference_hidden = _load_tensor_list(
        reference_dir / "raw_cached_hidden_states.pt"
    )
    actual_hidden = _load_tensor_list(actual_dir / "raw_cached_hidden_states.pt")
    if not (
        len(reference_logits)
        == len(actual_logits)
        == len(reference_hidden)
        == len(actual_hidden)
    ):
        raise ValueError(f"{label} forced-reference sample counts differ.")

    records: list[dict[str, Any]] = []
    for sample_idx, (reference, actual) in enumerate(
        zip(reference_logits, actual_logits)
    ):
        metrics = _tensor_metrics(actual, reference)
        reference_top2 = torch.topk(reference.float(), k=2)
        actual_top1 = int(torch.argmax(actual).item())
        reference_top1 = int(reference_top2.indices[0].item())
        reference_margin = float(
            (reference_top2.values[0] - reference_top2.values[1]).item()
        )
        top1_acceptable = actual_top1 == reference_top1 or (
            reference_margin <= THRESHOLDS["near_tie_margin"]
            and actual_top1 in {int(index) for index in reference_top2.indices.tolist()}
        )
        status = (
            "success"
            if metrics["max_abs"] <= THRESHOLDS["forced_logits_max_abs"]
            and metrics["mean_abs"] <= THRESHOLDS["forced_logits_mean_abs"]
            and top1_acceptable
            else "metric_failed"
        )
        records.append(
            {
                "check": "forced_prefix_logits",
                "topology": label,
                "sample_idx": sample_idx,
                "status": status,
                **metrics,
                "reference_top1": reference_top1,
                "actual_top1": actual_top1,
                "reference_top1_margin": reference_margin,
                "top1_acceptable": top1_acceptable,
            }
        )

        reference_layers = reference_hidden[sample_idx]
        actual_layers = actual_hidden[sample_idx]
        if set(reference_layers) != set(actual_layers):
            raise ValueError(f"{label} sample {sample_idx} hidden layer sets differ.")
        for layer_idx in sorted(reference_layers):
            layer_metrics = _tensor_metrics(
                actual_layers[layer_idx], reference_layers[layer_idx]
            )
            records.append(
                {
                    "check": "decoder_layer_output",
                    "topology": label,
                    "sample_idx": sample_idx,
                    "layer_idx": int(layer_idx),
                    "layer_kind": (
                        "embedding"
                        if layer_idx == -1
                        else "final_norm"
                        if layer_idx == 40
                        else "full_attention"
                        if layer_idx % 4 == 3
                        else "gated_deltanet"
                    ),
                    "status": (
                        "success"
                        if layer_metrics["cosine"]
                        >= THRESHOLDS["decoder_layer_cosine"]
                        else "metric_failed"
                    ),
                    **layer_metrics,
                }
            )
    return records


def _routing_records(
    label: str,
    reference_dir: Path,
    actual_dir: Path,
) -> list[dict[str, Any]]:
    reference = _load_tensor_list(reference_dir / "raw_moe_states.pt")
    actual = _load_tensor_list(actual_dir / "raw_moe_states.pt")
    if len(reference) != len(actual):
        raise ValueError(f"{label} routing sample counts differ.")
    records = []
    overlaps = []
    for sample_idx, (reference_layers, actual_layers) in enumerate(
        zip(reference, actual)
    ):
        if set(reference_layers) != set(actual_layers):
            raise ValueError(f"{label} sample {sample_idx} MoE layer sets differ.")
        for layer_idx in sorted(reference_layers):
            reference_ids = reference_layers[layer_idx]["topk_ids"]
            actual_ids = actual_layers[layer_idx]["topk_ids"]
            if reference_ids.shape != actual_ids.shape:
                raise ValueError(f"{label} sample {sample_idx} layer {layer_idx} shape differs.")
            row_overlaps = [
                len(set(left.tolist()) & set(right.tolist())) / reference_ids.shape[1]
                for left, right in zip(reference_ids, actual_ids)
            ]
            overlap = float(sum(row_overlaps) / len(row_overlaps))
            overlaps.append(overlap)
            records.append(
                {
                    "check": "cross_topology_routing",
                    "topology": label,
                    "sample_idx": sample_idx,
                    "layer_idx": int(layer_idx),
                    "status": (
                        "success"
                        if overlap >= THRESHOLDS["routing_layer_overlap"]
                        else "metric_failed"
                    ),
                    "topk_set_overlap": overlap,
                    "ordered_ids_equal": bool(torch.equal(reference_ids, actual_ids)),
                }
            )
    mean_overlap = float(sum(overlaps) / len(overlaps))
    records.append(
        {
            "check": "cross_topology_routing_aggregate",
            "topology": label,
            "status": (
                "success"
                if mean_overlap >= THRESHOLDS["routing_mean_overlap"]
                else "metric_failed"
            ),
            "mean_topk_set_overlap": mean_overlap,
            "min_layer_topk_set_overlap": min(overlaps),
        }
    )
    return records


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed Qwen3.6 MoE correctness artifacts."
    )
    parser.add_argument("--transformers", type=Path, required=True)
    for name in (
        "single-eager",
        "single-graph",
        "tp-eager",
        "tp-graph",
        "tp-ep-eager",
        "tp-ep-graph",
        "forced-single",
        "forced-tp",
        "forced-tp-ep",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        key: Path(value).resolve()
        for key, value in vars(args).items()
        if key != "output_dir"
    }
    for path in sources.values():
        _require_run(path)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    records = []
    records += _graph_records("single", sources["single_eager"], sources["single_graph"])
    records += _graph_records("pure_tp", sources["tp_eager"], sources["tp_graph"])
    records += _graph_records("tp_ep", sources["tp_ep_eager"], sources["tp_ep_graph"])
    for label, key in (
        ("single", "forced_single"),
        ("pure_tp", "forced_tp"),
        ("tp_ep", "forced_tp_ep"),
    ):
        records += _forced_reference_records(
            label, sources["transformers"], sources[key]
        )
    records += _routing_records(
        "single_vs_pure_tp", sources["forced_single"], sources["forced_tp"]
    )
    records += _routing_records(
        "single_vs_tp_ep", sources["forced_single"], sources["forced_tp_ep"]
    )

    failed = [record for record in records if record["status"] != "success"]
    aggregate = {
        "status": "success" if not failed else "metric_failed",
        "num_checks": len(records),
        "success_checks": len(records) - len(failed),
        "failed_checks": len(failed),
        "thresholds": THRESHOLDS,
    }
    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip(),
        "sources": {key: str(value) for key, value in sources.items()},
        "thresholds": THRESHOLDS,
    }
    _write_json(args.output_dir / "run_info.json", run_info)
    _write_json(args.output_dir / "raw_outputs.json", run_info["sources"])
    _write_json(args.output_dir / "parsed_outputs.json", {"checks": records})
    with (args.output_dir / "per_sample_results.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json(args.output_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
