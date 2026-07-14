from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)
FINAL_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{16,}")


class RunnerError(RuntimeError):
    """Raised when an evaluation boundary cannot be validated."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunnerError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerError(f"Invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {"path": str(repo), "commit": None, "dirty": None}

    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            text=True,
            capture_output=True,
        )
        return proc.stdout.strip()

    return {
        "path": str(repo.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--short")),
    }


def _load_dataset(dataset: str, split: str) -> list[dict[str, Any]]:
    try:
        from swebench.harness.utils import load_swebench_dataset
    except ImportError as exc:
        raise RunnerError(
            "swebench is not installed in this Python environment. Run the adapter "
            "with the environment that provides mini-SWE-agent and SWE-bench."
        ) from exc

    rows = list(load_swebench_dataset(dataset, split))
    if not rows:
        raise RunnerError(f"Dataset {dataset!r} split {split!r} is empty")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
            raise RunnerError(f"Dataset row {index} has no string instance_id")
    return rows


def _parse_slice(value: str | None, length: int) -> slice:
    if not value:
        return slice(0, length)
    match = re.fullmatch(r"(\d*):(\d*)", value)
    if match is None:
        raise RunnerError("--slice must use START:STOP with non-negative integers")
    start = int(match.group(1)) if match.group(1) else 0
    stop = int(match.group(2)) if match.group(2) else length
    if start > stop or stop > length:
        raise RunnerError(f"Invalid --slice {value!r} for {length} selected instances")
    return slice(start, stop)


def select_rows(
    rows: Sequence[dict[str, Any]],
    *,
    instance_ids_file: Path | None,
    instance_filter: str | None,
    slice_spec: str | None,
) -> list[dict[str, Any]]:
    by_id = {row["instance_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RunnerError("Dataset contains duplicate instance_id values")

    if instance_ids_file is not None:
        try:
            requested = [
                line.strip()
                for line in instance_ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except FileNotFoundError as exc:
            raise RunnerError(f"Instance id file does not exist: {instance_ids_file}") from exc
        if not requested:
            raise RunnerError(f"Instance id file is empty: {instance_ids_file}")
        if len(set(requested)) != len(requested):
            raise RunnerError(f"Instance id file contains duplicates: {instance_ids_file}")
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise RunnerError(f"Unknown instance ids: {unknown[:10]}")
        selected = [by_id[instance_id] for instance_id in requested]
    else:
        selected = list(rows)

    if instance_filter:
        try:
            pattern = re.compile(instance_filter)
        except re.error as exc:
            raise RunnerError(f"Invalid --filter regex: {exc}") from exc
        selected = [row for row in selected if pattern.search(row["instance_id"])]

    selected = selected[_parse_slice(slice_spec, len(selected))]
    if not selected:
        raise RunnerError("Instance selection is empty")
    return selected


def _instance_image_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as exc:
        raise RunnerError("Cannot import SWE-bench test spec support") from exc
    return [make_test_spec(row, namespace="swebench").instance_image_key for row in rows]


def _require_local_images(image_names: Sequence[str]) -> None:
    if not image_names:
        raise RunnerError("No Docker images were derived for the selected instances")
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", *image_names],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RunnerError("docker is not installed or is not on PATH") from exc
    if proc.returncode == 0:
        return

    missing = []
    for image in image_names:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            missing.append(image)
    preview = "\n".join(missing[:10])
    raise RunnerError(
        f"{len(missing)} selected SWE-bench Docker images are missing. The adapter "
        f"does not pull images automatically. First missing images:\n{preview}"
    )


def _validate_local_server_manifest(
    path: Path,
    *,
    api_base: str,
    served_model_name: str,
) -> dict[str, Any]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise RunnerError(f"Server manifest must be a JSON object: {path}")
    required = {
        "command",
        "model_path",
        "served_model_name",
        "cuda_visible_devices",
        "server_port",
        "engine_kwargs",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RunnerError(f"Server manifest is missing fields: {missing}")
    for field in ("command", "model_path", "served_model_name", "cuda_visible_devices"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise RunnerError(f"server manifest {field} must be a non-empty string")
    if not isinstance(manifest["engine_kwargs"], dict):
        raise RunnerError("server manifest engine_kwargs must be a JSON object")
    if SECRET_PATTERN.search(json.dumps(manifest, ensure_ascii=False)):
        raise RunnerError(
            "server manifest appears to contain an API key; store secrets only in the environment"
        )
    engine_kwargs = manifest["engine_kwargs"]
    if "max_model_len" not in engine_kwargs:
        raise RunnerError("server manifest engine_kwargs must record max_model_len")
    if not isinstance(engine_kwargs["max_model_len"], int) or engine_kwargs["max_model_len"] <= 0:
        raise RunnerError("server manifest max_model_len must be a positive integer")
    if not ({"sparse_method", "vllm_sparse_method"} & set(engine_kwargs)):
        raise RunnerError(
            "server manifest engine_kwargs must record sparse_method or vllm_sparse_method"
        )
    if manifest["served_model_name"] != served_model_name:
        raise RunnerError(
            "server manifest served_model_name does not match --served-model-name: "
            f"{manifest['served_model_name']!r} != {served_model_name!r}"
        )
    try:
        server_port = int(manifest["server_port"])
    except (TypeError, ValueError) as exc:
        raise RunnerError("server manifest server_port must be an integer") from exc
    parsed = urllib.parse.urlparse(api_base)
    if parsed.port is not None and server_port != parsed.port:
        raise RunnerError(
            f"server manifest port {manifest['server_port']} does not match API base {api_base}"
        )
    return manifest


def _models_url(api_base: str) -> str:
    return api_base.rstrip("/") + "/models"


def check_openai_server(
    api_base: str,
    *,
    api_key: str,
    served_model_name: str,
    timeout: float,
    use_environment_proxy: bool,
) -> None:
    request = urllib.request.Request(
        _models_url(api_base),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    opener = (
        urllib.request.build_opener()
        if use_environment_proxy
        else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RunnerError(
            f"OpenAI model health check failed for {_models_url(api_base)}: {exc}"
        ) from exc
    try:
        model_ids = {item["id"] for item in payload["data"]}
    except (KeyError, TypeError) as exc:
        raise RunnerError("OpenAI /models response does not contain data[].id") from exc
    if served_model_name not in model_ids:
        raise RunnerError(
            f"OpenAI server does not advertise {served_model_name!r}; available models: "
            f"{sorted(model_ids)}"
        )


def render_mini_config(
    *,
    step_limit: int,
    cost_limit: float,
    wall_time_limit_seconds: int,
    cost_tracking: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    api_base: str | None,
) -> str:
    lines = [
        "agent:",
        f"  step_limit: {step_limit}",
        f"  cost_limit: {cost_limit}",
        f"  wall_time_limit_seconds: {wall_time_limit_seconds}",
        "model:",
        f"  cost_tracking: {json.dumps(cost_tracking)}",
        "  model_kwargs:",
        "    drop_params: true",
        "    parallel_tool_calls: true",
        f"    max_tokens: {max_tokens}",
        f"    temperature: {temperature}",
        f"    top_p: {top_p}",
    ]
    if api_base:
        lines.append(f"    api_base: {json.dumps(api_base)}")
    lines.extend(["environment:", "  pull_timeout: 30", ""])
    return "\n".join(lines)


def _redact(line: str) -> str:
    return SECRET_PATTERN.sub("[redacted]", line)


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RunnerError(f"Failed to start command {shlex.join(command)}: {exc}") from exc
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = _redact(line)
            log.write(safe_line)
            log.flush()
            sys.stdout.write(safe_line)
            sys.stdout.flush()
        returncode = process.wait()
    if returncode != 0:
        raise RunnerError(
            f"Command exited with code {returncode}; see {log_path}: {shlex.join(command)}"
        )


def _trajectory_info(batch_dir: Path, instance_id: str) -> tuple[dict[str, Any] | None, str | None]:
    trajectory = batch_dir / instance_id / f"{instance_id}.traj.json"
    if not trajectory.exists():
        return None, None
    payload = _read_json(trajectory)
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        raise RunnerError(f"Trajectory has no info object: {trajectory}")
    return payload["info"], str(trajectory)


def validate_predictions(
    predictions: Any,
    expected_ids: Sequence[str],
    *,
    source: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(predictions, dict):
        raise RunnerError(f"Predictions must be a JSON object: {source}")
    expected = set(expected_ids)
    actual = set(predictions)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RunnerError(
            f"Prediction ids do not match {source}: missing={missing[:10]} extra={extra[:10]}"
        )
    for instance_id, prediction in predictions.items():
        if not isinstance(prediction, dict):
            raise RunnerError(f"Prediction for {instance_id} is not a JSON object")
        if prediction.get("instance_id") != instance_id:
            raise RunnerError(f"Prediction instance_id mismatch for {instance_id}")
        if not isinstance(prediction.get("model_patch"), str):
            raise RunnerError(f"Prediction model_patch is not a string for {instance_id}")
        if not isinstance(prediction.get("model_name_or_path"), str):
            raise RunnerError(f"Prediction model_name_or_path is not a string for {instance_id}")
    return predictions


def merge_batch_predictions(
    run_dir: Path,
    batches: Sequence[Sequence[str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    combined: dict[str, dict[str, Any]] = {}
    generation_rows: list[dict[str, Any]] = []
    for index, expected_ids in enumerate(batches):
        batch_name = f"batch_{index:03d}"
        batch_dir = run_dir / "batches" / batch_name
        predictions_path = batch_dir / "preds.json"
        predictions = validate_predictions(
            _read_json(predictions_path), expected_ids, source=predictions_path
        )
        overlap = sorted(set(combined) & set(predictions))
        if overlap:
            raise RunnerError(f"Duplicate predictions across batches: {overlap[:10]}")
        combined.update(predictions)

        for instance_id in expected_ids:
            prediction = predictions[instance_id]
            info, trajectory_path = _trajectory_info(batch_dir, instance_id)
            model_stats = (info or {}).get("model_stats") or {}
            if not isinstance(model_stats, dict):
                raise RunnerError(f"model_stats is not an object for {instance_id}")
            generation_rows.append(
                {
                    "instance_id": instance_id,
                    "exit_status": (info or {}).get("exit_status"),
                    "has_patch": bool(prediction["model_patch"]),
                    "model_patch_len": len(prediction["model_patch"]),
                    "model_stats": model_stats,
                    "trajectory_path": trajectory_path,
                }
            )
    return combined, generation_rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_results(
    *,
    expected_ids: Sequence[str],
    predictions: dict[str, dict[str, Any]],
    generation_rows: Sequence[dict[str, Any]],
    official_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation_by_id = {row["instance_id"]: row for row in generation_rows}
    if set(generation_by_id) != set(expected_ids):
        raise RunnerError("Generation rows do not exactly cover selected instances")

    report_total = official_report.get("total_instances")
    if report_total != len(expected_ids):
        raise RunnerError(
            f"Official report total_instances={report_total!r}, expected {len(expected_ids)}"
        )
    if official_report.get("submitted_instances") != len(expected_ids):
        raise RunnerError(
            "Official report submitted_instances does not match selected predictions"
        )
    completed = set(official_report.get("completed_ids", []))
    resolved = set(official_report.get("resolved_ids", []))
    unresolved = set(official_report.get("unresolved_ids", []))
    empty_patch = set(official_report.get("empty_patch_ids", []))
    errors = set(official_report.get("error_ids", []))
    if resolved & unresolved:
        raise RunnerError("Official report marks instances as both resolved and unresolved")
    if resolved | unresolved != completed:
        raise RunnerError(
            "Official report completed_ids do not match resolved_ids plus unresolved_ids"
        )
    if (completed & empty_patch) or (completed & errors) or (empty_patch & errors):
        raise RunnerError("Official report outcome id sets overlap")
    count_fields = {
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": len(empty_patch),
        "error_instances": len(errors),
    }
    for field, expected_count in count_fields.items():
        if official_report.get(field) != expected_count:
            raise RunnerError(
                f"Official report {field}={official_report.get(field)!r}, expected {expected_count}"
            )
    known = completed | empty_patch | errors
    unknown_report_ids = known - set(expected_ids)
    if unknown_report_ids:
        raise RunnerError(
            f"Official report contains unknown ids: {sorted(unknown_report_ids)[:10]}"
        )

    rows = []
    for instance_id in expected_ids:
        generation = generation_by_id[instance_id]
        if generation["trajectory_path"] is None:
            status = "parse_failed"
        elif instance_id in errors:
            status = "metric_failed"
        elif instance_id in completed:
            status = "success"
        elif instance_id in empty_patch or not generation["has_patch"]:
            status = "model_failed"
        else:
            status = "metric_failed"
        if status not in FINAL_STATUSES:
            raise AssertionError(f"Unknown final status: {status}")
        rows.append(
            {
                "instance_id": instance_id,
                "status": status,
                "resolved": instance_id in resolved if instance_id in completed else None,
                "official_outcome": (
                    "resolved"
                    if instance_id in resolved
                    else "unresolved"
                    if instance_id in unresolved
                    else "empty_patch"
                    if instance_id in empty_patch
                    else "error"
                    if instance_id in errors
                    else "missing"
                ),
                "generation_exit_status": generation["exit_status"],
                "has_patch": generation["has_patch"],
                "model_patch_len": generation["model_patch_len"],
                "model_stats": generation["model_stats"],
                "trajectory_path": generation["trajectory_path"],
                "prediction_model": predictions[instance_id]["model_name_or_path"],
            }
        )

    total_cost = sum(
        float((row.get("model_stats") or {}).get("instance_cost") or 0.0)
        for row in generation_rows
    )
    total_calls = sum(
        int((row.get("model_stats") or {}).get("api_calls") or 0)
        for row in generation_rows
    )
    summary = {
        "total_instances": len(expected_ids),
        "resolved_instances": len(resolved),
        "score": len(resolved) / len(expected_ids),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "generation_exit_status_counts": dict(
            sorted(
                Counter(row.get("exit_status") or "missing" for row in generation_rows).items()
            )
        ),
        "total_instance_cost": total_cost,
        "total_api_calls": total_calls,
        "official_report": official_report,
    }
    return rows, summary


class SweBenchLiteRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(__file__).resolve().parents[2]
        self.swe_bench_dir = args.swe_bench_dir.expanduser().resolve()
        self.run_dir = args.run_dir.expanduser().resolve()
        self.status_path = self.run_dir / "status.jsonl"
        self.run_config_path = self.run_dir / "run_config.json"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.invocations_path = self.run_dir / "invocations.jsonl"
        self.mini_config_path = self.run_dir / "mini_swe_agent_config.yaml"
        self.instances_path = self.run_dir / "instances.txt"
        self.images_path = self.run_dir / "images.txt"
        self.predictions_path = self.run_dir / "preds_all.json"
        self.generation_results_path = self.run_dir / "generation_results.jsonl"
        self.official_dir = self.run_dir / "official"
        self.extra_mini_configs = [path.expanduser().resolve() for path in args.mini_extra_config]
        self.extra_mini_config_records: list[dict[str, str]] = []
        self.extra_mini_config_snapshots: list[Path] = []
        self.run_id = args.run_id or f"swe_bench_lite_{self.run_dir.name}"
        if re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_id) is None:
            raise RunnerError(
                "--run-id may only contain letters, numbers, dot, underscore, and dash"
            )
        if args.step_limit <= 0 or args.wall_time_limit_seconds <= 0:
            raise RunnerError("step and wall-time limits must be positive")
        if args.max_tokens <= 0 or args.eval_timeout <= 0 or args.health_timeout <= 0:
            raise RunnerError(
                "token, evaluation-timeout, and health-timeout limits must be positive"
            )
        if args.batch_size <= 0 or args.mini_workers <= 0 or args.eval_workers <= 0:
            raise RunnerError("batch size and worker counts must be positive")
        if args.cost_limit < 0:
            raise RunnerError("--cost-limit must be non-negative")
        if not 0.0 <= args.temperature:
            raise RunnerError("--temperature must be non-negative")
        if not 0.0 < args.top_p <= 1.0:
            raise RunnerError("--top-p must be in (0, 1]")
        if args.cost_tracking == "ignore_errors" and args.cost_limit > 0:
            raise RunnerError(
                "A positive --cost-limit is not reliable with --cost-tracking=ignore_errors; "
                "set --cost-limit=0 or provide model cost metadata and use default tracking"
            )

        self.requires_model_api = args.stage in {"prepare", "generate", "all"}
        self.api_key = os.environ.get(args.api_key_env, "")
        if self.requires_model_api and not self.api_key:
            raise RunnerError(f"Required API key environment variable is empty: {args.api_key_env}")

        self.rows: list[dict[str, Any]] = []
        self.instance_ids: list[str] = []
        self.image_names: list[str] = []

    def _status(self, stage: str, status: str, detail: str) -> None:
        _append_jsonl(
            self.status_path,
            {"time": _now(), "stage": stage, "status": status, "detail": detail},
        )

    def _model_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if not self.args.api_proxy_from_environment:
            for key in PROXY_ENV_VARS:
                env.pop(key, None)
        no_proxy = ",".join(
            value for value in (env.get("NO_PROXY"), env.get("no_proxy")) if value
        )
        no_proxy_values = [item for item in no_proxy.split(",") if item]
        for host in sorted(LOCAL_HOSTS):
            if host not in no_proxy_values:
                no_proxy_values.append(host)
        env["NO_PROXY"] = ",".join(no_proxy_values)
        env["no_proxy"] = env["NO_PROXY"]
        if self.args.offline_dataset:
            env["HF_DATASETS_OFFLINE"] = "1"
            env["HF_HUB_OFFLINE"] = "1"
        else:
            env.pop("HF_DATASETS_OFFLINE", None)
            env.pop("HF_HUB_OFFLINE", None)
        env.pop("MSWEA_GLOBAL_CALL_LIMIT", None)
        env.pop("MSWEA_GLOBAL_COST_LIMIT", None)
        return env

    def _load_selection(self) -> None:
        if not self.swe_bench_dir.is_dir():
            raise RunnerError(f"SWE-bench checkout does not exist: {self.swe_bench_dir}")
        if self.args.offline_dataset:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
        rows = _load_dataset(self.args.dataset, self.args.split)
        self.rows = select_rows(
            rows,
            instance_ids_file=self.args.instance_ids_file,
            instance_filter=self.args.filter,
            slice_spec=self.args.slice,
        )
        self.instance_ids = [row["instance_id"] for row in self.rows]
        self.image_names = _instance_image_names(self.rows)

    def _prepare_extra_mini_configs(self) -> None:
        records = []
        snapshots = []
        snapshot_dir = self.run_dir / "mini_extra_configs"
        for index, source in enumerate(self.extra_mini_configs):
            try:
                content = source.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise RunnerError(f"Extra mini-SWE-agent config does not exist: {source}") from exc
            if SECRET_PATTERN.search(content) or re.search(r"(?im)^\s*api_key\s*:", content):
                raise RunnerError(
                    f"Extra mini-SWE-agent config appears to contain an API key: {source}"
                )
            snapshot = snapshot_dir / f"{index:02d}_{source.name}"
            if snapshot.exists() and snapshot.read_text(encoding="utf-8") != content:
                raise RunnerError(f"Extra mini-SWE-agent config snapshot changed: {snapshot}")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text(content, encoding="utf-8")
            records.append(
                {
                    "source": str(source),
                    "snapshot": str(snapshot),
                    "sha256": _sha256_bytes(content.encode("utf-8")),
                }
            )
            snapshots.append(snapshot)
        self.extra_mini_config_records = records
        self.extra_mini_config_snapshots = snapshots

    def _semantic_config(self, server_manifest: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.args.dataset,
            "split": self.args.split,
            "instance_ids": self.instance_ids,
            "dataset_rows_sha256": _canonical_hash(self.rows),
            "model": self.args.model,
            "api_base": self.args.api_base,
            "served_model_name": self.args.served_model_name,
            "mini_base_config": self.args.mini_base_config,
            "mini_extra_configs": self.extra_mini_config_records,
            "mini_command": self.args.mini_command,
            "batch_size": self.args.batch_size,
            "step_limit": self.args.step_limit,
            "cost_limit": self.args.cost_limit,
            "wall_time_limit_seconds": self.args.wall_time_limit_seconds,
            "cost_tracking": self.args.cost_tracking,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "seed": None,
            "seed_control": "not exposed by this adapter",
            "eval_timeout": self.args.eval_timeout,
            "server_manifest_sha256": _canonical_hash(server_manifest) if server_manifest else None,
        }

    def prepare(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._status("prepare", "running", "validating dataset, server, and Docker images")
        self._load_selection()
        self._prepare_extra_mini_configs()
        self.instances_path.write_text("\n".join(self.instance_ids) + "\n", encoding="utf-8")
        self.images_path.write_text("\n".join(self.image_names) + "\n", encoding="utf-8")

        server_manifest = None
        parsed_api = urllib.parse.urlparse(self.args.api_base) if self.args.api_base else None
        is_local_api = parsed_api is not None and parsed_api.hostname in LOCAL_HOSTS
        if is_local_api and self.args.server_manifest is None:
            raise RunnerError("A local --api-base requires --server-manifest")
        if self.args.server_manifest is not None:
            if not self.args.api_base:
                raise RunnerError("--server-manifest requires --api-base")
            server_manifest = _validate_local_server_manifest(
                self.args.server_manifest,
                api_base=self.args.api_base,
                served_model_name=self.args.served_model_name,
            )

        if self.args.api_base and self.requires_model_api:
            check_openai_server(
                self.args.api_base,
                api_key=self.api_key,
                served_model_name=self.args.served_model_name,
                timeout=self.args.health_timeout,
                use_environment_proxy=self.args.api_proxy_from_environment,
            )
        requires_images = self.args.stage in {"prepare", "generate", "evaluate", "all"}
        if requires_images and not self.args.allow_image_pulls:
            _require_local_images(self.image_names)

        semantic_config = self._semantic_config(server_manifest)
        if self.run_config_path.exists():
            existing = _read_json(self.run_config_path)
            if existing != semantic_config:
                raise RunnerError(
                    f"Run configuration differs from existing {self.run_config_path}; "
                    "use a new --run-dir instead of mixing experiments"
                )
        else:
            _write_json(self.run_config_path, semantic_config)

        config_text = render_mini_config(
            step_limit=self.args.step_limit,
            cost_limit=self.args.cost_limit,
            wall_time_limit_seconds=self.args.wall_time_limit_seconds,
            cost_tracking=self.args.cost_tracking,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            api_base=self.args.api_base,
        )
        if (
            self.mini_config_path.exists()
            and self.mini_config_path.read_text(encoding="utf-8") != config_text
        ):
            raise RunnerError(f"Generated mini-SWE-agent config changed: {self.mini_config_path}")
        self.mini_config_path.write_text(config_text, encoding="utf-8")

        if server_manifest is not None:
            _write_json(self.run_dir / "server_manifest.json", server_manifest)
        if not self.manifest_path.exists():
            _write_json(
                self.manifest_path,
                {
                    "created_at": _now(),
                    "adapter_git": _git_state(self.repo_root),
                    "swe_bench_git": _git_state(self.swe_bench_dir),
                    "python": {"executable": sys.executable, "version": sys.version},
                    "packages": {
                        name: _package_version(name)
                        for name in ("mini-swe-agent", "swebench", "litellm", "datasets")
                    },
                    "run_config": semantic_config,
                    "run_id": self.run_id,
                    "run_dir": str(self.run_dir),
                    "api_key_env": self.args.api_key_env,
                    "api_proxy_from_environment": self.args.api_proxy_from_environment,
                    "offline_dataset": self.args.offline_dataset,
                },
            )
        _append_jsonl(
            self.invocations_path,
            {
                "time": _now(),
                "argv": sys.argv,
                "mini_workers": self.args.mini_workers,
                "eval_workers": self.args.eval_workers,
                "batch_size": self.args.batch_size,
                "stage": self.args.stage,
            },
        )
        self._status(
            "prepare",
            "completed",
            f"instances={len(self.instance_ids)} "
            f"images_checked={requires_images and not self.args.allow_image_pulls}",
        )

    def _batches(self) -> list[list[str]]:
        return [
            self.instance_ids[index : index + self.args.batch_size]
            for index in range(0, len(self.instance_ids), self.args.batch_size)
        ]

    def generate(self) -> None:
        batches = self._batches()
        model_env = self._model_env()
        mini_prefix = shlex.split(self.args.mini_command)
        if not mini_prefix:
            raise RunnerError("--mini-command is empty")

        for index, instance_ids in enumerate(batches):
            batch_name = f"batch_{index:03d}"
            batch_dir = self.run_dir / "batches" / batch_name
            batch_dir.mkdir(parents=True, exist_ok=True)
            ids_path = batch_dir / "instances.txt"
            ids_path.write_text("\n".join(instance_ids) + "\n", encoding="utf-8")
            done_path = batch_dir / "batch_done.json"
            predictions_path = batch_dir / "preds.json"
            if done_path.exists():
                validate_predictions(
                    _read_json(predictions_path), instance_ids, source=predictions_path
                )
                self._status(batch_name, "skipped", "validated completed batch")
                continue

            instance_regex = "^(?:" + "|".join(re.escape(item) for item in instance_ids) + ")$"
            command = [
                *mini_prefix,
                "swebench",
                "--subset",
                self.args.dataset,
                "--split",
                self.args.split,
                "--filter",
                instance_regex,
                "--workers",
                str(self.args.mini_workers),
                "--output",
                str(batch_dir),
                "--model",
                self.args.model,
                "--environment-class",
                "docker",
                "-c",
                self.args.mini_base_config,
            ]
            for extra_config in self.extra_mini_config_snapshots:
                command.extend(["-c", str(extra_config)])
            command.extend(["-c", str(self.mini_config_path)])
            self._status(
                batch_name,
                "running",
                f"instances={len(instance_ids)} workers={self.args.mini_workers}",
            )
            try:
                _run_logged(
                    command,
                    cwd=self.swe_bench_dir,
                    env=model_env,
                    log_path=self.run_dir / "logs" / f"{batch_name}_mini.log",
                )
                predictions = validate_predictions(
                    _read_json(predictions_path), instance_ids, source=predictions_path
                )
                _write_json(
                    done_path,
                    {
                        "completed_at": _now(),
                        "instances": len(instance_ids),
                        "predictions_sha256": _canonical_hash(predictions),
                    },
                )
            except Exception as exc:
                self._status(batch_name, "failed", str(exc))
                raise
            self._status(batch_name, "completed", f"instances={len(instance_ids)}")

        self._status("merge", "running", "validating and merging batch predictions")
        combined, generation_rows = merge_batch_predictions(self.run_dir, batches)
        validate_predictions(combined, self.instance_ids, source=self.predictions_path)
        _write_json(self.predictions_path, combined)
        _write_jsonl(self.generation_results_path, generation_rows)
        _write_json(
            self.run_dir / "prediction_merge_summary.json",
            {
                "expected": len(self.instance_ids),
                "combined": len(combined),
                "predictions_sha256": _canonical_hash(combined),
                "generation_results_sha256": _sha256_file(self.generation_results_path),
            },
        )
        self._status("merge", "completed", f"predictions={len(combined)}")

    def _official_report_path(self) -> Path:
        candidates = sorted(self.official_dir.glob(f"*.{self.run_id}.json"))
        if len(candidates) != 1:
            raise RunnerError(
                f"Expected one official report matching *.{self.run_id}.json in "
                f"{self.official_dir}, found {len(candidates)}"
            )
        return candidates[0]

    def evaluate(self) -> None:
        predictions = validate_predictions(
            _read_json(self.predictions_path), self.instance_ids, source=self.predictions_path
        )
        merge_summary = _read_json(self.run_dir / "prediction_merge_summary.json")
        if _canonical_hash(predictions) != merge_summary["predictions_sha256"]:
            raise RunnerError("preds_all.json changed after generation merge")
        self.official_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self.args.dataset,
            "--split",
            self.args.split,
            "--predictions_path",
            str(self.predictions_path),
            "--max_workers",
            str(self.args.eval_workers),
            "--timeout",
            str(self.args.eval_timeout),
            "--run_id",
            self.run_id,
            "--report_dir",
            str(self.official_dir),
            "--instance_ids",
            *self.instance_ids,
        ]
        self._status(
            "evaluate",
            "running",
            f"instances={len(self.instance_ids)} workers={self.args.eval_workers}",
        )
        try:
            _run_logged(
                command,
                cwd=self.swe_bench_dir,
                env=self._model_env(),
                log_path=self.run_dir / "logs" / "official_eval.log",
            )
            report_path = self._official_report_path()
            report = _read_json(report_path)
            if not isinstance(report, dict) or report.get("total_instances") != len(
                self.instance_ids
            ):
                raise RunnerError(
                    f"Official report does not cover selected instances: {report_path}"
                )
        except Exception as exc:
            self._status("evaluate", "failed", str(exc))
            raise
        self._status("evaluate", "completed", f"report={report_path}")

    def summarize(self) -> None:
        batches = self._batches()
        predictions = validate_predictions(
            _read_json(self.predictions_path), self.instance_ids, source=self.predictions_path
        )
        _, generation_rows = merge_batch_predictions(self.run_dir, batches)
        report_path = self._official_report_path()
        official_report = _read_json(report_path)
        if not isinstance(official_report, dict):
            raise RunnerError(f"Official report must be a JSON object: {report_path}")
        per_sample, summary = normalize_results(
            expected_ids=self.instance_ids,
            predictions=predictions,
            generation_rows=generation_rows,
            official_report=official_report,
        )
        per_sample_path = self.run_dir / "per_sample_results.jsonl"
        _write_jsonl(per_sample_path, per_sample)
        summary.update(
            {
                "written_at": _now(),
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "model": self.args.model,
                "dataset": self.args.dataset,
                "split": self.args.split,
                "official_report_path": str(report_path),
                "official_eval_log_path": str(self.run_dir / "logs" / "official_eval.log"),
                "official_instance_log_root": str(
                    self.swe_bench_dir / "logs" / "run_evaluation" / self.run_id
                ),
                "predictions_path": str(self.predictions_path),
                "generation_results_path": str(self.generation_results_path),
                "per_sample_results_path": str(per_sample_path),
                "run_config_path": str(self.run_config_path),
                "run_manifest_path": str(self.manifest_path),
            }
        )
        _write_json(self.run_dir / "final_summary.json", summary)
        self._status(
            "summarize",
            "completed",
            f"resolved={summary['resolved_instances']}/{summary['total_instances']}",
        )

    def run(self) -> None:
        try:
            self.prepare()
        except Exception as exc:
            self._status("prepare", "failed", str(exc))
            raise
        if self.args.stage in {"generate", "all"}:
            self.generate()
        if self.args.stage in {"evaluate", "all"}:
            self.evaluate()
        if self.args.stage in {"summarize", "all"}:
            try:
                self.summarize()
            except Exception as exc:
                self._status("summarize", "failed", str(exc))
                raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run mini-SWE-agent and the official SWE-bench Lite harness without "
            "vendoring either project."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "evaluate", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--swe-bench-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-ids-file", type=Path, default=None)
    parser.add_argument("--filter", default=None, help="Regex applied before --slice.")
    parser.add_argument("--slice", default=None, help="START:STOP applied after --filter.")

    parser.add_argument(
        "--model", required=True, help="LiteLLM model id, e.g. openai/sparsevllm-swe."
    )
    parser.add_argument(
        "--api-base", default=None, help="OpenAI-compatible API base ending in /v1."
    )
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--server-manifest", type=Path, default=None)
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument(
        "--api-proxy-from-environment",
        action="store_true",
        help="Keep HTTP proxy variables for model API calls. Direct access is the default.",
    )

    parser.add_argument("--mini-command", default="mini-extra")
    parser.add_argument("--mini-base-config", default="swebench.yaml")
    parser.add_argument(
        "--mini-extra-config",
        action="append",
        type=Path,
        default=[],
        help="Additional provider config merged before the generated shared config.",
    )
    parser.add_argument("--mini-workers", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--wall-time-limit-seconds", type=int, default=1800)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--cost-tracking", choices=("default", "ignore_errors"), default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--allow-image-pulls",
        action="store_true",
        help="Skip the local-image check and allow external tools to pull/build missing images.",
    )
    parser.add_argument(
        "--allow-dataset-download",
        action="store_true",
        help="Allow Hugging Face access instead of forcing cached/offline dataset loading.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.served_model_name is None:
        args.served_model_name = args.model.rsplit("/", 1)[-1]
    if args.cost_tracking is None:
        args.cost_tracking = "ignore_errors" if args.api_base else "default"
    args.offline_dataset = not args.allow_dataset_download
    try:
        SweBenchLiteRunner(args).run()
    except RunnerError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
