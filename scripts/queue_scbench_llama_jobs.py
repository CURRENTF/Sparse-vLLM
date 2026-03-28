#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/haojitai/miniconda3/envs/svllm/bin/python"

QWEN_MODEL_PATH = "/home/haojitai/models/Qwen2.5-7B-Instruct-1M"
QWEN_TOKENIZER_PATH = QWEN_MODEL_PATH
QWEN_COMPRESSOR_PATH = (
    "/home/haojitai/checkpoints/compressor/"
    "cluster_e2e_cs256_biasFalse_l2_ratio0.1_clusMean_before_rope_lr0.0002_"
    "cdownmlp_swiglud3072_cuplinear_0125_222950"
)
LONG_BENCH_DATA_ROOT = "/home/haojitai/datasets/LongBench"

LLAMA_MODEL_PATH = "/home/haojitai/models/Llama-3.1-8B-Instruct"
LLAMA_COMPRESSOR_PATH = (
    "/home/haojitai/checkpoints/compressor/"
    "cluster_e2e_cs512_biasFalse_l2_ratio0.1_clusMean_before_rope_lr0.0002_"
    "cdownmlp_swiglud3072_cuplinear_0125_051527"
)

SCBENCH_PREPROCESSED_ROOT = "/home/haojitai/datasets/SCBench-preprocessed"
OUTPUT_ROOT = Path("/home/haojitai/outputs")

FIXED_GPU_IDS = [4, 5, 6, 7]
FIXED_VISIBLE_GPUS = ",".join(str(gpu) for gpu in FIXED_GPU_IDS)
FIXED_WS = len(FIXED_GPU_IDS)

QWEN_SCBENCH_4TASKS = "scbench_kv,scbench_qa_eng,scbench_summary,scbench_summary_with_needles"
QWEN_SCBENCH_MERGED = "scbench_kv,scbench_qa_eng,scbench_summary_with_needles,scbench_many_shot"
QWEN_SCBENCH_MANYSHOT = "scbench_many_shot"
LLAMA_DELTAKV_TASKS = "scbench_kv,scbench_qa_eng,scbench_mf,scbench_summary_with_needles"
KVZIP_TASKS = "scbench_kv,scbench_qa_eng,scbench_summary_with_needles,scbench_many_shot"

QWEN_FULL_ATTN_LAYERS = "0,1,2,4,7,14"
LLAMA_FULL_ATTN_LAYERS = "0,1,2,8,18"
ALPHAS = [0.001, 0.02, 0.05, 0.1, 0.2]


@dataclass
class Job:
    name: str
    kind: str
    stride_alpha: float | None = None
    token_budget: float | None = None
    ratio: float | None = None


def timestamp() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


def fmt_float(v: float) -> str:
    return str(v).replace(".", "p")


def qwen_longbench_model_name(alpha: float) -> str:
    return f"qwen25-7b-lb-deltakv-alpha{fmt_float(alpha)}-omnitrue"


def qwen_scbench_job_name(task_group: str, alpha: float) -> str:
    return f"qwen25-7b-scbench-{task_group}-alpha{fmt_float(alpha)}-omnitrue"


def qwen_deltakv_hyper_param(alpha: float) -> dict[str, Any]:
    return {
        "chunk_prefill_size": 32768,
        "num_top_tokens_in_prefill": 4096,
        "chunk_prefill_accel_omnikv": False,
        "deltakv_use_omnikv_selection": True,
        "num_top_tokens": 0.11,
        "full_attn_layers": QWEN_FULL_ATTN_LAYERS,
        "num_recent_tokens": 128,
        "num_sink_tokens": 8,
        "use_compression": True,
        "use_cluster": True,
        "cluster_ratio": 0.1,
        "stride_alpha": alpha,
    }


def llama_deltakv_hyper_param(token_budget: float) -> dict[str, Any]:
    return {
        "model_cls": "deltakv",
        "compressor_path": LLAMA_COMPRESSOR_PATH,
        "chunk_prefill_size": 32768,
        "num_top_tokens_in_prefill": token_budget,
        "chunk_prefill_accel_omnikv": False,
        "deltakv_use_omnikv_selection": True,
        "num_top_tokens": token_budget,
        "full_attn_layers": LLAMA_FULL_ATTN_LAYERS,
        "num_recent_tokens": 128,
        "num_sink_tokens": 8,
        "use_compression": True,
        "use_cluster": True,
        "cluster_ratio": 0.1,
        "stride_alpha": 0.0,
    }


def qwen_scbench_output_dir(job: Job) -> str:
    return str(OUTPUT_ROOT / "benchmark" / "scbench_alpha_queue" / job.name)


def llama_scbench_output_dir(job: Job) -> str:
    return str(OUTPUT_ROOT / "benchmark" / "scbench_queue" / job.name)


def kvzip_output_dir(job: Job) -> str:
    return str(OUTPUT_ROOT / "benchmark" / "scbench_preprocessed_queue" / job.name)


def build_queue(mode: str = "full") -> list[Job]:
    if mode == "remaining":
        return [
            Job(name="llama_kvzip_scbench_ratio0p30_rerun", kind="llama_kvzip_scbench", ratio=0.30),
            Job(name="llama_deltakv_scbench_b0p17_full012818_rerun", kind="llama_deltakv_scbench", token_budget=0.17),
            Job(name="qwen25-7b-scbench-merged-alpha0-omnitrue-rerun", kind="qwen_scbench_merged_alpha", stride_alpha=0.0),
        ]

    queue: list[Job] = []

    for alpha in ALPHAS:
        queue.append(
            Job(
                name=qwen_longbench_model_name(alpha),
                kind="qwen_longbench_alpha",
                stride_alpha=alpha,
            )
        )

    for alpha in ALPHAS:
        queue.append(
            Job(
                name=qwen_scbench_job_name("4tasks", alpha),
                kind="qwen_scbench_4tasks_alpha",
                stride_alpha=alpha,
            )
        )

    for alpha in ALPHAS:
        queue.append(
            Job(
                name=qwen_scbench_job_name("manyshot", alpha),
                kind="qwen_scbench_manyshot_alpha",
                stride_alpha=alpha,
            )
        )

    queue.append(
        Job(
            name="qwen25-7b-scbench-merged-alpha0-omnitrue",
            kind="qwen_scbench_merged_alpha",
            stride_alpha=0.0,
        )
    )

    queue.extend(
        [
            Job(name="llama_deltakv_scbench_b0p11_full012818", kind="llama_deltakv_scbench", token_budget=0.11),
            Job(name="llama_kvzip_scbench_ratio0p20", kind="llama_kvzip_scbench", ratio=0.20),
            Job(name="llama_deltakv_scbench_b0p05_full012818", kind="llama_deltakv_scbench", token_budget=0.05),
            Job(name="llama_kvzip_scbench_ratio0p30", kind="llama_kvzip_scbench", ratio=0.30),
            Job(name="llama_deltakv_scbench_b0p17_full012818", kind="llama_deltakv_scbench", token_budget=0.17),
        ]
    )
    return queue


def build_command(job: Job, ws: int | str) -> list[str]:
    ws_str = str(ws)

    if job.kind == "qwen_longbench_alpha":
        assert job.stride_alpha is not None
        return [
            PYTHON,
            "-u",
            "benchmark/long_bench/pred.py",
            "--model",
            job.name,
            "--model_path",
            QWEN_MODEL_PATH,
            "--tokenizer_path",
            QWEN_TOKENIZER_PATH,
            "--ws",
            ws_str,
            "--batch_size",
            "1",
            "--backend",
            "hf",
            "--model_cls",
            "deltakv",
            "--compressor_path",
            QWEN_COMPRESSOR_PATH,
            "--temperature",
            "0",
            "--top_p",
            "1.0",
            "--top_k",
            "20",
            "--hyper_param",
            json.dumps(qwen_deltakv_hyper_param(job.stride_alpha), ensure_ascii=False, separators=(",", ":")),
        ]

    if job.kind == "qwen_scbench_4tasks_alpha":
        assert job.stride_alpha is not None
        return [
            PYTHON,
            "-u",
            "benchmark/scbench/run_scbench.py",
            "--task",
            QWEN_SCBENCH_4TASKS,
            "--model_name_or_path",
            QWEN_MODEL_PATH,
            "--output_dir",
            qwen_scbench_output_dir(job),
            "--attn_type",
            "deltakv",
            "--kv_type",
            "dense",
            "--use_chat_template",
            "--trust_remote_code",
            "--max_seq_length",
            "196608",
            "--ws",
            ws_str,
            "--hyper_param",
            json.dumps(
                {
                    "model_cls": "deltakv",
                    "compressor_path": QWEN_COMPRESSOR_PATH,
                    **qwen_deltakv_hyper_param(job.stride_alpha),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]

    if job.kind == "qwen_scbench_manyshot_alpha":
        assert job.stride_alpha is not None
        return [
            PYTHON,
            "-u",
            "benchmark/scbench/run_scbench.py",
            "--task",
            QWEN_SCBENCH_MANYSHOT,
            "--model_name_or_path",
            QWEN_MODEL_PATH,
            "--output_dir",
            qwen_scbench_output_dir(job),
            "--attn_type",
            "deltakv",
            "--kv_type",
            "dense",
            "--use_chat_template",
            "--trust_remote_code",
            "--max_seq_length",
            "196608",
            "--ws",
            ws_str,
            "--hyper_param",
            json.dumps(
                {
                    "model_cls": "deltakv",
                    "compressor_path": QWEN_COMPRESSOR_PATH,
                    **qwen_deltakv_hyper_param(job.stride_alpha),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]

    if job.kind == "qwen_scbench_merged_alpha":
        assert job.stride_alpha is not None
        return [
            PYTHON,
            "-u",
            "benchmark/scbench/run_scbench.py",
            "--task",
            QWEN_SCBENCH_MERGED,
            "--model_name_or_path",
            QWEN_MODEL_PATH,
            "--output_dir",
            qwen_scbench_output_dir(job),
            "--attn_type",
            "deltakv",
            "--kv_type",
            "dense",
            "--use_chat_template",
            "--trust_remote_code",
            "--max_seq_length",
            "196608",
            "--ws",
            ws_str,
            "--hyper_param",
            json.dumps(
                {
                    "model_cls": "deltakv",
                    "compressor_path": QWEN_COMPRESSOR_PATH,
                    **qwen_deltakv_hyper_param(job.stride_alpha),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]

    if job.kind == "llama_deltakv_scbench":
        assert job.token_budget is not None
        return [
            PYTHON,
            "-u",
            "benchmark/scbench/run_scbench.py",
            "--task",
            LLAMA_DELTAKV_TASKS,
            "--model_name_or_path",
            LLAMA_MODEL_PATH,
            "--output_dir",
            llama_scbench_output_dir(job),
            "--attn_type",
            "deltakv",
            "--kv_type",
            "dense",
            "--use_chat_template",
            "--trust_remote_code",
            "--max_seq_length",
            "131072",
            "--ws",
            ws_str,
            "--hyper_param",
            json.dumps(llama_deltakv_hyper_param(job.token_budget), ensure_ascii=False, separators=(",", ":")),
        ]

    if job.kind == "llama_kvzip_scbench":
        assert job.ratio is not None
        return [
            PYTHON,
            "-u",
            "benchmark/scbench/run_kvzip_preprocessed.py",
            "--task",
            KVZIP_TASKS,
            "--data_root",
            SCBENCH_PREPROCESSED_ROOT,
            "--output_dir",
            kvzip_output_dir(job),
            "--model_name_or_path",
            LLAMA_MODEL_PATH,
            "--max_seq_length",
            "131072",
            "--ratio",
            str(job.ratio),
            "--level",
            "pair",
            "--kv_type",
            "evict",
            "--prefill_chunk_size",
            "16000",
            "--ws",
            ws_str,
        ]

    raise ValueError(f"Unknown job kind: {job.kind}")


def log(message: str, log_f):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    print(line, file=log_f, flush=True)


def job_extra_env(job: Job) -> dict[str, str]:
    if job.kind == "qwen_longbench_alpha":
        return {
            "DELTAKV_LONGBENCH_DATA_DIR": LONG_BENCH_DATA_ROOT,
            "DELTAKV_OUTPUT_DIR": str(OUTPUT_ROOT),
        }
    return {}


def command_env_prefix(visible_gpus: str, extra_env: dict[str, str]) -> str:
    repo_src = str(REPO_ROOT / "src")
    env_parts = [
        f"CUDA_VISIBLE_DEVICES={visible_gpus}",
        f"PYTHONPATH={shlex.quote(repo_src)}:${{PYTHONPATH}}",
    ]
    for key, value in extra_env.items():
        env_parts.append(f"{key}={shlex.quote(value)}")
    return " ".join(env_parts)


def render_command(job: Job, ws: int | str, visible_gpus: str) -> str:
    cmd = build_command(job, ws)
    return f"{command_env_prefix(visible_gpus, job_extra_env(job))} " + " ".join(
        shlex.quote(part) for part in cmd
    )


def print_commands(queue: list[Job]) -> int:
    print(f"# queue_size={len(queue)}")
    print(f"# fixed_gpus={FIXED_VISIBLE_GPUS}")
    print(f"# fixed_ws={FIXED_WS}")
    for idx, job in enumerate(queue, start=1):
        print()
        print(f"# {idx:02d} {job.name}")
        print(render_command(job, FIXED_WS, FIXED_VISIBLE_GPUS))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--queue-mode", choices=["full", "remaining"], default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = build_queue(args.queue_mode)
    if args.print_commands:
        return print_commands(queue)

    run_dir = OUTPUT_ROOT / "queue_runs" / f"bench_job_queue_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "queue.log"
    state_path = run_dir / "queue_plan.json"

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "queue": [asdict(job) for job in queue],
                "assumptions": {
                    "mode": "serial_fixed_gpus",
                    "fixed_visible_gpus": FIXED_VISIBLE_GPUS,
                    "fixed_ws": FIXED_WS,
                    "queue_mode": args.queue_mode,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(log_path, "a", encoding="utf-8") as log_f:
        log(f"queue_dir={run_dir}", log_f)
        log(f"queue_plan={state_path}", log_f)
        log(f"running serially on fixed_gpus={FIXED_VISIBLE_GPUS}", log_f)

        for idx, job in enumerate(queue, start=1):
            log(f"starting job {idx}/{len(queue)}: {job.name}", log_f)

            chosen_gpus = FIXED_GPU_IDS
            ws = FIXED_WS
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = FIXED_VISIBLE_GPUS
            env["PYTHONPATH"] = (
                f"{REPO_ROOT / 'src'}:{env['PYTHONPATH']}"
                if env.get("PYTHONPATH")
                else str(REPO_ROOT / "src")
            )
            env.update(job_extra_env(job))

            cmd = build_command(job, ws)
            job_log = run_dir / f"{idx:02d}_{job.name}.log"
            launch_meta = {
                "job": asdict(job),
                "gpus": chosen_gpus,
                "command": cmd,
            }
            with open(run_dir / f"{idx:02d}_{job.name}.json", "w", encoding="utf-8") as f:
                json.dump(launch_meta, f, indent=2, ensure_ascii=False)

            log(f"launching on gpus={chosen_gpus}: {render_command(job, ws, env['CUDA_VISIBLE_DEVICES'])}", log_f)
            with open(job_log, "w", encoding="utf-8") as jf:
                jf.write(json.dumps(launch_meta, indent=2, ensure_ascii=False) + "\n\n")
                jf.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=jf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                ret = proc.wait()

            log(f"job {job.name} finished with code={ret} log={job_log}", log_f)

        log("all queued jobs finished", log_f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
