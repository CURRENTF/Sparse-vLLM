# SWE-bench Lite

## Purpose

`benchmark/swe_bench_lite/run.py` is a thin adapter around two external,
upstream-owned components:

- mini-SWE-agent generates a patch for each issue through an OpenAI-compatible
  model API.
- `swebench.harness.run_evaluation` applies each patch and computes the official
  SWE-bench result in Docker.

The adapter does not reimplement SWE-bench datasets, environments, tests, or
metrics. It supports `SWE-bench/SWE-bench_Lite`, split `test`, including the
full 300-instance benchmark and smaller smoke selections.

Upstream sources:

- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)
- [SWE-bench Lite dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite)

## Prerequisites

Use a separate Python environment that contains both `mini-swe-agent` with its
SWE-bench extra and the official SWE-bench harness. Keep the upstream checkout
outside this repository, for example at `../SWE-bench`.

Set `SWE_BENCH_PYTHON` to that environment's Python. The shell entrypoint also
puts the same environment's `bin/` directory first on `PATH`, so `mini-extra`
and the imported SWE-bench harness cannot accidentally come from different
environments.

The adapter defaults to cached/offline Hugging Face access and requires every
selected Docker image to exist locally. It does not silently download the
dataset or hundreds of gigabytes of images. Use `--allow-dataset-download` or
`--allow-image-pulls` only when that network and storage use is intentional.

The selected instance and image names are written to `instances.txt` and
`images.txt` in the run directory, including when the local-image check fails.

## Sparse-vLLM Server

Start `sparsevllm.entrypoints.openai.api_server` as a separate long-running
process. A typical command is:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=$PWD/src \
python -m sparsevllm.entrypoints.openai.api_server \
  --model <MODEL_PATH> \
  --served-model-name sparsevllm-swe \
  --host 127.0.0.1 \
  --port 18000 \
  --engine-kwargs /path/to/engine_kwargs.json \
  --request-log-dir /path/to/server_requests
```

For example, `engine_kwargs.json` can contain:

```json
{
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.88,
  "max_model_len": 131072,
  "engine_prefill_chunk_size": 4096,
  "sparse_method": "vanilla"
}
```

Create a JSON manifest next to the server logs. Do not put API keys or other
secrets in it.

```json
{
  "command": "python -m sparsevllm.entrypoints.openai.api_server --model <MODEL_PATH> --served-model-name sparsevllm-swe --host 127.0.0.1 --port 18000 --engine-kwargs /path/to/engine_kwargs.json",
  "model_path": "<MODEL_PATH>",
  "served_model_name": "sparsevllm-swe",
  "cuda_visible_devices": "2",
  "server_port": 18000,
  "engine_kwargs": {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.88,
    "max_model_len": 131072,
    "engine_prefill_chunk_size": 4096,
    "sparse_method": "vanilla"
  }
}
```

For a local API, the adapter requires this manifest and snapshots it into the
benchmark run. It also checks `/v1/models` and verifies that the advertised
model matches `--served-model-name`. It does not start, restart, or stop the
server.

## One-Instance Smoke

Run one instance before committing to all 300. `openai/` is the LiteLLM
provider prefix; `sparsevllm-swe` is the exact model name advertised by the
server. Sparse-vLLM does not require authentication, but LiteLLM expects a
non-empty OpenAI key, so use a dummy local value.

```bash
export OPENAI_API_KEY=local-sparsevllm

SWE_BENCH_PYTHON=/path/to/swebench-env/bin/python \
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage all \
  --swe-bench-dir ../SWE-bench \
  --run-dir /path/to/outputs/swe-bench-lite/sparsevllm-smoke \
  --model openai/sparsevllm-swe \
  --api-base http://127.0.0.1:18000/v1 \
  --served-model-name sparsevllm-swe \
  --server-manifest /path/to/server_manifest.json \
  --slice 0:1 \
  --batch-size 1 \
  --mini-workers 1 \
  --eval-workers 1 \
  --step-limit 80 \
  --cost-tracking ignore_errors \
  --cost-limit 0
```

Local models have no provider billing metadata. With
`--cost-tracking ignore_errors`, cost is recorded as zero and `--cost-limit`
cannot enforce a budget; the bounded controls are `--step-limit` and
`--wall-time-limit-seconds`.

## Full Lite Run

Use a new run directory and remove `--slice 0:1`. Keep model, API, prompt,
decoding, dataset, and server settings identical to the smoke run. Start with
conservative generation concurrency and raise `--mini-workers` only after the
server is stable under simultaneous tool-calling requests.

```bash
export OPENAI_API_KEY=local-sparsevllm

SWE_BENCH_PYTHON=/path/to/swebench-env/bin/python \
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage all \
  --swe-bench-dir ../SWE-bench \
  --run-dir /path/to/outputs/swe-bench-lite/sparsevllm-lite300 \
  --model openai/sparsevllm-swe \
  --api-base http://127.0.0.1:18000/v1 \
  --served-model-name sparsevllm-swe \
  --server-manifest /path/to/server_manifest.json \
  --batch-size 50 \
  --mini-workers 1 \
  --eval-workers 6 \
  --step-limit 80 \
  --cost-tracking ignore_errors \
  --cost-limit 0
```

Completed batches are validated and skipped on rerun. Partial mini-SWE-agent
batch directories are passed back to mini-SWE-agent, whose default behavior is
to skip completed trajectories. The adapter only merges declared numeric batch
directories, so backup directories cannot introduce duplicate predictions.

## Separate Generation And Evaluation

The LLM server is needed only for `prepare` and `generate`. Official evaluation
does not call the model API. To run the stages separately, repeat the same
semantic arguments and change only `--stage` and operational worker counts:

```bash
COMMON_ARGS=(
  --swe-bench-dir ../SWE-bench
  --run-dir /path/to/outputs/swe-bench-lite/sparsevllm-lite300
  --model openai/sparsevllm-swe
  --api-base http://127.0.0.1:18000/v1
  --served-model-name sparsevllm-swe
  --server-manifest /path/to/server_manifest.json
  --batch-size 50
  --step-limit 80
  --cost-tracking ignore_errors
  --cost-limit 0
)

# Requires the model API and API key.
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage generate --mini-workers 1 "${COMMON_ARGS[@]}"

# Requires Docker images, but not the model API or API key.
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage evaluate --eval-workers 6 "${COMMON_ARGS[@]}"

# Requires only completed generation and official report artifacts.
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage summarize "${COMMON_ARGS[@]}"
```

The adapter rejects semantic configuration changes in an existing run
directory. Use a new run directory when changing the model, dataset selection,
step limit, decoding settings, API endpoint, or server configuration.

The shared adapter fixes `temperature=0`, `top_p=1`, and `max_tokens=4096` by
default. It records `seed=null` because this OpenAI-compatible route does not
currently expose a shared seed control.

## External API Providers

For DeepSeek or another LiteLLM provider, omit `--api-base`,
`--served-model-name`, and `--server-manifest`, then select the provider key
environment variable explicitly. Model calls are direct by default: proxy
environment variables are removed from the mini-SWE-agent process. Use
`--api-proxy-from-environment` only when the provider requires that proxy.

```bash
export DEEPSEEK_API_KEY=<KEY>

SWE_BENCH_PYTHON=/path/to/swebench-env/bin/python \
bash scripts/benchmarks/run_swe_bench_lite.sh \
  --stage all \
  --swe-bench-dir ../SWE-bench \
  --run-dir /path/to/outputs/swe-bench-lite/deepseek-lite300 \
  --model deepseek/deepseek-v4-flash \
  --api-key-env DEEPSEEK_API_KEY \
  --mini-extra-config /path/to/deepseek_nonthinking.yaml \
  --cost-tracking default \
  --cost-limit 0.05 \
  --step-limit 80
```

The optional provider config used above contains only the DeepSeek-specific
request field:

```yaml
model:
  model_kwargs:
    extra_body:
      thinking:
        type: disabled
```

Provider-specific request fields, such as DeepSeek thinking controls, are not
part of the shared adapter configuration. Pass them with a provider-specific
`--mini-extra-config`; the adapter hashes and snapshots that file. Do not reuse
such a config for Sparse-vLLM, and never store an API key in it.

## Outputs

Each run directory contains:

| Artifact | Meaning |
| --- | --- |
| `run_config.json` | Immutable semantic experiment configuration and selected instance ids. |
| `run_manifest.json` | Code revisions, package versions, Python, credential variable name, and runtime policy. |
| `server_manifest.json` | Snapshot of the local Sparse-vLLM server configuration, when applicable. |
| `invocations.jsonl` | Stage invocations and operational concurrency settings. |
| `status.jsonl` | Append-only stage and batch status events. |
| `batches/*/*traj.json` | Raw mini-SWE-agent trajectories and model responses. |
| `preds_all.json` | Strictly validated patches in official SWE-bench prediction format. |
| `generation_results.jsonl` | Parsed generation status and model statistics per instance. |
| `official/*.json` | Unmodified official SWE-bench aggregate report. |
| `per_sample_results.jsonl` | Normalized per-instance status and official outcome. |
| `final_summary.json` | Aggregate score, status counts, API calls, cost, and artifact paths. |

The normalized `status` values are `success`, `invalid_input`, `model_failed`,
`parse_failed`, `metric_failed`, and `skipped_by_policy`. `success` means the
official harness completed the instance; the separate `resolved` field records
whether its tests passed. The benchmark score is always
`resolved_instances / total_instances`, including empty patches and failures in
the denominator.
