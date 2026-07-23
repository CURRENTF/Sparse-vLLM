# Simulated Deep Research Serving Benchmark

This benchmark is a deterministic serving workload, not a research-quality
evaluation. BrowseComp-Plus remains the separate end-to-end quality benchmark.

## Workload

The default run models one research job:

- 10 sequential research rounds.
- 20 parallel subagent requests per round.
- Each subagent receives 64 query tokens plus a heavy-tailed article length:
  60% sample 1,000-8,000 tokens, 25% sample 8,001-16,000, 10% sample
  16,001-32,000, and 5% sample 32,001-64,000.
- Subagent outputs use a second heavy-tailed distribution: 90% sample
  100-600 tokens and 10% sample 800-1,500, with `ignore_eos=true`.
- After every subagent barrier, one main-agent request compresses the 20
  answers into a uniformly sampled 512-1,024 token round summary.
- Main-agent prompts use a stable synthetic system/query prefix, accumulated
  prior round summaries, and the current round's answers. This creates
  deterministic cross-round prefix reuse without retaining every raw answer.
- One final main-agent request consumes all round summaries and generates a
  uniformly sampled 1,000-2,000 tokens.

The defaults therefore issue 211 model requests: 200 SnapKV subagent requests,
10 OmniKV/vanilla round-summary requests, and one OmniKV/vanilla final request.
Synthetic token ids make requested prompt lengths exact and avoid turning this
serving benchmark into a tokenizer-content benchmark. The default expected
article and subagent-output lengths are approximately 10,500 and 430 tokens,
respectively. The maximum request needs a context limit of at least 65,564
tokens.

## Non-Uniform Router Contract

The runner calls the smart router's `/v1/completions` endpoint.

- Subagents send `svllm_method_preference=snapkv`.
- Main-agent requests send
  `svllm_method_preference=omnikv,vanilla`.
- The preflight requires at least two entries in `/health`'s
  `healthy_workers`.
- Every response must include `X-SparseVLLM-Worker`,
  `X-SparseVLLM-Route-Reason`, and `X-SparseVLLM-Sparse-Method`.
- A request routed to the wrong method is recorded as `metric_failed`, and the
  run stops after the active round has been recorded.

This contract requires separate workers because one worker advertises one
sparse method. A typical deployment uses one GPU for the main-agent
OmniKV/vanilla worker and another GPU for the SnapKV subagent worker. The
main-agent worker should enable prefix caching because the workload
deliberately reuses its growing prefix. The benchmark does not start or
reconfigure those services.

## Run

Start the workers and router using the OpenAI serving runbook, then run:

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:18180/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/<RUN_NAME>
```

The router must have workers whose advertised methods satisfy
`--subagent-methods snapkv` and `--main-agent-methods omnikv,vanilla`.
For the default profile, configure every routed worker with
`max_model_len >= 65564`; 69,632 leaves a small explicit margin.

The weighted distributions are configurable:

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:18180/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/<RUN_NAME> \
  --article-token-buckets 60:1000:8000,25:8001:16000,10:16001:32000,5:32001:64000 \
  --subagent-output-token-buckets 90:100:600,10:800:1500
```

For an explicitly labeled single-worker baseline, disable the multi-worker
router checks:

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:8000/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/<BASELINE_NAME> \
  --allow-direct-server
```

`--allow-direct-server` omits router-only method hints and disables
router-header and actual-method validation. It must not be used for a result
claimed as a non-uniform router run.

## Artifacts

Each output directory is created as a new directory; an existing directory is
rejected. The run writes:

- `run_info.json`: command, resolved workload, Git state, environment, router
  health, model context limit, and benchmark-critical resolved configuration
  reported by every worker.
- `raw_outputs.jsonl`: raw HTTP responses and response headers.
- `parsed_outputs.jsonl`: extracted text, finish reason, usage, and parse
  status.
- `per_sample_results.jsonl`: phase, requested/actual token counts, latency,
  method preference, actual worker, actual method, expected reusable
  main-agent prefix tokens, and explicit status.
- `round_metrics.jsonl`: barrier time, p50/p95/max subagent latency, straggler
  gap, and main-agent latency for every completed round.
- `aggregate_metrics.json`: end-to-end research-job throughput, token
  throughput, latency percentiles, status counts, and route distributions.

Any failed HTTP request, malformed response, token-count mismatch, wrong method
route, or missing route header remains visible in the artifacts and makes the
run fail.

`expected_reusable_prefix_tokens` is the raw common-prefix length. A block-based
prefix cache can reuse the largest whole-block prefix, so observed hit tokens
may be lower by fewer than one cache block per hit.
