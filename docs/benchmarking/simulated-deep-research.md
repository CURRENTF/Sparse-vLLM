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
- The preflight requires at least two healthy workers for the selected model
  and verifies that the advertised methods and context capacities cover both
  agent roles. Same-model workers with unrelated methods do not constrain the
  benchmark.
- Every worker eligible for main-agent routing must have prefix caching
  enabled and report a positive integer `prefix_cache_block_size`.
- Every response must include `X-SparseVLLM-Worker`,
  `X-SparseVLLM-Route-Reason`, `X-SparseVLLM-Sparse-Method`, and
  `X-SparseVLLM-Prefix-Matched-Tokens`.
- A request routed to the wrong method is recorded as `metric_failed`, and the
  run stops after the active round has been recorded.
- Preflight rejects a router or benchmark-eligible worker that reports
  `git_dirty=true`. An installed package may report `git_dirty=null` only when
  it provides a package version.

This contract requires separate workers because one worker advertises one
sparse method. A typical deployment uses one GPU for the main-agent
OmniKV/vanilla worker and another GPU for the SnapKV subagent worker. The
main-agent worker must enable prefix caching because the workload deliberately
reuses its growing prefix; preflight rejects any eligible main-agent worker
without it. The benchmark does not start or reconfigure those services.

## Run

Start the workers and router using the OpenAI serving runbook, then run:

```bash
python -m benchmark.simulated_deep_research.run \
  --base-url http://127.0.0.1:18180/v1 \
  --model <SERVED_MODEL_NAME> \
  --output-dir outputs/simulated_deep_research/<RUN_NAME>
```

The client timeout must exceed the router's upstream timeout by at least
`--router-timeout-margin-s`. The defaults use a 930-second client timeout, a
30-second margin, and `SPARSEVLLM_ROUTER_REQUEST_TIMEOUT_S=900` for the
systemd router. Keep the separate router control-plane timeout short; its
default is 5 seconds.
The worker serves internal routing-load and prefix-match probes from immutable
dispatcher snapshots, so a long synchronous prefill step does not turn that
short control timeout into a false worker failure. The full `/v1/worker/load`
control endpoint remains synchronized with the engine and is not used by the
router's short-timeout probe.
The router must have workers whose advertised methods satisfy
`--subagent-methods snapkv` and `--main-agent-methods omnikv,vanilla`.
For the default profile, every eligible subagent worker needs
`max_model_len >= 65564`, while every eligible main-agent worker needs
`max_model_len >= 40368`. A larger shared limit such as 69,632 is also valid.
The preflight reads these limits from each eligible worker instead of applying
the router model card's cross-method minimum.

The synthetic workload remains exactly reproducible from `seed`; the runner
does not inject a per-run nonce. Consequently, rerunning the same seed against
an uncleared worker could encounter cache entries from an earlier run. The
runner detects this instead of silently measuring the warm cache: an actual
match larger than this run's block-aligned expectation, including any hit on
the first main-agent request, is `metric_failed`. Clear or restart the
prefix-cache worker before rerunning a fixed seed.

The runner is executed from the source tree and requires successful Git
inspection before preflight. If the client commit or worktree status cannot be
inspected, the run fails with `invalid_input` and writes failure artifacts
instead of producing a result with unknown client provenance.

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

- `run_info.json`: command, resolved workload, client Git state, environment,
  router health and code revision, role-specific context requirements, and
  benchmark-critical configuration and code revision reported by every
  worker.
- `client_worktree.patch`: emitted when tracked client files are dirty. Its
  path and SHA-256 are recorded in `run_info.json`; untracked client files are
  rejected because a tracked Git patch cannot reproduce them.
- `raw_outputs.jsonl`: exact request URL, timeout, prompt seed and payload,
  plus raw HTTP responses, response headers, and the parsed router observation.
- `parsed_outputs.jsonl`: extracted text, finish reason, usage, and parse
  status.
- `per_sample_results.jsonl`: phase, requested/actual token counts, latency,
  method preference, actual worker, actual method, expected reusable
  main-agent prefix tokens, selected-worker block size, block-aligned expected
  tokens, the number of successful prior prompts on that worker, the selected
  worker's actual matched prefix tokens, and explicit status.
- `round_metrics.jsonl`: barrier time, p50/p95/max subagent latency, straggler
  gap, main-agent latency, and explicit status for every attempted round,
  including a round whose main-agent request fails.
- `aggregate_metrics.json`: end-to-end research-job throughput, token
  throughput, latency percentiles, status counts, route distributions, and
  separate attempted/completed round counts. Its top-level and per-phase
  `prefix_cache` objects include raw expected, block-aligned expected, and
  actual token totals, expected and actual hit counts, partial hits, unexpected
  zero hits, and unexpected excess hits.

Any failed HTTP request, malformed response, token-count mismatch, wrong method
route, missing route header, invalid matched-token header, expected cacheable
reuse with zero actual matched tokens, or actual reuse beyond this run's
expectation remains visible in the artifacts and makes the run fail.

`expected_reusable_prefix_tokens` is the largest raw common-prefix length
between the current main-agent prompt and every successful prior main-agent
prompt handled by the selected worker in this run. Prompts handled by another
worker never raise this expectation. For block size `b`, each prior prompt
contributes the minimum of its whole-block common prefix, the current prompt's
cacheable portion `floor((current_length - 1) / b) * b`, and the prior prompt's
materialized portion `floor(prior_length / b) * b`.
`block_aligned_expected_reusable_prefix_tokens` is the largest contribution.
Only the current prompt retains a final token for logits; if a prior prompt
ends exactly on a block boundary, that final complete block was materialized
and remains reusable. A raw prefix shorter than one block therefore allows an
actual zero hit. A nonzero partial hit is recorded without failing because
cache capacity can reduce reuse. Zero actual reuse when the block-aligned
expectation is nonzero, or actual reuse above that expectation, is
`metric_failed`.
