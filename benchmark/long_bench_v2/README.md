# LongBench v2

This directory keeps LongBench v2 separate from the existing LongBench v1
runner under `benchmark/long_bench/`.

The official THUDM repository is pinned as the `upstream/` Git submodule. The
native runner uses its official zero-shot prompt and records the submodule
commit in every result. Initialize it after cloning:

```bash
git submodule update --init benchmark/long_bench_v2/upstream
```

The official dataset is distributed separately through Hugging Face and is not
duplicated in the source submodule. Export the `zai-org/LongBench-v2` train
split (the former `THUDM/LongBench-v2` alias redirects there) to one local JSON
or JSONL file, then set:

```bash
export SPARSEVLLM_LONGBENCH_V2_DATA=<LONGBENCH_V2_JSON_OR_JSONL>
```

`pred.py` runs the native Sparse-vLLM engine, selects a deterministic subset in
configured post-chat-template token buckets, and never truncates a source
prompt. A bucket with insufficient samples that fit the requested model budget
fails explicitly. It saves the selected identities and hashes, raw responses,
parsed answers, per-sample statuses, aggregate metrics, runtime configuration,
and source/submodule provenance.

For paired external-runtime evaluation, `--engine vllm` accepts constructor
options through `--engine-kwargs`, and `--engine sglang-http --server-url ...`
uses an already-running server with matching model identity and disabled radix
cache. Both receive the exact prepared token IDs and use the same answer parser
and aggregate metrics as the native runner. Preserve the external server launch
configuration separately; a matching method label alone does not establish
algorithm or budget parity.

`--official-length medium` filters the dataset's official length label before
token-bucket selection. `--prepare-only` exports prepared samples for reuse with
`--prepared-samples`; model/tokenizer, dataset, prompt, seed and bucket identities
must match. Preparation is not a model evaluation. If a common context-limited
subset is used, retain excluded sample IDs and reasons and label its accuracy as
a subset score, not the complete official length-cohort score. Full runs retain
incremental per-sample output in `sample_results.partial.jsonl`.

As in the official evaluator, a non-empty model response that does not contain
the required answer pattern is retained with `status="parse_failed"` and scored
as incorrect. Model/runtime failures remain fatal and invalidate the run.

The canonical 120-sample, greedy-decoding profile is a repository regression
gate, not a reproduction of the full 503-sample LongBench v2 leaderboard.

Use `benchmark/sparsevllm_regression/run_suite.py --layer longbench_v2` for the
canonical gate. The default quality layer runs LongBench v1, LongBench v2, and
RULER; use `--quality_benchmarks` only for an intentional focused run.
