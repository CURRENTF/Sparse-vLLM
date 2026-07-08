# Benchmark Entrypoints

This directory contains runnable benchmark code. The stable runbook lives in
[`docs/benchmarking/README.md`](../docs/benchmarking/README.md); keep this file
as a lightweight source-tree map.

| Directory | Main entrypoints | Notes |
| --- | --- | --- |
| `long_bench/` | `pred.py`, `eval.py` | LongBench prediction and scoring with HF or Sparse-vLLM backends. |
| `math_bench/` | `pred.py`, `eval.py` | GSM8K, AIME 2024, MATH-500, and HMMT Nov tasks. |
| `scbench/` | `run_scbench.py`, `run_scbench_preprocessed.py`, `compute_scores.py`, `run_kvzip_preprocessed.py` | SCBench standard, preprocessed, scoring, and KVZip routes. |
| `claw_eval/` | `run_sparsevllm_claw_eval.sh` | Claw-Eval through the shared Sparse-vLLM OpenAI-compatible server. |
| `microbench.py` | `microbench.py` | Synthetic prompt-length throughput benchmark for TTFT, prefill/decode tok/s, ITL, and peak memory. |
| `multimodal/` | `video_qa/`, `image_qa/`, `visual_cache/` | Video QA, image QA, and visual-cache benchmark runners. |
| `ruler_vt/` | `pred.py` | Self-contained RULER variable-tracking runner. |
| `niah/` | `test_niah.py`, `gen_niah.py` | Needle-in-a-haystack generation and evaluation utility. |
| `sparsevllm_regression/` | `run_suite.py` | Fixed Sparse-vLLM quality/logits/perf/stress regression harness. |

Do not store local experiment ledgers in this directory. Put reproducible
commands, stable runbook notes, and result-interpretation rules in
`docs/benchmarking/`.
