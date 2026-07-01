# SparseVLLM Regression Rubrics

This file records the stable ABCD rubrics for the SparseVLLM regression
harness. It should not contain dated campaign results, open bug lists, run IDs,
or remote log paths.

The executable grading rules live in
`benchmark/sparsevllm_regression/grading.py`. This document is the human-facing
summary of those rules.

## Common Semantics

- `A`, `B`, and `C` are passing grades with decreasing confidence.
- `D` is a required-gate failure.
- `N/A` means the gate was skipped by policy, such as a method without an HF
  logits reference.
- Dated run results and blockers belong in run artifacts, issue notes, or a
  campaign-specific report outside this rubric file.

## Quality

Quality compares LongBench-mini sparse score against the vanilla score from the
same run.

| Grade | Rule |
| --- | --- |
| `A` | Sparse score loss `< 0.1` vs vanilla. |
| `B` | Sparse score loss `<= 0.5`. |
| `C` | Sparse score loss `<= 1.0`. |
| `D` | Sparse score loss `> 1.0`, missing aggregate score, or failed quality run. |

## Logits

Logits compares HF-reference decode outputs with SparseVLLM decode outputs when
the method declares an HF logits reference.

| Grade | Rule |
| --- | --- |
| `A` | Every decode step matches top-1, mean top-5 overlap `>= 0.8`, mean top-10 overlap `>= 0.9`, and p99 absolute difference is within the configured threshold when one is set. |
| `B` | Top-1 matches for every decode step and mean top-5 overlap `>= 0.8`, but at least one `A` condition is missed. |
| `C` | Top-1 matches for every decode step, but `B` and `A` overlap conditions are missed. |
| `D` | Missing decode-step metrics, top-1 mismatch, or failed logits run. |
| `N/A` | No HF logits reference exists for that method. |

## Performance

Performance grades decode speedup and, when required, decode CUDA graph
activation.

| Grade | Rule |
| --- | --- |
| `A` | Speedup `>= 2.0` and required decode CUDA graph is active. |
| `B` | Speedup `>= 1.5`. |
| `C` | Speedup `> 1.0`. |
| `D` | Speedup `<= 1.0`, failed performance run, or expected decode CUDA graph is inactive. |

When the performance policy sets `require_speedup=false`, speedup is recorded
but not required; the gate records `A` if the decode CUDA graph requirement is
satisfied.

## Memory

Memory compares expected and observed memory savings.

| Grade | Rule |
| --- | --- |
| `A` | Observed saving is positive and absolute error from expected saving is `<= 0.05`. |
| `B` | Observed saving is positive and absolute error is `<= 0.10`. |
| `C` | Observed saving is positive and absolute error is `<= 0.20`. |
| `D` | Missing memory accounting, non-positive observed saving, or absolute error `> 0.20`. |

## Stress

Stress grades the fixed-length admission/decode stress test.

| Grade | Rule |
| --- | --- |
| `A` | Completed, no crash, no preemptions, full admission window reached, and utilization is OK. |
| `B` | Completed with no preemptions, but at least one `A` condition is not fully met. |
| `C` | Completed with one or more preemptions. |
| `D` | Crashed, stuck, failed rows were emitted, or the run did not finish. |

## Stress V2

Stress V2 grades synthetic serving-trace runs with shared-prefix and multi-turn
workloads.

| Grade | Rule |
| --- | --- |
| `A` | All cases succeed, at least one prefix-cache case observes cache hits, prompt lengths vary, and minimum eligible cache-hit rate is `>= 0.80`. |
| `B` | All `A` structural conditions hold and minimum eligible cache-hit rate is `>= 0.50`. |
| `C` | All `A` structural conditions hold but minimum eligible cache-hit rate is `< 0.50`. |
| `D` | Missing summary, no cases, any failed case, no prefix-cache-enabled case, no observed prefix-cache hit, or no variable prompt lengths. |

## Worst Required Grade

The harness reports `worst_required_grade` as the minimum non-`N/A` grade across
the required gates in a run.
