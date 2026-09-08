# Scripts Layout

Top-level scripts are grouped by their primary use:

- `benchmarks/`: full benchmark drivers, experiment runners, and job queues.
- `official_experiments/`: project-maintained paper experiments, adapters, configs, recorded data, and figure reproduction.
- `data/`: dataset download and background-download wrappers.
- `profiling/`: low-level performance microbenchmarks and kernel benchmarks.
- `validation/`: correctness checks, output comparisons, and implementation smoke tests.

Prefer adding new scripts to one of these folders instead of placing files directly
under `scripts/`.
