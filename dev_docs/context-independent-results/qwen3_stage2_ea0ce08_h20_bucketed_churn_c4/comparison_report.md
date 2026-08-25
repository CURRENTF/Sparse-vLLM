# Standardized LLM Efficiency Benchmark Report

- **Model**: `Qwen3-8B`
- **Tensor parallel size**: `1`
- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates
- **Timestamp**: `2026-08-25 19:10:25`

| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sparsevllm-vanilla` | oversubscribed_churn | 933-1022 | 24-32 | 4 | 6.17 | 173.69 | 100.0% | 100.0% | 567.10/1094.46 | 96.0% | 26.7% | 77.13 | success |
| `sparsevllm-vanilla` | oversubscribed_churn | 7383-8141 | 24-32 | 4 | 0.97 | 27.28 | 100.0% | 100.0% | 4408.25/8045.88 | 98.7% | 12.3% | 77.13 | success |
