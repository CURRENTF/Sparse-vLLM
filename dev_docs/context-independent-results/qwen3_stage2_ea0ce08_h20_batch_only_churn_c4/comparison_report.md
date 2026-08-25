# Standardized LLM Efficiency Benchmark Report

- **Model**: `Qwen3-8B`
- **Tensor parallel size**: `1`
- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates
- **Timestamp**: `2026-08-25 19:07:06`

| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sparsevllm-vanilla` | oversubscribed_churn | 933-1022 | 24-32 | 4 | 6.22 | 175.04 | 100.0% | 100.0% | 565.52/1089.85 | 95.7% | 26.2% | 77.10 | success |
| `sparsevllm-vanilla` | oversubscribed_churn | 7383-8141 | 24-32 | 4 | 0.98 | 27.65 | 100.0% | 100.0% | 4388.50/7991.03 | 99.2% | 11.4% | 77.10 | success |
