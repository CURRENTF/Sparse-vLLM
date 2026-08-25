# Standardized LLM Efficiency Benchmark Report

- **Model**: `Qwen3-8B`
- **Tensor parallel size**: `1`
- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates
- **Timestamp**: `2026-08-25 19:12:05`

| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sparsevllm-vanilla` | oversubscribed_churn | 936-1002 | 25-32 | 1 | 1.16 | 34.48 | 100.0% | 100.0% | 555.38/1029.54 | 29.6% | 17.6% | 76.97 | success |
| `sparsevllm-vanilla` | oversubscribed_churn | 7487-8057 | 25-28 | 1 | 0.62 | 16.55 | 100.0% | 100.0% | 1718.13/2525.23 | 66.1% | 13.8% | 76.97 | success |
