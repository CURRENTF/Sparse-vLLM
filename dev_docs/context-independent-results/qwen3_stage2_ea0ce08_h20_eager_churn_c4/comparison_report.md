# Standardized LLM Efficiency Benchmark Report

- **Model**: `Qwen3-8B`
- **Tensor parallel size**: `1`
- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates
- **Timestamp**: `2026-08-25 19:13:02`

| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sparsevllm-vanilla` | oversubscribed_churn | 933-1022 | 24-32 | 4 | 3.32 | 93.33 | 100.0% | 100.0% | 782.41/1668.45 | 53.2% | 15.4% | 76.98 | success |
| `sparsevllm-vanilla` | oversubscribed_churn | 7383-8141 | 24-32 | 4 | 0.87 | 24.50 | 100.0% | 100.0% | 4587.49/8506.29 | 88.6% | 10.8% | 76.98 | success |
