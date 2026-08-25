# Standardized LLM Efficiency Benchmark Report

- **Model**: `Qwen3-8B`
- **Tensor parallel size**: `1`
- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates
- **Timestamp**: `2026-08-25 19:06:11`

| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sparsevllm-vanilla` | oversubscribed_churn | 936-1002 | 25-32 | 1 | 3.30 | 98.19 | 100.0% | 100.0% | 269.86/429.85 | 91.0% | 52.0% | 77.08 | success |
| `sparsevllm-vanilla` | oversubscribed_churn | 7487-8057 | 25-28 | 1 | 0.90 | 23.79 | 100.0% | 100.0% | 1478.22/2065.70 | 97.5% | 21.0% | 77.08 | success |
