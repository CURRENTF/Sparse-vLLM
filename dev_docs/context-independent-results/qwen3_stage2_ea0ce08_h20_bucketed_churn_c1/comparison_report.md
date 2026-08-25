# Standardized LLM Efficiency Benchmark Report

- **Model**: `Qwen3-8B`
- **Tensor parallel size**: `1`
- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates
- **Timestamp**: `2026-08-25 19:09:31`

| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sparsevllm-vanilla` | oversubscribed_churn | 936-1002 | 25-32 | 1 | 3.28 | 97.55 | 100.0% | 100.0% | 270.58/430.92 | 92.0% | 58.3% | 77.08 | success |
| `sparsevllm-vanilla` | oversubscribed_churn | 7487-8057 | 25-28 | 1 | 0.89 | 23.68 | 100.0% | 100.0% | 1480.00/2071.02 | 97.7% | 20.3% | 77.08 | success |
