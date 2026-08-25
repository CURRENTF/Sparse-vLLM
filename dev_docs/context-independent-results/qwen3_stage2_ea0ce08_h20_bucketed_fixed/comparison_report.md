# Standardized LLM Efficiency Benchmark Report

- **Model**: `Qwen3-8B`
- **Tensor parallel size**: `1`
- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates
- **Timestamp**: `2026-08-25 19:08:52`

| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sparsevllm-vanilla` | fixed_batch | 1024-1024 | 32-32 | 1 | 3.10 | 99.30 | 50.0% | 100.0% | 114.77/115.10 | 93.6% | 45.4% | 77.13 | success |
| `sparsevllm-vanilla` | fixed_batch | 923-1023 | 32-32 | 4 | 6.21 | 198.63 | 100.0% | 50.0% | 420.69/427.01 | 95.5% | 27.7% | 77.13 | success |
| `sparsevllm-vanilla` | fixed_batch | 8192-8192 | 32-32 | 1 | 0.81 | 25.81 | 83.9% | 100.0% | 1012.87/1012.91 | 98.5% | 20.8% | 77.13 | success |
| `sparsevllm-vanilla` | fixed_batch | 7469-8112 | 32-32 | 4 | 0.96 | 30.75 | 100.0% | 29.8% | 3830.88/3867.87 | 99.0% | 12.9% | 77.13 | success |
