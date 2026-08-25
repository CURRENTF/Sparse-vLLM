# Standardized LLM Efficiency Benchmark Report

- **Model**: `Qwen3-8B`
- **Tensor parallel size**: `1`
- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates
- **Timestamp**: `2026-08-25 19:11:24`

| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sparsevllm-vanilla` | fixed_batch | 1024-1024 | 32-32 | 1 | 1.09 | 35.03 | 33.9% | 100.0% | 114.17/114.33 | 31.1% | 17.9% | 76.97 | success |
| `sparsevllm-vanilla` | fixed_batch | 923-1023 | 32-32 | 4 | 3.23 | 103.28 | 100.0% | 73.7% | 420.59/426.75 | 50.0% | 15.8% | 76.97 | success |
| `sparsevllm-vanilla` | fixed_batch | 8192-8192 | 32-32 | 1 | 0.55 | 17.68 | 64.0% | 100.0% | 1011.68/1012.44 | 68.0% | 13.2% | 76.97 | success |
| `sparsevllm-vanilla` | fixed_batch | 7469-8112 | 32-32 | 4 | 0.86 | 27.62 | 100.0% | 39.1% | 3828.07/3864.70 | 87.4% | 11.4% | 76.97 | success |
