# Standardized LLM Efficiency Benchmark Report

- **Model**: `Qwen3-8B`
- **Tensor parallel size**: `1`
- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates
- **Timestamp**: `2026-08-25 19:07:53`

| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sparsevllm-vanilla` | fixed_batch | 1024-1024 | 32-32 | 1 | 3.15 | 100.76 | 50.1% | 100.0% | 114.54/114.87 | 93.4% | 61.6% | 77.10 | success |
| `sparsevllm-vanilla` | fixed_batch | 923-1023 | 32-32 | 4 | 6.29 | 201.26 | 100.0% | 49.9% | 420.92/427.03 | 96.0% | 27.8% | 77.10 | success |
| `sparsevllm-vanilla` | fixed_batch | 8192-8192 | 32-32 | 1 | 0.81 | 25.91 | 82.9% | 100.0% | 1013.10/1013.36 | 97.7% | 23.6% | 77.10 | success |
| `sparsevllm-vanilla` | fixed_batch | 7469-8112 | 32-32 | 4 | 0.98 | 31.24 | 100.0% | 30.1% | 3831.01/3868.23 | 99.5% | 11.4% | 77.10 | success |
