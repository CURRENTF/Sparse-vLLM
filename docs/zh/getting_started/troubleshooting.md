# 故障排查

## `SamplingParams` 不允许 greedy decode

`SamplingParams.temperature` 必须大于 `1e-10`。如需近似 greedy decode，请使用 `1e-5` 之类的极小 temperature。

## `Mixed long/short batch detected`

Sparse-vLLM 要求每一步只能运行长文本批次或短文本批次，不能混合运行。

## `Insufficient KV cache slots to admit prompt`

engine 无法为 prompt 或 prompt chunk 分配足够的 KV slot。请提高 `gpu_memory_utilization`，减小 `max_model_len` 或 batch size，或者降低 keep-token budget。
