# Troubleshooting

## `SamplingParams` Does Not Allow Greedy Decoding

`SamplingParams.temperature` must be `> 1e-10`. Use a tiny temperature such
as `1e-5` for almost-greedy decoding.

## `Mixed long/short batch detected`

Sparse-vLLM enforces that each step runs either a long-text batch or a
short-text batch, never both.

## `Insufficient KV cache slots to admit prompt`

The engine cannot allocate enough KV slots for the prompt or prompt chunk.
Increase `gpu_memory_utilization`, reduce `max_model_len` or batch size, or
reduce the keep-token budgets.
