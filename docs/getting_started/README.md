# Getting Started

This page covers environment setup, checkpoint download, and a minimal
Sparse-vLLM usage example.

## Install

```bash
conda create -n svllm python=3.10 -y
conda activate svllm
pip install torch==2.8.0 transformers[torch]==4.53.3 accelerate deepspeed==0.15.4 torchvision datasets==4.1.0 bitsandbytes
pip install fire matplotlib seaborn wandb loguru ansible
MAX_JOBS=8 pip install flash-attn --no-build-isolation
pip install -e .
```

## DeltaKV Checkpoints

Compressor-backed DeltaKV runs require a local checkpoint directory. Download
the compressor that matches the base model before passing
`deltakv_checkpoint_path`.

| Base model | Compressor checkpoint |
| --- | --- |
| `Qwen/Qwen2.5-7B-Instruct-1M` | [`JitaiHao/Qwen2.5-7B-Instruct-1M-Compressor`](https://huggingface.co/JitaiHao/Qwen2.5-7B-Instruct-1M-Compressor) |
| `Qwen/Qwen2.5-32B-Instruct` | [`JitaiHao/Qwen2.5-32B-Instruct-Compressor`](https://huggingface.co/JitaiHao/Qwen2.5-32B-Instruct-Compressor) |
| `meta-llama/Llama-3.1-8B-Instruct` | [`JitaiHao/Llama-3.1-8B-Instruct-Compressor`](https://huggingface.co/JitaiHao/Llama-3.1-8B-Instruct-Compressor) |

```bash
export DELTAKV_CKPT_ROOT=<CHECKPOINT_ROOT>/compressor
mkdir -p "$DELTAKV_CKPT_ROOT"

huggingface-cli download JitaiHao/Qwen2.5-7B-Instruct-1M-Compressor \
  --local-dir "$DELTAKV_CKPT_ROOT/Qwen2.5-7B-Instruct-1M-Compressor"

huggingface-cli download JitaiHao/Qwen2.5-32B-Instruct-Compressor \
  --local-dir "$DELTAKV_CKPT_ROOT/Qwen2.5-32B-Instruct-Compressor"

huggingface-cli download JitaiHao/Llama-3.1-8B-Instruct-Compressor \
  --local-dir "$DELTAKV_CKPT_ROOT/Llama-3.1-8B-Instruct-Compressor"
```

Use the downloaded local directory as `deltakv_checkpoint_path`. Do not reuse a
compressor checkpoint with a different base model unless it was trained for that
model and its layer/head dimensions match.

## Minimal Usage

```python
from sparsevllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen2.5-7B-Instruct-1M",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    engine_prefill_chunk_size=4096,
    sparse_method="omnikv",
    full_attention_layers="0,1,2,4,7,14",
    decode_keep_tokens=2096,
    prefill_keep_tokens=8192,
    chunk_prefill_accel_omnikv=False,
)

outputs = llm.generate(
    prompts=["Write a short story about sparse attention."],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=128),
)
print(outputs[0]["text"])
llm.exit()
```

## Key Parameters

Sparse-vLLM runtime knobs are defined in `src/sparsevllm/config.py` and can be
passed as keyword args to `LLM(...)`. Use canonical public names; legacy names
such as `chunk_prefill_size`, `vllm_sparse_method`, `num_top_tokens`,
`model_cls`, and `compressor_path` are rejected at public runtime/API
boundaries.

Common knobs:

- `tensor_parallel_size`: number of GPU ranks to spawn.
- `gpu_memory_utilization`: fraction of total GPU memory to allocate for the KV cache.
- `max_model_len`: max prompt plus generated tokens allowed.
- `engine_prefill_chunk_size`: Sparse-vLLM prefill scheduling and memory-admission chunk size.
- `max_num_batched_tokens`, `max_num_seqs_in_batch`, `max_decoding_seqs`: scheduler throughput and latency constraints.

Sparse knobs:

- `sparse_method`: method selector.
- `deltakv_checkpoint_path`: local DeltaKV compressor checkpoint directory or file.
- `sink_keep_tokens`: always-kept prefix/sink tokens.
- `recent_keep_tokens`: always-kept recent tail tokens.
- `decode_keep_tokens`: decode-time top/important token budget.
- `prefill_keep_tokens`: prefill/finalization top/important token budget.
- `full_attention_layers`: comma-separated layer indices or list of layers that run full attention.

## Documentation Map

- [Core sparse methods](../features/sparse-methods.md)
- [Benchmarks](../benchmarking/README.md)
- [DeltaKV](../features/deltakv.md)
- [Troubleshooting](troubleshooting.md)
