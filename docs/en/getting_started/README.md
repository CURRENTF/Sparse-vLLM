# Getting Started

This page covers environment setup, checkpoint download, and a minimal
Sparse-vLLM usage example.


## Install with Conda

```bash
conda create -n sparse-vllm-cu130-py312 python=3.12 -y
conda activate sparse-vllm-cu130-py312

python -m pip config --site set global.extra-index-url \
  "https://download.pytorch.org/whl/cu130 https://flashinfer.ai/whl/cu130"
python -m pip install "transformers==5.13.1" -e ".[cu130]"
python -m pip check

# Optional
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

## Install with uv

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install -e ".[cu130]"

# Optional
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
```


Use `cu129` instead of `cu130` for CUDA 12.9.

`einops`, `sglang-kernel`, and the training, benchmark, and test packages are all
runtime dependencies, so workflow-specific extras are not required.

Sparse-vLLM supports Qwen3.5/Qwen3.6/Qwen3.8 checkpoints in unquantized BF16
and block-scaled FP8 formats. All three share the `qwen3_5` runtime architecture
and support matrix.

Its causal Conv1D and decode packing paths remain repo-owned Triton kernels.
The GDN core is resolved once during model preparation: validated H100 runs
with `flashinfer-python>=0.6.17` bind FlashInfer prefill plus the repo fused
Triton decode kernel, while other supported contracts bind the repo Triton
implementation. The lower project dependency floor remains legal because the
FlashInfer GDN Provider rejects older package versions before execution.

`flashinfer-cubin` is an optional acceleration package:

```bash
pip install flashinfer-cubin --index-url https://flashinfer.ai/whl
```

Block-scaled FP8 Linear selects an implementation from the local operator
registry using the active CUDA device capabilities. An SM90 operation matching
the BF16 block-scale contract uses the optimized FlashInfer implementation;
other supported SM90 contracts use generic Triton. Any model whose FP8 Linear
operation matches a profiled shape and contract on the RTX PRO 6000 binds the
same model-independent dispatch plan: `M < 512` uses Triton, while `M >= 512`
uses an atomic provider composed of SGL per-token-group activation quantization
and FlashInfer's public `gemm_fp8_nt_groupwise` CUTLASS kernel. Unprofiled SM120
shapes and other supported native-FP8 CUDA devices use generic Triton. The
binding report records the profile and both routes. No Hub kernel is downloaded
during warmup. The current profile was measured with Qwen3-30B TP1 shapes, but
the model name is provenance rather than a selection key.

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
- `decode_keep_tokens`: shared sparse top/important token budget.
- `full_attention_layers`: comma-separated layer indices or list of layers that run full attention.

## Documentation Map

- [Core sparse methods](../features/sparse-methods.md)
- [Benchmarks](../benchmarking/README.md)
- [DeltaKV](../features/deltakv.md)
- [Troubleshooting](troubleshooting.md)
