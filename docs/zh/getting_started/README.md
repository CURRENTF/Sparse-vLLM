# 快速开始

本页介绍环境安装、checkpoint 下载和最小 Sparse-vLLM 使用示例。

## 使用 Conda 安装

```bash
conda create -n svllm python=3.10 -y
conda activate svllm

pip install torch==2.11.0 torchvision==0.26.0 triton==3.6.0 \
  --index-url https://download.pytorch.org/whl/cu130

# FlashInfer publishes the CUDA-specific JIT cache on a separate index.
pip install "flashinfer-jit-cache>=0.6.15" \
  --index-url https://flashinfer.ai/whl/cu130

pip install -e .

# Optional
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

## 使用 uv 安装

项目使用 CUDA 13.0 build：

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.11.0 torchvision==0.26.0 triton==3.6.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install "flashinfer-jit-cache>=0.6.15" \
  --index-url https://flashinfer.ai/whl/cu130

uv pip install -e .

# Optional
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
```

对于 Qwen3.5/Qwen3.6 mixed-attention inference，还需安装可选 Python 依赖：

```bash
# uv
uv pip install -e ".[qwen35]"

# Conda/pip
pip install -e ".[qwen35]"
```

如果只需为 vanilla、OmniKV 或 QuEST 启用 prefix-cache offload：

```bash
pip install -e ".[prefix-offload]"
```

Sparse-vLLM 当前支持未量化 BF16 和 block-scaled FP8 格式的 Qwen3.5/Qwen3.6 checkpoint。

其 prefill causal Conv1D 和 decode Conv1D/GDN packing path 使用仓库本地 Triton kernel，不需要 `sglang-kernel` 或编译仓库 CUDA extension。

必需的 `flashinfer-jit-cache` package 提供针对特定 CUDA toolkit 版本构建的 module。请选择与 PyTorch 所用 CUDA 版本匹配的 index。`flashinfer-cubin` 是可选加速 package，包含特定架构的 device binary：

```bash
pip install "flashinfer-jit-cache>=0.6.15" \
  --index-url https://flashinfer.ai/whl/cu130

# Optional
pip install flashinfer-cubin --index-url https://flashinfer.ai/whl
```

Block-scaled FP8 Linear 会根据当前 CUDA device capability，从本地 operator registry 选择实现。SM90 使用优化后的 FlashInfer 实现；其他支持原生 FP8 的 CUDA device 使用通用 Triton 实现。warmup 期间不会下载 Hub kernel。

## DeltaKV Checkpoint

基于 compressor 的 DeltaKV run 需要本地 checkpoint 目录。传入 `deltakv_checkpoint_path` 前，请下载与 base model 匹配的 compressor。

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

将下载后的本地目录用作 `deltakv_checkpoint_path`。除非 compressor 是针对另一个 base model 训练且 layer/head dimension 匹配，否则不要跨 base model 复用 compressor checkpoint。

## 最小用法

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

## 关键参数

Sparse-vLLM runtime 参数定义在 `src/sparsevllm/config.py` 中，可以作为 keyword argument 传给 `LLM(...)`。请使用规范 public name；`chunk_prefill_size`、`vllm_sparse_method`、`num_top_tokens`、`model_cls` 和 `compressor_path` 等 legacy name 会在 public runtime/API 边界被拒绝。

常用参数：

- `tensor_parallel_size`：启动的 GPU rank 数。
- `gpu_memory_utilization`：分配给 KV cache 的 GPU 总显存比例。
- `max_model_len`：允许的最大 prompt 加生成 token 数。
- `engine_prefill_chunk_size`：Sparse-vLLM prefill scheduling 与 memory admission 的 chunk size。
- `max_num_batched_tokens`、`max_num_seqs_in_batch`、`max_decoding_seqs`：scheduler 吞吐量与延迟约束。

稀疏参数：

- `sparse_method`：方法 selector。
- `deltakv_checkpoint_path`：本地 DeltaKV compressor checkpoint 目录或文件。
- `sink_keep_tokens`：始终保留的 prefix/sink token。
- `recent_keep_tokens`：始终保留的 recent tail token。
- `decode_keep_tokens`：共享的 sparse top/important token budget。
- `full_attention_layers`：运行 full attention 的 layer index，以逗号分隔的字符串或列表表示。

## 文档导航

- [核心稀疏方法](../features/sparse-methods.md)
- [基准测试](../benchmarking/README.md)
- [DeltaKV](../features/deltakv.md)
- [故障排查](troubleshooting.md)
