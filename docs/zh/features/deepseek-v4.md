# DeepSeek-V4-Flash-0731（实验性）

Sparse-vLLM 为原始 DeepSeek-V4-Flash-0731 checkpoint 提供实验性原生运行时，
可执行完整 decoder、分块 prefill、decode 和连续批处理，并支持 decode CUDA Graph
及 GPU radix prefix cache。短输入和 4K 上下文的独立对照表明，相对 checkpoint
原参考实现的数值误差与 vLLM 相近。跨引擎 logits 和部分贪心预测存在差异，
不保证输出完全一致。

## 安装

使用 Python 3.12 和[快速开始](../getting_started/README.md)中的 CUDA 13.0
环境。在仓库根目录安装模型扩展依赖：

```bash
uv sync --extra cu130 --extra deepseek-v4
```

该 extra 提供所需版本的 FlashInfer 和 Hadamard transform 包。构建原生扩展需要
可用的 CUDA toolkit 和 C++ 编译器。请将原始配置、tokenizer、safetensors 索引及
全部权重分片保存在同一 checkpoint 目录。下面的聊天示例还使用 checkpoint 自带的
`encoding/encoding_dsv4.py` 格式化函数。

可选复用独立 vLLM 0.29 环境中的路由和带截断 SwiGLU 算子：将
`SPARSEVLLM_VLLM_MOE_LIBRARY` 指向该环境的
`vllm/_moe_C_stable_libtorch.abi3.so`，并保留同目录的
`_C_stable_libtorch.abi3.so` 和 wheel 元数据。
此方式只加载算子库，不导入 vLLM 引擎，也不更换 Sparse-vLLM 的 Torch 环境。
未提供该可选库时仍可使用可移植路由与激活实现；配置了不兼容的库则在启动时明确报错。

## 四卡示例

以下配置已在四张 H100 80 GB 上运行验证。启动前设置模型目录并选择四张可用 GPU：

```bash
export MODEL_ROOT="<MODEL_ROOT>"
export CUDA_VISIBLE_DEVICES=0,1,2,3
uv run --no-sync python run_deepseek_v4.py
```

将以下内容保存为 `run_deepseek_v4.py`。先按 checkpoint 格式构造对话，再传入
token ID，避免依赖通用聊天模板。

```python
import os
from pathlib import Path
import runpy

from transformers import AutoTokenizer
from sparsevllm import LLM, SamplingParams


def main():
    model_root = Path(os.environ["MODEL_ROOT"])
    encode_messages = runpy.run_path(
        str(model_root / "encoding" / "encoding_dsv4.py")
    )["encode_messages"]
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    conversation = encode_messages(
        [{"role": "user", "content": "Explain why the sky is blue."}],
        thinking_mode="chat",
    )
    prompt = tokenizer.encode(conversation, add_special_tokens=False)
    engine = LLM(
        str(model_root),
        tensor_parallel_size=1,
        data_parallel_size=4,
        expert_parallel_size=4,
        moe_backend="agrs",
        max_model_len=4352,
        max_num_batched_tokens=128,
        engine_prefill_chunk_size=128,
        mlp_chunk_size=128,
        max_num_seqs_in_batch=2,
        max_decoding_seqs=2,
        gpu_memory_utilization=0.75,
        decode_graph=True,
        enable_prefix_caching=True,
        prefix_cache_max_blocks=256,
    )
    try:
        outputs = engine.generate(
            [prompt], SamplingParams(temperature=0.0, max_tokens=32)
        )
        print(tokenizer.decode(outputs[0]["token_ids"], skip_special_tokens=True))
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
```

运行时会根据 checkpoint 自动选择 `sparse_method`；显式设置
`sparse_method="deepseek_v4"` 与自动选择等价。不能使用其他稀疏方法替换模型的
原生注意力。示例使用同步 Python API；增量请求可使用常规 `add_request`、`step`
和 `is_finished` 接口。

## 运行约束

- 保留原始 FP8 权重与 UE8M0 scale、MXFP4 专家权重、压缩配置和位置编码。
  此运行时不接受通用 BF16 转换后的 checkpoint。
- Attention TP 必须为 1，专家内部 TP 也必须为 1。因此上述 DP 部署要求
  `EP=world_size=DP`，专家数必须能被 EP 规模整除。DP 使用
  `moe_backend="agrs"`；DeepEP V1 不满足该模型的 FP32 专家归约要求。
- Prefix cache 属于各自的 DP 副本。命中时会恢复原生压缩状态及可复用的 KV 数据。
  不支持 prefix offload 或 chain cache。比较冷启动请求时可设置
  `enable_prefix_caching=False` 禁用复用。
- `max_model_len` 包含输入和生成 token。KV 容量需要同时覆盖滑动窗口、压缩历史
  及保留的前缀状态；即使还有请求槽位，也可能因缓存不足而推迟接纳。启动或接纳
  报告显存不足时，可减小上下文长度、并发或保留前缀容量。
- Eager 诊断可设置 `decode_graph=False`。本次集成不包含推测解码及 checkpoint
  的 next-token prediction 层。

该示例是容量有界的起点。数值验证覆盖 H100 上的短输入和 4K 上下文，
尚未验证 checkpoint 声明的最大上下文长度。
