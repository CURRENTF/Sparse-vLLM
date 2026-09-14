# DP Attention 与 Expert Parallelism

GLM-4.7-Flash 支持单机 DP attention：每张 GPU 处理自己的请求和 KV cache，
routed experts 则跨卡分片。配置要求 `tensor_parallel_size=1`，并且
`data_parallel_size=expert_parallel_size`。DP 与 EP 共用 rank 轴，
DP=EP=2 使用两张 GPU。`data_parallel_size=1` 保留现有 TP/EP 执行方式。

Attention 与 routed experts 的并行方式需要分开描述。GLM 的两卡配置为：

| 布局 | `tensor_parallel_size` | `expert_parallel_size` | `data_parallel_size` | Attention 计算 |
| --- | ---: | ---: | ---: | --- |
| DP attention + EP | 1 | 2 | 2 | 两卡分别处理不同请求。 |
| TP attention + EP | 2 | 2 | 1 | 两卡切分同一批请求的 attention。 |
| 重复 attention 计算 + EP | 1 | 2 | 1 | 两卡对同一批请求各算一遍完整 attention。 |

三种布局的 routed experts 都在同样的两张卡上分片。选择 attention 并行方式时，
应比较 **DP attention + EP 与 TP attention + EP**；重复 attention 计算的基线
必须单独注明。这些布局同时用于 prefill 和 decode。

```python
import os
from sparsevllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(
        os.environ["MODEL_PATH"],
        tensor_parallel_size=1,
        expert_parallel_size=2,
        data_parallel_size=2,
        decode_graph=True,
        max_num_batched_tokens=2048,
        max_num_seqs_in_batch=8,
        max_decoding_seqs=8,
    )
    try:
        prompts = [
            llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for question in ["What is 2 + 3?", "中国的首都是哪里？"]
        ]
        print(llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32)))
    finally:
        llm.exit()
```

启动 Python 前用 `CUDA_VISIBLE_DEVICES` 选择参与的 GPU。
worker 通过 spawn 创建，因此脚本需要 `__main__` 入口保护。
服务的 engine 配置也可传入相同参数。

## 容量与请求归属

`max_num_batched_tokens`、`max_num_seqs_in_batch`、`max_decoding_seqs`、KV 容量
和 prefix-cache 预算均按**每副本**生效。例如，两副本各设置
`max_decoding_seqs=8`，在各自显存和调度约束允许时最多同时 decode 16 个请求。
KV 容量不能跨副本借用。`worker_load()` 返回聚合计数，`replicas` 字段保留
各副本的 cache 和负载记录。

新请求优先选择负载最小的副本，radix prefix 命中用于打破负载相同的候选之间的
平局。请求接收后固定在原副本，取消和 chain-cache 续接也遵循相同归属。
一个副本中的缓存前缀不会自动复制到其他副本。前端支持 `generate`、请求接收和
取消、`step` 以及异步服务 dispatcher，不暴露单一 `scheduler` 或 `model_runner`；
集成代码应使用公开引擎方法。

## 通信与 decode graph

DP 路径默认使用 All-Gather / Reduce-Scatter（AG/RS）。每个 MoE block 聚合
activation，复用现有 router 和本地 expert 算子，再将求和后的结果分发回 token
所属副本。Shared experts 在所属副本执行。现有 TP/EP 继续使用 All-Reduce。

兼容的单机 2/4/8 卡配置在 prefill 和 decode 中优先使用 FlashInfer mixed communication，
不再受设备名称或 batch 大小的调优配置限制。要求 SM90/SM100、FP16/BF16 行按
16 字节对齐，且组内支持 multicast 和 peer access；operator report 会显示实际实现。
这个可选路径需要 `cuda-python` 和 NVIDIA NVSHMEM，单机执行也需要这两个
依赖，首次使用会编译 kernel。所有副本按统一的 token capacity 选择通信实现，
混合 prefill/decode 步骤也保持一致；超出已准备容量范围时使用 NCCL。
可选依赖缺失或硬件不兼容时在启动阶段选择 NCCL；已安装 provider 的初始化或执行
失败会明确报错，不会静默切换实现。

设置 `moe_communication_backend="all2all"` 可选择 DeepEP V1 normal NVLink
通信。`"auto"` 为 DP 保留 AG/RS 默认值，`"agrs"` 则显式选择 AG/RS。
All-to-all 继续复用现有本地 expert 算子，要求单机 EP=DP∈{2,4,8}、
attention TP=1、MoE TP=1、BF16 activation、hidden size 可被 256 整除，
且每对 EP 设备均支持 NVLink peer access。不使用 RDMA、low-latency 模式或
shared-expert 重叠。

DeepEP 为可选依赖。先安装基础运行环境，再在同一 Torch/CUDA 环境中构建：

```bash
python -m pip install --no-build-isolation -e '.[all2all]'
```

该 extra 固定使用官方 DeepEP V1.2.1 源码，并包含用于拓扑检查的
`nvidia-ml-py`。构建需要兼容的 CUDA toolkit/compiler；目标 GPU 架构的构建
设置见 [DeepEP 安装说明](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1)。
运行时接受具备所需公开 API 的 DeepEP >=1.2.1,<2。仅当选择 `all2all` 时，
才在启动阶段检查依赖、扩展兼容性和 NVLink；缺失或不兼容会明确报错，
运行中发生错误也不会静默切换为 AG/RS。Operator runtime report 会显示实际
通信 provider 和版本。Normal 模式在小 batch 下可能比 AG/RS 延迟更高，
应按相同请求负载实测选择。

Decode CUDA Graph 在启动时捕获。各副本统一通信的 batch capacity，但可以执行
各自的长文本或短文本 attention 路径。空闲副本只参与 experts，不创建虚假请求
或 KV 状态。当某副本 prefill、另一副本 decode 时，该步使用 eager；后续纯 decode
步骤恢复 graph replay，不需要运行时重新捕获。采样在 graph 外执行，DP 模式会
拒绝 `decode_graph_capture_sampling=True`。

## 稀疏方法与 prefix cache

GLM DP 支持 Vanilla、StreamingLLM、SnapKV、H2O、OmniKV、QuEST、R-KV，
并保留各方法原有的模型约束。Vanilla/OmniKV 支持 radix prefix cache，
StreamingLLM/SnapKV/H2O/R-KV 支持 chain cache。GLM QuEST 仍不支持 prefix
caching。稀疏选择与物理缓存生命周期由各副本独立维护。

Prefix 检查、子树删除和 eviction priority 更新通过 `replicas` 返回各副本结果。
DP 模式暂不支持基于评分的 prefix-prune job。Efficiency probe 的请求模式支持
DP；连续 decode 测量以及直接访问单一 scheduler 的工具需要另外接入 DP。

本实现要求 attention TP=1、MoE TP=1，尚未提供多机执行、DeepEP V2 或
DeepSeek V4 支持。AG/RS 会复制 token 分发，收益取决于负载均衡和 attention
计算量；比较 DP+EP 与 TP+EP 时应对齐全局请求和 token 预算。不同并行布局或通信后端的 BF16
结果不保证逐位相同。
