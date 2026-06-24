# Sparse-vLLM Platform 抽象落地计划

## 背景

Sparse-vLLM 当前推理主链路默认运行在 CUDA/Triton 环境中。当前计划支持的平台限定为 CUDA、ROCm、NPU、CPU。要接入这些设备，不能只把字符串 `"cuda"` 替换为设备名，因为实际耦合同时存在于设备管理、分布式通信、显存估算、CUDA Graph、Triton kernel、cache manager 张量分配、sparse controller 临时张量和依赖安装等多个层面。

本计划参考 SGLang 的 platform/plugin 设计，但按 Sparse-vLLM 当前代码规模做轻量化落地：先建立清晰边界和 CUDA 等价实现，再逐步把硬编码迁移到边界内。平台抽象不负责实现所有算子；算子和 native extension 应通过独立 registry 或 backend 包接入。

## 目标

1. 让 engine 主流程通过 `current_platform` 获取设备、内存、同步、分布式 backend、graph 支持等能力，减少散落的 `torch.cuda.*` 和 `device="cuda"`。
2. 让 CUDA、ROCm、NPU、CPU 后端可以通过 in-tree 类或 out-of-tree 插件注册，不需要在主流程里持续增加 `if cuda / if npu / if rocm / if cpu`。
3. 保持抽象轻量：platform 只表达设备与运行时能力，不承载具体 attention、KV 压缩、采样、RMSNorm 等算子实现。
4. 允许底层算子和 native extension 后端专用构建、专用发布、专用依赖安装。
5. 迁移过程保持 CUDA 行为不变，失败时显式报错，不引入静默 fallback。

## 非目标

1. 不在第一阶段支持多个异构设备在同一个 engine 内混跑。
2. 不要求所有设备共享同一个 wheel 或同一套 native extension。
3. 不把 DeltaKV、SnapKV、OmniKV 等 sparse method 的算法状态迁移到 platform。
4. 不把所有 Triton kernel 一次性替换为跨平台实现。
5. 不为了抽象隐藏缺失算子；某设备不支持某能力时应明确抛错。

## SGLang 可借鉴点

SGLang 的实现提供了几个适合 Sparse-vLLM 借鉴的边界：

1. SGLang 将设备身份和基础操作拆在独立设备基础层里，例如 `device_name`、`device_type`、`get_device()`、`set_device()`、`get_available_memory()`、`empty_cache()`、`synchronize()` 和分布式 backend。
2. `SRTPlatform` 在设备操作之上提供推理子系统工厂和能力标记，例如 graph runner、KV pool、allocator、compile backend、attention backend、FP8 和 graph 支持。
3. `current_platform` 是 lazy singleton。首次访问时解析当前平台，避免模块导入阶段过早拉起厂商依赖。
4. platform plugin 通过 Python entry points 注册；设置 `SGLANG_PLATFORM` 时只加载指定插件，避免未选择的硬件包导入其 SDK 依赖。
5. 对 out-of-tree backend，SGLang 用 dispatch key 和 registry 让算子实现可以外挂；但 in-tree 旧路径仍有不少 `is_cuda()` / `is_npu()` 分支。这说明迁移应分阶段完成。

对应参考文件：

- `reference/sglang/python/sglang/srt/platforms/device_mixin.py`
- `reference/sglang/python/sglang/srt/platforms/interface.py`
- `reference/sglang/python/sglang/srt/platforms/__init__.py`
- `reference/sglang/python/sglang/srt/plugins/__init__.py`
- `reference/sglang/python/sglang/srt/layers/utils/multi_platform.py`
- `reference/sglang/docs/platforms/plugin.md`

## Sparse-vLLM 当前耦合点

### Engine 和 worker

`src/sparsevllm/engine/model_runner.py` 当前直接绑定 CUDA：

- `torch.cuda.set_device(rank)`
- `dist.init_process_group("nccl", ...)`
- `torch.set_default_device("cuda")`
- `torch.cuda.synchronize()`
- 采样参数用 `pin_memory=True` 后 `.cuda(non_blocking=True)`
- decode graph 直接实例化 `DecodeCudaGraphRunner`

这些属于 platform 第一阶段应该接管的主路径。

### Graph capture

`src/sparsevllm/engine/decode_cuda_graph.py` 当前直接使用：

- `torch.cuda.CUDAGraph`
- `torch.cuda.graph(graph)`
- `torch.cuda.synchronize()`
- 多处 `device="cuda"`

这应抽象成 `GraphRunner` 或 `GraphBackend`，并把配置名从 CUDA 专名逐步迁移到设备无关名。

### Cache manager

`src/sparsevllm/engine/cache_manager/base.py` 用 `torch.cuda.mem_get_info()` 和 `torch.cuda.memory_stats()` 估算 KV capacity。各 cache manager 中大量张量直接用 `device="cuda"` 创建。

这里不能只放到 platform 一层解决。推荐做两件事：

1. `CacheManager` 持有 `self.device`，所有普通张量创建用 `device=self.device`。
2. 显存查询和 allocator peak 信息通过 `current_platform` 获取。

### Sparse controller

`src/sparsevllm/engine/sparse_controller.py` 创建 attention score、top-k index、mask、临时索引时大量使用 `device="cuda"`，并在模块导入时直接导入 `sparsevllm.triton_kernel.omnikv_fused`。

这部分需要分成两类：

- 普通 tensor 创建改为使用当前 batch tensor 的 device 或 `current_platform.device_type`。
- Triton 专用 fused op 移到 op registry，按后端 lazy import。

### Attention 和 Triton kernel

`src/sparsevllm/layers/attention.py` 模块导入时直接 import Triton kernel，并在同一文件内定义 `@triton.jit` 的 `store_kvcache_kernel`。

这说明 attention 需要从“直接调用 CUDA/Triton kernel”变成“调用 attention backend 接口”。platform 只负责选择默认 backend 或 dispatch key，具体实现属于 op backend。

### Profiler 和工具层

`src/sparsevllm/utils/profiler.py` 使用 `torch.cuda.is_current_stream_capturing()` 和 `torch.cuda.synchronize()`。这类同步和 capture 检测应由 platform 提供保守默认。

### 依赖安装

当前 `pyproject.toml` 把 `flash-attn`、`torch` 等作为基础依赖。后续多后端支持需要把设备相关依赖拆到 extras 或平台插件包中，避免 NPU/ROCm 环境安装 CUDA-only 依赖失败。

## 总体设计

推荐形成三层边界：

```text
engine / scheduler / model / cache-manager
        |
        v
current_platform      op_registry / backend_registry
        |                         |
        v                         v
cuda/rocm/npu platform        cuda-triton/npu/vendor ops
```

原则：

1. engine 主流程只依赖 `current_platform` 和 registry，不直接导入厂商 SDK。
2. platform 可以返回默认 backend 名称，但不实现具体 attention kernel。
3. op backend 可以后端专用，并可由插件包注册。
4. sparse method 仍归 cache manager 和 sparse controller 管理，不归 platform。
5. 不支持时抛 `NotImplementedError` 或明确 `RuntimeError`，不自动降级到慢路径，除非配置显式允许。

## 建议目录结构

```text
src/sparsevllm/platforms/
  __init__.py              # current_platform lazy singleton
  interface.py             # Platform 基类
  device_runtime.py        # 设备 runtime API 的薄封装
  cuda.py                  # in-tree CUDA 实现
  rocm.py                  # in-tree ROCm 实现，复用 CUDA-like torch API
  cpu.py                   # 可选：仅用于 import/unit test，不承诺完整推理

src/sparsevllm/ops/
  registry.py              # op/backend 注册
  attention.py             # attention backend 接口
  cuda_triton/
    attention.py
    kvcache.py
    deltakv.py
  native/
    attention.py           # 可选：测试或 fallback，但必须显式选择

src/sparsevllm/engine/graph/
  base.py                  # DecodeGraphRunner 接口
  cuda.py                  # 现有 DecodeCudaGraphRunner 迁入或包装
```

可选的外部插件包结构：

```text
sparsevllm_npu/
  pyproject.toml
  sparsevllm_npu/
    __init__.py            # activate()
    platform.py            # NpuPlatform
    ops.py                 # register_ops()
    graph.py               # NPU graph runner, 如果支持
```

## Platform 职责边界

### Platform 应包含

#### 设备身份

- `name`: `"cuda"`、`"rocm"`、`"npu"`、`"cpu"` 等。
- `device_type`: PyTorch 设备字符串。ROCm 在 PyTorch 中通常仍是 `"cuda"`。
- `is_cuda()`、`is_rocm()`、`is_npu()`、`is_cuda_alike()` 等便捷判断。
- `get_device(local_rank) -> torch.device`

#### 设备初始化和生命周期

- `check_available() -> bool`
- `validate_environment() -> None`
- `init_backend() -> None`
- `set_device(device) -> None`
- `inference_mode()`
- `seed_everything(seed)`

#### 内存和同步

- `get_total_memory(device_id=0) -> int`
- `get_available_memory(device_id=0) -> tuple[int, int]`
- `get_allocator_peak_stats(device=None) -> AllocatorStats`
- `empty_cache() -> None`
- `synchronize() -> None`
- `is_stream_capturing() -> bool`

#### 分布式

- `get_distributed_backend() -> str`
- `barrier_device_ids(rank) -> list[int] | None`
- `get_communicator_cls() -> type | None`

#### 能力标记

- `supports_graph_capture() -> bool`
- `supports_torch_compile() -> bool`
- `supports_triton() -> bool`
- `supports_pin_memory() -> bool`
- `supports_fp8() -> bool`
- `supports_bfloat16() -> bool`

#### 子系统选择

- `get_default_attention_backend() -> str`
- `get_decode_graph_runner_cls() -> type | None`
- `get_dispatch_key() -> str`
- `apply_config_defaults(config) -> None`
- `validate_config(config) -> None`

### Platform 不应包含

1. 不包含具体 attention / RMSNorm / sampling / DeltaKV reconstruction kernel 实现。
2. 不包含 sparse method 的调度语义、cache layout、eviction 策略。
3. 不包含模型结构分支，例如 Qwen2/Qwen3 的模型选择。
4. 不包含 benchmark 或实验参数解释。
5. 不通过 broad try/except 静默修复缺失依赖。

## 接口草案

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class PlatformEnum(Enum):
    CUDA = auto()
    ROCM = auto()
    NPU = auto()
    CPU = auto()


@dataclass(frozen=True)
class AllocatorStats:
    peak_allocated_bytes: int
    current_allocated_bytes: int


class Platform:
    name: str = "unknown"
    device_type: str = "cpu"
    enum: PlatformEnum = PlatformEnum.CPU
    supported_quantization: tuple[str, ...] = ()

    def check_available(self) -> bool:
        return False

    def validate_environment(self) -> None:
        pass

    def init_backend(self) -> None:
        pass

    def get_device(self, local_rank: int = 0) -> torch.device:
        raise NotImplementedError

    def set_device(self, device: torch.device) -> None:
        raise NotImplementedError

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        raise NotImplementedError

    def get_allocator_stats(self, device: torch.device | None = None) -> AllocatorStats:
        return AllocatorStats(peak_allocated_bytes=0, current_allocated_bytes=0)

    def empty_cache(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def is_stream_capturing(self) -> bool:
        return False

    def get_distributed_backend(self) -> str:
        return "gloo"

    def supports_graph_capture(self) -> bool:
        return False

    def supports_triton(self) -> bool:
        return False

    def supports_pin_memory(self) -> bool:
        return True

    def get_default_attention_backend(self) -> str:
        return "native"

    def get_decode_graph_runner_cls(self):
        return None

    def get_dispatch_key(self) -> str:
        return self.name

    def apply_config_defaults(self, config) -> None:
        pass

    def validate_config(self, config) -> None:
        pass
```

CUDA 实现第一版只需要覆盖现有行为：

```python
class CudaPlatform(Platform):
    name = "cuda"
    device_type = "cuda"
    enum = PlatformEnum.CUDA

    def check_available(self) -> bool:
        return bool(torch.cuda.is_available() and torch.version.hip is None)

    def get_device(self, local_rank: int = 0) -> torch.device:
        return torch.device("cuda", local_rank)

    def set_device(self, device: torch.device) -> None:
        torch.cuda.set_device(device)

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.cuda.mem_get_info(device_id)

    def get_allocator_stats(self, device=None) -> AllocatorStats:
        stats = torch.cuda.memory_stats(device)
        return AllocatorStats(
            peak_allocated_bytes=int(stats["allocated_bytes.all.peak"]),
            current_allocated_bytes=int(stats["allocated_bytes.all.current"]),
        )

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_stream_capturing(self) -> bool:
        return torch.cuda.is_current_stream_capturing()

    def get_distributed_backend(self) -> str:
        return "nccl"

    def supports_graph_capture(self) -> bool:
        return True

    def supports_triton(self) -> bool:
        return True

    def get_default_attention_backend(self) -> str:
        return "cuda_triton"
```

## Platform 发现和插件机制

建议使用 Python entry points：

```toml
[project.entry-points."sparsevllm.platforms"]
npu = "sparsevllm_npu:activate"

[project.entry-points."sparsevllm.ops"]
npu_ops = "sparsevllm_npu.ops:register_ops"
```

环境变量：

- `SPARSEVLLM_PLATFORM`: 指定平台插件名。设置后只加载该 platform entry point。
- `SPARSEVLLM_PLUGINS`: 可选，限制加载哪些通用插件或 op 插件。

发现流程：

```text
访问 current_platform
  |
  |-- SPARSEVLLM_PLATFORM 已设置
  |     |-- 只查 entry point 元数据
  |     |-- 只 load 指定插件
  |     |-- activate() 返回 Platform 类路径或对象
  |     |-- 不可用则 RuntimeError
  |
  |-- SPARSEVLLM_PLATFORM 未设置
        |-- 优先检查 in-tree CUDA
        |-- 再检查 in-tree ROCm
        |-- 再尝试已安装 platform plugin，例如 NPU
        |-- 多个 plugin 同时可用则 RuntimeError，要求显式选择
        |-- 显式请求 CPU 时使用 CpuPlatform
        |-- 都不可用则 RuntimeError
```

注意：插件的 `activate()` 必须轻量，尽量只做硬件可用性检查。厂商重依赖应放在平台类方法或 op 注册函数内部 lazy import，避免未选择平台时导入失败。

## Op registry 设计

platform 只返回 dispatch key 或默认 backend；具体 kernel 由 op registry 管。

```python
class AttentionBackend:
    name: str

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        raise NotImplementedError

    def prefill_attention(
        self,
        q,
        k_cache,
        v_cache,
        out,
        *,
        req_indices,
        cu_seqlens_q,
        context_lens,
        prompt_cache_lens,
        max_input_len,
        active_slots,
        attn_score,
    ) -> None:
        raise NotImplementedError

    def decode_attention(
        self,
        q,
        k_cache,
        v_cache,
        out,
        *,
        active_slots,
        req_indices,
        context_lens,
        attn_score,
        num_heads,
        num_kv_heads,
        head_dim,
    ) -> None:
        raise NotImplementedError
```

调用侧：

```python
backend = get_attention_backend(current_platform.get_default_attention_backend())
backend.store_kvcache(...)
backend.prefill_attention(...)
backend.decode_attention(...)
```

这样 `Attention.forward` 只处理通用流程：写 KV、获取 read view、调用 backend、通知 sparse controller/cache manager。Triton、NPU vendor op、torch native 实现都在各自 backend 内。

## 配置命名迁移

当前配置中有 CUDA 专名：

- `decode_cuda_graph`
- `decode_cuda_graph_capture_sampling`
- `decode_cuda_graph_capture_sizes`
- `omnikv_decode_cuda_graph`
- `gpu_memory_utilization`
- 环境变量 `CUDA_SYNC_SVLLM`

建议新增设备无关名称，并保留旧名作为兼容 alias：

| 旧名 | 新名 | 迁移方式 |
| --- | --- | --- |
| `decode_cuda_graph` | `decode_graph` | 旧名继续可用，内部归一到新名 |
| `decode_cuda_graph_capture_sampling` | `decode_graph_capture_sampling` | 同上 |
| `decode_cuda_graph_capture_sizes` | `decode_graph_capture_sizes` | 同上 |
| `omnikv_decode_cuda_graph` | `omnikv_decode_graph` | 同上 |
| `CUDA_SYNC_SVLLM` | `SPARSEVLLM_SYNC_DEVICE` | 保留旧环境变量兼容 |
| `gpu_memory_utilization` | `device_memory_utilization` | 可后置迁移，避免破坏 benchmark 参数 |

第一阶段可以只在文档和内部变量中引入新名，不立即删除旧名。

## 依赖和包拆分

短期：

- 保持现有安装方式不变。
- 新 platform/plugin 代码必须 lazy import 设备专用依赖。
- CI 仍以 CUDA 路径为主。

中期：

- 把 CUDA-only 依赖放到 extras，例如 `sparsevllm[cuda]`。
- ROCm/NPU 等用独立插件包承载依赖，例如 `sparsevllm-npu`、`sparsevllm-rocm`。
- op backend 插件负责注册自己的 kernel，不要求主包安装所有厂商 SDK。

推荐分层：

```text
sparsevllm                 # 纯 Python core + CUDA 兼容路径
sparsevllm-cuda-ops         # 可选，CUDA/Triton/C++ 扩展
sparsevllm-rocm             # ROCm 平台和 ops
sparsevllm-npu              # NPU 平台和 ops
```

如果继续把 CUDA ops 留在主包，也应确保非 CUDA 平台 import `sparsevllm` 时不会立刻 import Triton/CUDA-only 模块。

## 分阶段迁移计划

### Phase 0：冻结现状和确定边界

产出：

- 增加 CUDA 耦合 inventory 文档或 issue。
- 明确第一批必须通过 platform 访问的 API。
- 为当前 CUDA 行为补充最小回归测试。

验收：

- 不改运行行为。
- 能列出主路径中哪些文件仍直接依赖 CUDA。

### Phase 1：引入 platform skeleton 和 CUDA 实现

新增：

- `src/sparsevllm/platforms/interface.py`
- `src/sparsevllm/platforms/device_runtime.py`
- `src/sparsevllm/platforms/cuda.py`
- `src/sparsevllm/platforms/rocm.py`
- `src/sparsevllm/platforms/__init__.py`

行为：

- `current_platform` 默认解析 CUDA。
- ROCm 可先只识别 `torch.cuda.is_available() and torch.version.hip is not None`，不承诺完整推理。
- 未支持平台使用 base platform，关键方法 `NotImplementedError`。

验收：

- CUDA 推理行为不变。
- import `sparsevllm.platforms` 不导入 Triton kernel。
- 单测可 monkeypatch `current_platform` 验证选择逻辑。

### Phase 2：迁移 engine/worker/profiler

修改：

- `ModelRunner` 使用 `current_platform.get_device(rank)` 和 `set_device()`。
- `dist.init_process_group()` 的 backend 来自 `current_platform.get_distributed_backend()`。
- `torch.set_default_device("cuda")` 改为当前 worker device type。
- `prepare_sample()` 使用 `current_platform.supports_pin_memory()` 和 `.to(device, non_blocking=...)`。
- `Profiler` 通过 platform 做 synchronize 和 stream capture 检查。

验收：

- 不再在 `model_runner.py` 中直接调用 `torch.cuda.set_device()`、`torch.cuda.synchronize()`、硬编码 `"nccl"`。
- 旧 CUDA 配置仍能跑通。

### Phase 3：迁移 cache manager 和 sparse controller 的普通张量设备

修改：

- `CacheManager.__init__` 接收或解析 `self.device`。
- `CacheManager._get_available_slots_info()` 使用 platform memory API。
- cache manager 中普通张量创建统一使用 `device=self.device`。
- sparse controller 中临时 tensor 使用 `state.context_lens.device`、`q.device` 或 `cache_manager.device`。

注意：

- 这一阶段不迁移 Triton kernel 实现。
- DeltaKV/SnapKV/Quest 等方法的 cache manager 仍保持方法所有权，只把设备创建方式改掉。

验收：

- cache manager 基类不直接依赖 `torch.cuda.mem_get_info()`。
- 标准 cache manager 的普通张量不再硬编码 `device="cuda"`。
- CUDA benchmark 结果应在正常波动范围内。

### Phase 4：引入 attention/op backend registry

新增：

- `src/sparsevllm/ops/registry.py`
- `src/sparsevllm/ops/attention.py`
- `src/sparsevllm/ops/cuda_triton/attention.py`

修改：

- `layers/attention.py` 不在模块顶层直接 import Triton kernels。
- CUDA Triton backend lazy import 当前 kernel。
- `store_kvcache_kernel` 从 `layers/attention.py` 移到 CUDA op backend。

验收：

- `Attention.forward` 不含后端判断，只调 attention backend。
- 未安装或未选择 CUDA backend 时，不因 import attention 模块直接失败。
- 如果选择平台缺少 attention backend，启动或首次执行时明确报错。

### Phase 5：Graph runner 泛化

修改：

- `DecodeCudaGraphRunner` 包装为 `CudaDecodeGraphRunner`。
- `current_platform.get_decode_graph_runner_cls()` 返回 graph runner 类。
- `Config` 内部用 `decode_graph`，旧 `decode_cuda_graph` 继续 alias。

验收：

- CUDA graph 行为不变。
- 非 CUDA 平台启用 graph 但 platform 不支持时，配置校验阶段明确报错。

### Phase 6：插件和外部后端试点

新增：

- `SPARSEVLLM_PLATFORM` 选择逻辑。
- platform entry point 文档。
- 一个最小 out-of-tree demo 插件或 in-repo test plugin。

建议先做一个 mock/test platform，而不是直接做完整 NPU：

- 用于验证插件发现、依赖隔离、dispatch key、配置默认值。
- 不承诺实际推理。

验收：

- 安装 demo plugin 后，`SPARSEVLLM_PLATFORM=demo` 可以解析到对应 platform。
- 未选择该 plugin 时，不导入它的重依赖。
- 多个 plugin 同时可用时要求显式选择。

### Phase 7：后端实现和能力矩阵

为每个真实后端建立能力矩阵：

| 能力 | CUDA | ROCm | NPU | CPU | 说明 |
| --- | --- | --- | --- | --- | --- |
| device init | yes | planned | planned | planned | platform |
| distributed | nccl | nccl/rccl | hccl | gloo | platform |
| memory stats | yes | planned | planned | psutil | platform |
| decode graph | yes | unknown | unknown | no | graph backend |
| prefill attention | triton | planned | planned | native/test only | op backend |
| decode attention | triton | planned | planned | native/test only | op backend |
| DeltaKV Triton variants | yes | unknown | no | no | op backend |
| torch.compile layernorm/sampler | yes | unknown | unknown | possible | platform capability |

每个 backend 合入前必须填写：

- 支持的 sparse method。
- 支持的模型。
- 支持的 dtype。
- 不支持能力的明确错误。
- 最小 correctness test。
- 最小性能 smoke test。

## 避免 `if gpu / if npu` 的规则

1. 设备身份判断只允许出现在 platform discovery、平台实现、op registry、配置兼容层。
2. engine 主流程不得直接写 `if current_platform.is_npu()` 来选择业务逻辑；应调用 platform factory 或 registry。
3. 算子选择通过 op backend 名称和 registry 完成。
4. sparse method 选择继续由 method registry 和 cache manager factory 管理，不与设备后端混在一起。
5. 某个后端需要替换类时，优先提供 registry/factory；不要通过 monkey patch 修改主流程。

允许的例外：

- 临时迁移期可以保留旧 CUDA 分支，但必须标注 TODO 和迁移阶段。
- 配置 alias 兼容层可以识别旧名字。
- 测试可以直接断言某平台行为。

## 测试计划

### 单元测试

- platform discovery：CUDA、ROCm、base、指定插件、多插件冲突。
- platform API：memory stats、distributed backend、device mapping、sync no-op。
- config alias：`decode_cuda_graph` 和 `decode_graph` 等价。
- op registry：注册、重复注册、缺失 backend 报错。

### 集成测试

- CUDA vanilla 短文本推理。
- CUDA sparse method smoke test：vanilla、snapkv、omnikv、quest、deltakv-triton-v4 中至少覆盖当前常用路径。
- decode graph 打开和关闭的结果一致性。
- import test：在不导入 CUDA op backend 的情况下 import `sparsevllm.layers.attention`。

### 设备插件测试

- demo plugin 安装后可被 entry point 发现。
- `SPARSEVLLM_PLATFORM` 设置后只加载指定平台。
- 未选择的插件不导入厂商依赖。

## 风险和处理

### 风险：抽象过重

处理：platform 只保留设备、能力、生命周期、factory，不放算法逻辑。attention、sampling、DeltaKV kernel 走 op registry。

### 风险：抽象后错误变成 fallback

处理：缺失能力默认抛错。只有用户显式配置 `allow_native_fallback` 之类选项时才允许 fallback，并在结果和日志中记录。

### 风险：Triton import 仍在模块顶层失败

处理：CUDA/Triton backend 必须 lazy import。核心层只 import registry/interface。

### 风险：ROCm 和 CUDA 都使用 `torch.cuda` API

处理：platform identity 与 PyTorch device type 分离。`RocmPlatform.name == "rocm"`，但 `device_type == "cuda"`。

### 风险：配置名变更破坏现有实验

处理：新增通用名，旧名长期保留 alias。日志中提示 normalized key，但不改变实验语义。

### 风险：cache manager 改动影响性能

处理：先只把 device 作为局部变量传递，不引入复杂 tensor factory。性能敏感路径避免额外 Python dispatch。

## 推荐优先级

第一批必须做：

1. `current_platform` + `CudaPlatform`。
2. `ModelRunner` 和 `Profiler` 迁移。
3. `CacheManager._get_available_slots_info()` 迁移。
4. `CacheManager.device` 传递。
5. 配置 alias 设计。

第二批再做：

1. attention backend registry。
2. `DecodeCudaGraphRunner` 泛化。
3. Triton kernel lazy import。
4. op plugin entry point。

第三批做真实后端：

1. ROCm 可用性验证。
2. NPU platform plugin skeleton。
3. NPU attention/cache/op backend 能力矩阵。
4. 按 sparse method 逐个接入，而不是一次性承诺全量支持。

## 最小落地顺序

建议按下面 PR 顺序推进：

1. `refactor: add platform skeleton`
   - 新增 platform 包和 CUDA 实现。
   - 不改主流程。

2. `refactor: route runner device ops through platform`
   - 迁移 `ModelRunner`、`Profiler`、distributed backend。

3. `refactor: pass runtime device to cache managers`
   - cache manager 拿到 `self.device`。
   - 标准 cache manager 先迁移。

4. `refactor: add attention backend registry`
   - 把 CUDA Triton attention 包成 backend。
   - `Attention.forward` 改为调用 backend。

5. `refactor: generalize decode graph runner`
   - 配置 alias 和 graph runner factory。

6. `docs: add platform plugin guide`
   - 写插件包模板和后端能力矩阵。

## 结论

Sparse-vLLM 的 platform 抽象应采用“小 platform + 独立 op backend”的设计。platform 解决设备发现、依赖校验、device/memory/sync/distributed/graph 能力选择；算子和后端优化通过 registry 或插件包注册。这样既能避免主流程里堆满 `if cuda / if npu / if rocm / if cpu`，又承认不同设备的算子实现不可避免需要专用代码。

第一阶段的成功标准不是立刻跑通 NPU，而是让 CUDA 路径通过同一个 platform 边界运行，并把后续 NPU/ROCm 接入点稳定下来。
