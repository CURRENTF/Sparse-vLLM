# GLM-4.7-Flash 支持实施计划

- 状态：Implemented（H100 TP1/TP2/TP4、EP1 已验证）
- 更新时间：2026-08-05
- 目标分支：codex/glm-4.7-flash
- 目标模型：zai-org/GLM-4.7-Flash / glm4_moe_lite

## 1. 摘要

本计划为 Sparse-vLLM 增加 GLM-4.7-Flash 的第一阶段支持。实现以 latent MLA 为基础，不通过持久化展开式 K/V 获得临时兼容，也不把 GLM 特例塞进现有显式 K/V 接口。

核心决策如下：

1. 从固定版本的 LightLLM vendor 最小 MLA Triton 数学核心，移除 LightLLM runtime 依赖，并先以独立 Torch oracle 验证。
2. 将当前 PrefillComputeView / DecodeComputeView 从固定双 K/V 改为“公共 metadata + tagged payload”。
3. 保留 CacheManager 对 slot、请求生命周期和 logical view 的所有权；由 storage strategy 管理显式 K/V 或 MLA latent 的物理张量。
4. 模型只表达 GLM 投影、部分 RoPE、K/V absorption、MoE/router/shared-expert 等语义；模型不选择或直接导入具体 kernel。
5. MLA kernel 通过 OpSpec -> OpResolver -> Provider -> kernel 路径绑定，初始化后不做运行时静默 fallback。
6. MoE 只抽取 Qwen3-MoE 中真正通用的 packed-expert 物理执行部分，不构建万能 MoE 模型框架。
7. GLM response parsing 继续委托 Transformers；本地只提供缺失的声明式 response template。

首版目标是 BF16、eager、vanilla latent MLA、TP1 和 TP4/EP1 的短上下文正确性。CUDA Graph、Prefix Cache、稀疏方法、MTP、量化和实用长上下文能力均需后续单独验证。

## 2. 目标与非目标

### 2.1 目标

- 正确加载 Glm4MoeLiteForCausalLM 的基础模型权重。
- 正确执行层 0–46；明确识别并跳过 checkpoint 中单独的 MTP 层 47。
- 使用 latent MLA cache：每层每 token 持久化 512 维 latent 和 64 维 RoPE key。
- 支持 chunked prefill 的完整历史可见性。
- 支持 padded static/eager decode，不发生 slot=-1 越界写。
- 支持 GLM 的首层 Dense、后续 routed MoE、shared expert 和 biased-sigmoid routing。
- 复用当前 TP/EP、packed expert、provider registry、linear、RMSNorm 和 SwiGLU 基础设施。
- 提供可复现、artifact-backed 的 tiny 与真实模型验证。

### 2.2 非目标

首版不包含：

- MTP/speculative decoding。
- 非 BF16 checkpoint 或量化权重。
- CUDA Graph。
- Prefix Cache 或 prefix offload。
- SnapKV、PyramidKV、OmniKV、QuEST、H2O、RKV、StreamingLLM、DeltaKV 等稀疏方法。
- EP 大于 1 的真实模型支持声明。
- 128K/202K 长上下文吞吐或显存能力声明。
- LightLLM Python package 或 runtime 依赖。
- 在 kernel 失败后切换到 Torch/其他 backend 的运行时 fallback。

## 3. 已知模型契约

官方配置来源：

- https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py

关键结构：

| 项目 | 值 |
| --- | ---: |
| model_type | glm4_moe_lite |
| 基础层数 | 47 |
| MTP 层数 | 1 |
| hidden size | 2048 |
| attention heads | 20 |
| KV heads | 20（展开语义） |
| q LoRA rank | 768 |
| kv LoRA rank | 512 |
| qk no-PE dim | 192 |
| qk RoPE dim | 64 |
| qk head dim | 256 |
| value head dim | 256 |
| routed experts | 64 |
| experts per token | 4 |
| shared experts | 1 |
| MoE intermediate size | 1536 |
| Dense intermediate size | 10240 |
| routed scaling factor | 1.8 |
| first Dense layers | 1 |
| dtype | BF16 |

必须区分以下维度：

~~~
kv_lora_rank = 512
rope_dim = 64
latent_cache_width = 576
qk_head_dim = 192 + 64 = 256
softmax_scale = 256 ** -0.5
~~~

不能从 latent cache width 576 推导 attention scale，也不能使用 Transformers 映射后的 config.head_dim=64 作为 Sparse-vLLM cache head dimension。

## 4. 当前运行时约束

当前实现从抽象层开始假定显式双 K/V：

- src/sparsevllm/engine/cache_manager/base.py 的 PrefillComputeView 和 DecodeComputeView 固定包含 k_cache / v_cache。
- CacheManager 使用 num_key_value_heads 和 head_dim 推导物理 cache shape 与容量。
- StandardCacheManager 直接分配 K/V tensor，并通过 slot table 暴露计算视图。
- TritonAttentionBackend 直接消费 view.k_cache / view.v_cache。
- eager decode 仍可能经过 static batch bucket padding；padding sequence 的 slot_mapping 可以是 -1。
- chunked prefill 的 view 表示完整可见上下文，而不仅是当前输入 chunk。

因此禁止以下临时方案：

- 把 latent 和 RoPE tensor 假装成 k_cache / v_cache。
- 在 ComputeView 中增加一组互斥的 optional tensor 字段。
- 通过未校验的 metadata 字典把 MLA tensor 旁路给 kernel。
- 只展开当前 prefill chunk，忽略已缓存历史。
- 原样复制 LightLLM 的 infer_state wrapper。

## 5. 目标架构

~~~
Glm4MoeLiteForCausalLM
  |
  +-- Glm4MoeLiteAttention
  |     |
  |     +-- q_a/q_b, kv_a/kv_b, partial RoPE, K/V absorption
  |     |
  |     +-- MLAAttention (semantic layer)
  |           |
  |           +-- AttentionComputeView[AttentionPayload]
  |           |     +-- AttentionViewMeta
  |           |     +-- MlaLatentPayload
  |           |
  |           +-- MlaAttentionOpSpec
  |           +-- OpResolver(DeviceCaps)
  |           +-- MlaTritonProvider
  |           +-- vendored LightLLM-derived Triton kernels
  |
  +-- Glm4MoeLiteMlp
        +-- Dense Qwen3MLP (layer 0)
        +-- GlmBiasedSigmoidRouter
        +-- shared PackedMoeExperts
        +-- Qwen3MLP shared expert

CacheManager
  +-- slot allocator / request lifetime / logical selection
  +-- ExplicitKVStorage | MlaLatentStorage
~~~

所有权原则：

- 模型拥有模型语义和逻辑权重。
- CacheManager 拥有 slot、请求生命周期和 logical visibility。
- Storage 拥有物理 cache tensor、store 和内存核算。
- ComputeView 描述一次 attention 计算可见的 metadata 与 payload。
- Provider 拥有 kernel-specific layout、workspace、launch config 和调用。
- Platform / DeviceCaps 拥有设备发现和能力事实。

## 6. ComputeView 改造

### 6.1 设计

建议引入：

~~~python
@dataclass(frozen=True)
class AttentionViewMeta:
    active_slots: torch.Tensor
    req_indices: torch.Tensor
    context_lens: torch.Tensor
    max_context_len: int | None = None
    attn_score: torch.Tensor | None = None
    temp_slots: torch.Tensor | None = None


@dataclass(frozen=True)
class ExplicitKVPayload:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    backend: str = "dense"
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MlaLatentPayload:
    latent_cache: torch.Tensor
    rope_cache: torch.Tensor


AttentionPayload = ExplicitKVPayload | MlaLatentPayload


@dataclass
class PrefillComputeView:
    meta: AttentionViewMeta
    payload: AttentionPayload


@dataclass
class DecodeComputeView:
    meta: AttentionViewMeta
    payload: AttentionPayload
~~~

Prefill 和 decode 保留不同类型，因为二者的执行、workspace 和生命周期不同；二者共用 payload 类型和公共 metadata。

### 6.2 不变量

- 不允许一个 view 同时包含显式 K/V 与 MLA latent。
- 通用 Attention 必须显式要求 ExplicitKVPayload。
- MLAAttention 必须显式要求 MlaLatentPayload。
- payload 类型不匹配时在进入 kernel 前失败。
- backend 和 backend-specific metadata 只属于 ExplicitKVPayload 或对应 provider，不属于公共 metadata。
- AttentionViewMeta 不携带模型权重或 kernel workspace。

### 6.3 迁移策略

先进行不改变行为的机械迁移：

~~~
view.k_cache          -> view.payload.k_cache
view.v_cache          -> view.payload.v_cache
view.active_slots     -> view.meta.active_slots
view.context_lens     -> view.meta.context_lens
view.backend          -> view.payload.backend
~~~

所有现有 manager 在该提交中只生成 ExplicitKVPayload。迁移提交不得同时引入 MLA 逻辑，以便将现有测试失败明确归因于接口改造。

## 7. Cache storage 契约

### 7.1 目标

避免创建以下类层次：

~~~
GlmStandardCacheManager
GlmSnapKVCacheManager
GlmQuestCacheManager
...
~~~

CacheManager 继续按 sparse method 选择；物理 payload 通过 composition 委托给 storage。

### 7.2 建议接口

~~~python
class AttentionCacheStorage(Protocol):
    layout: CacheLayout

    def allocate(
        self,
        *,
        num_layers: int,
        num_slots: int,
        device: torch.device,
    ) -> None: ...

    def layer_payload(self, layer_idx: int) -> AttentionPayload: ...

    def store(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        payload: Any,
    ) -> None: ...

    def bytes_per_slot_per_layer(self) -> int: ...

    def accounting_tensors(self) -> tuple[torch.Tensor, ...]: ...
~~~

具体类：

~~~
ExplicitKVStorage
  k_cache [layers, slots, kv_heads, head_dim]
  v_cache [layers, slots, kv_heads, head_dim]

MlaLatentStorage
  latent_cache [layers, slots, 1, 512]
  rope_cache   [layers, slots, 1, 64]
~~~

### 7.3 内存核算

所有以下逻辑必须从 storage 获取，不再使用全局固定公式 2 * num_kv_heads * head_dim：

- 每 slot 每 layer 字节数。
- 最大可分配 slot 数。
- prefix block 字节数。
- allocator/debug tensor 汇总。
- live-token 和 reserved-token 内存报告。

accounting_tensors() 必须显式返回 storage 持有的物理 tensor，不能依赖递归扫描任意对象。

### 7.4 Padding 安全

store() 和 vendored copy kernel 必须跳过 slot_mapping < 0。以下两种模式都必须覆盖：

- 真实 request slot。
- static/eager padded batch 的 slot=-1。

禁止依赖“关闭 CUDA Graph 后不会 padding”的假设。

## 8. LightLLM MLA kernel vendoring

### 8.1 固定上游

- 仓库：https://github.com/ModelTC/lightllm
- commit：65c174ee95ac6a6fd36b18b63d0b33d97e76b770
- license：Apache-2.0

本地目录：

~~~
src/sparsevllm/triton_kernel/mla/
~~~

### 8.2 最小来源映射

| 上游文件 | 本地计划文件 | 处理方式 |
| --- | --- | --- |
| common/basemodel/triton_kernel/mla_att/decode_att/gqa_flash_decoding_stage1.py | decode_stage1.py | 保留 Triton 数学核心，改为显式 tensor/stride/config API |
| common/basemodel/triton_kernel/mla_att/decode_att/gqa_flash_decoding_stage2.py | decode_stage2.py | 保留归并数学核心，移除 LightLLM 状态依赖 |
| common/basemodel/triton_kernel/mla_att/decode_att/gqa_flash_decoding.py | decode_schedule.py | 不原样复制；仅参考 stage 调度，本地重写 workspace/launch API |
| common/basemodel/triton_kernel/kv_copy/mla_copy_kv.py | copy_latent.py | 增加 slot=-1 mask 与本地 layout 校验 |
| models/deepseek2/triton_kernel/sample_kv.py | gather_latent.py | 适配 slot-indirected ragged/full-history gather |

每个衍生文件必须：

- 保留或补充 Apache-2.0 SPDX 标识。
- 注明原始仓库、固定 commit 和原文件路径。
- 注明本地修改。
- 不 import lightllm。

目录同时包含：

~~~
README.md
LICENSE.lightllm
~~~

### 8.3 明确不复制

- LightLLM infer_state。
- request manager / memory manager。
- 动态挂载到 infer_state 的 workspace。
- frozendict。
- LightLLM global kernel config 或 autotuner。
- LightLLM device utilities。
- forward 内的动态 vSM 探测和 allocation。
- context_flashattention_nopad_with_v.py 作为 GLM 首版 prefill kernel。

最后一项不能直接用于 GLM：其契约要求 q_nope_dim == v_dim，而 GLM 是 192 与 256。

### 8.4 Vendor kernel API

Decode 输入：

~~~
q_nope_absorbed [batch, local_heads, 512]
q_rope          [batch, local_heads, 64]
latent_cache    [slots, 1, 512]
rope_cache      [slots, 1, 64]
active_slots    [rows, max_context_len]
req_indices     [batch]
context_lens    [batch]
softmax_scale   scalar, fixed from qk_head_dim=256
workspace       explicit tensors
launch_config   immutable value object
~~~

输出：

~~~
latent_output [batch, local_heads, 512]
~~~

Kernel 不读取 HF config、model type、环境变量或 backend 名称。

## 9. MLA operator/provider

新增：

~~~
src/sparsevllm/operators/mla_attention.py
~~~

建议 spec：

~~~python
@dataclass(frozen=True)
class MlaAttentionOpSpec:
    num_q_heads: int
    kv_lora_rank: int
    rope_dim: int
    qk_head_dim: int
    value_head_dim: int
    activation_dtype: torch.dtype
    cache_dtype: torch.dtype
    tp_size: int
    cuda_graph: bool
~~~

softmax_scale 可以由经过校验的 qk_head_dim 得出，也可作为只读规范化字段保存；禁止从 kv_lora_rank + rope_dim 得出。

Provider 选择链：

~~~
MLAAttention semantic call
  -> MlaAttentionOpSpec
  -> OpResolver(DeviceCaps)
  -> MlaTritonProvider
  -> prepare workspace/layout
  -> decode_stage1/decode_stage2
~~~

约束：

- provider 在 model construction/ModelRunner 初始化阶段绑定一次。
- forward hot path 不 resolve provider。
- supports(spec, caps) 检查 platform、Triton、dtype、维度、TP、graph 和已验证硬件。
- 未验证的 GPU 架构不宣称支持。
- 首版只有一个 production provider；Torch 只用于测试 oracle。
- kernel/JIT 失败直接暴露，不 catch 后切换实现。
- Provider 使用 Platform / DeviceCaps，通用代码不直接探测 torch.cuda。

LightLLM 固定 commit 没有目标 H100/GLM shape 的现成 tuning config，因此首版使用明确、安全的静态 launch config；只有 H100 数值和性能实测后才将该 shape 标记为已验证。

## 10. Prefill 数据路径

MLA persistent cache 只保存 latent/rope，但 chunked prefill 必须看到完整历史。

正确顺序：

1. 当前 chunk 计算 q_a/q_b 和 kv_a。
2. 对当前 chunk 的 512 维 latent 做 RMSNorm。
3. 对当前 chunk 的 64 维 key 应用 RoPE。
4. 通过 padding-safe store 写入持久 latent/rope cache。
5. 根据 active_slots + req_indices + context_lens gather 每条序列的完整历史 latent/rope，包括之前 chunk 和当前 chunk。
6. 使用 kv_b_proj 将完整历史 latent 临时展开为：
   - k_nope [visible_tokens, local_heads, 192]
   - v [visible_tokens, local_heads, 256]
7. 将历史 RoPE key 扩展/映射到 local heads，与 k_nope 形成 256 维 K。
8. 使用 Sparse-vLLM 现有 contiguous/slot-compatible 256 维 prefill attention kernel。
9. prefill 完成后释放 gather/expanded workspace，不将其写回持久 cache。

需要单独的临时 workset：

~~~python
@dataclass
class MlaPrefillWorkset:
    gathered_latent: torch.Tensor
    gathered_rope: torch.Tensor
    expanded_k: torch.Tensor
    expanded_v: torch.Tensor
    packed_offsets: torch.Tensor
~~~

Workspace 必须有显式显存预算和有界分配。无法满足预算时应在运行 kernel 前报错，不允许降级为缺失历史或缩短上下文。

## 11. Decode 数据路径

1. 当前 token 计算：
   - q_nope [batch, local_heads, 192]
   - q_rope [batch, local_heads, 64]
   - latent [batch, 1, 512]
   - k_rope [batch, 1, 64]
2. padding-safe store 将 latent/k_rope 写入当前 slot。
3. 使用 K_b 将 query no-PE 部分吸收到 latent 空间：

   ~~~
   q_nope_absorbed [batch, local_heads, 512]
   ~~~

4. MLA provider 直接对 MlaLatentPayload 计算 attention，输出：

   ~~~
   latent_output [batch, local_heads, 512]
   ~~~

5. 使用 V_b 将输出恢复为：

   ~~~
   value_output [batch, local_heads, 256]
   ~~~

6. flatten local heads，进入 row-parallel o_proj。

Decode 计算中的 logits 是：

~~~
q_nope_absorbed @ latent_cache.T
+ q_rope @ rope_cache.T
~~~

整体使用 256 ** -0.5 scale。

## 12. 模型实现

新增：

~~~
src/sparsevllm/models/glm4_moe_lite.py
~~~

模型文件负责：

- token embedding / final norm / LM head。
- q_a_proj、q_a_layernorm、q_b_proj。
- kv_a_proj_with_mqa、kv_a_layernorm、kv_b_proj。
- 192+64 partial/interleaved RoPE。
- K/V absorption 和 V reconstruction。
- Dense/MoE layer topology。
- checkpoint 逻辑名称映射。

模型文件不负责：

- import vendored kernel。
- 选择 provider。
- 探测 GPU。
- 分配持久 cache。
- 维护 slot table。
- 运行时 backend fallback。

TP 布局建议：

- q_a_proj、kv_a_proj_with_mqa 和对应 low-rank norm：replicated。
- q_b_proj、kv_b_proj：按 local attention heads 做 column/local-head shard。
- o_proj：row parallel。
- 每 rank 持有本 rank 的 K_b/V_b head slices。
- persistent latent/rope cache 在各 TP rank 上语义一致。

首版门禁以 TP1 和 TP4 为目标；本轮同时补测 TP2。支持声明只列入真实
checkpoint 已实测的 `TP=1/2/4`，不从配置可构造性外推其他 TP 规模。

## 13. MoE 复用边界

### 13.1 抽取内容

从 src/sparsevllm/models/qwen3_moe.py 抽取 packed-expert 物理执行：

- W13/W2 storage。
- unquantized TP intermediate shard。
- EP local expert ownership。
- MoeOpSpec 和 provider resolution。
- projection loading。
- local expert completeness validation。

建议形成共享 PackedMoeExperts，由 Qwen3-MoE 与 GLM 分别组合。

### 13.2 不抽取内容

首轮不抽取：

- Qwen softmax router。
- 整个 Qwen MoE block。
- model-specific chunking/all-reduce/debug。
- checkpoint regex。
- remote-expert skip bookkeeping。
- MiniMax packed-expert 实现。

这些内容保留在模型层，以避免为了两个不同模型构建过度通用的 MoE framework。

### 13.3 GLM MoE

~~~
layer 0:
  Qwen3MLP(hidden=2048, intermediate=10240)

layer 1-46:
  routed = PackedMoeExperts(x, topk_ids, topk_weights)
  shared = Qwen3MLP(hidden=2048, intermediate=1536)(x)
  output = world_reduce(routed) + shared
~~~

Router 必须复现：

1. FP32 gate linear。
2. sigmoid。
3. correction bias 只影响 expert selection。
4. 从原始 sigmoid score gather selected weights。
5. Top-4 normalization。
6. 乘 routed_scaling_factor=1.8。

当前 MiniMax router kernel 硬编码 256 experts/Top-8。将其泛化为支持 GLM 64/Top-4 应作为独立提交；不复制一个 GLM 专属基础 kernel，也不在 packed-expert 抽取提交中顺带迁移 MiniMax storage。

## 14. 权重加载与 MTP 边界

Loader 要求：

- 所有基础层 0–46 权重必须被消费。
- 所有本地 EP experts 的 gate/up/down 权重必须完整。
- 只允许跳过其他 EP rank 持有的 expert 权重。
- e_score_correction_bias 保持 FP32。
- 只允许精确跳过 model.layers.47.* MTP 权重。
- 任何其他 missing/unexpected tensor 直接失败。

加载结束时记录：

- loaded tensor count。
- local expert shard count。
- remote EP skipped count。
- MTP skipped count。
- 其他 skipped/missing/unexpected count。

MTP 不得通过宽泛 silent ignore 机制掩盖错误层数或命名变化。

## 15. Response parsing

Sparse-vLLM 继续使用 Transformers parser：

~~~
tokenizer.parse_response()        # 非流式
tokenizer.get_response_parser()   # 流式
~~~

官方 GLM tokenizer 当前缺少 response_template，因此本地仅增加声明式 fallback，描述：

- <think>...</think>。
- <tool_call>function-name ... </tool_call>。
- <arg_key>...</arg_key>。
- <arg_value>...</arg_value>。

禁止实现另一套 XML parser、streaming state machine 或 tool-call parser。

CLI 建议新增通用 auto；现有 qwen3、minimax_m2 保留兼容，glm47 仅作为 alias，不选择另一套 parser 实现。

## 16. 首版兼容矩阵

| 能力 | 首版状态 | 门禁 |
| --- | --- | --- |
| BF16 | 已实现并验证 | tiny + 真实权重验证 |
| NVIDIA H100 80GB HBM3 | 已实现并验证 | `triton_h100` provider 实机门禁 |
| eager | 已实现并验证 | 必须显式启用 |
| vanilla latent MLA | 已实现并验证 | kernel/storage/model 全链路 |
| TP1 | 已实现并验证 | 真实模型 |
| TP2/EP1 | 已实现并验证 | 真实模型 32 步 greedy 对齐 |
| TP4/EP1 | 已实现并验证 | 真实模型、multi-chunk、ragged batch 与 API 并发 |
| EP > 1 | 暂不支持 | 后续独立验证 |
| CUDA Graph | 暂不支持 | 配置阶段拒绝 |
| Prefix Cache/offload | 暂不支持 | 配置阶段拒绝 |
| sparse methods | 暂不支持 | method registry 显式拒绝 |
| MTP | 暂不支持 | loader 精确跳过 |
| quantization | 暂不支持 | checkpoint config 拒绝 |
| 128K/202K | 未验证 | 不写入支持声明 |

配置校验必须发生在模型构造和 cache 分配之前，给出具体不兼容原因。

## 17. 分阶段执行计划

### 阶段 0：同步与基线

工作：

- 同步当前分支与目标 main 基线。
- 确认 worktree 干净。
- 记录 Python、Torch、Transformers、Triton、CUDA 和 GPU 信息。
- 校验本地 checkpoint 是否具备 48 个 shard、index 和可追踪 revision/hash。
- 将 Hugging Face、Xet、Triton、TorchInductor、临时文件和实验输出路由到
  非系统盘 `${SPARSEVLLM_ARTIFACT_ROOT}`。

门禁：

- 不覆盖现有 partial checkpoint。
- checkpoint 不完整时不运行真实模型加载。
- GPU 任务前重新检查全部设备，仅使用空闲设备。

### 阶段 1：Vendor 独立 MLA kernels

首个提交：

~~~
feat: vendor lightllm mla kernels
~~~

内容：

~~~
src/sparsevllm/triton_kernel/mla/
  __init__.py
  decode_stage1.py
  decode_stage2.py
  decode_schedule.py
  copy_latent.py
  gather_latent.py
  README.md
  LICENSE.lightllm
tests/test_mla_kernels.py
~~~

门禁：

- 零 LightLLM runtime import。
- 所有函数使用显式 tensor/stride/workspace/config API。
- copy 正确跳过 slot=-1。
- gather 支持 ragged、non-contiguous、完整历史和 padded rows。
- Torch oracle 覆盖 local heads 5/10/20 和边界长度。
- 空闲 H100 数值 smoke 通过后才合入。

该阶段不接 Model、CacheManager 或 OpenAI serving。

### 阶段 2：ComputeView tagged payload 迁移

工作：

- 引入 AttentionViewMeta、ExplicitKVPayload、MlaLatentPayload。
- 将现有 prefill/decode view 改为 meta + payload。
- 迁移所有现有 manager/backend/tests 到 ExplicitKVPayload。

门禁：

- 不加入 MLA 执行逻辑。
- 当前 explicit-KV 测试全部保持通过。
- git diff 中不存在 optional payload field bag。
- 通用 backend 在 payload 错型时 fail fast。

### 阶段 3：Storage composition

工作：

- 引入 ExplicitKVStorage 和 MlaLatentStorage。
- StandardCacheManager 委托 allocation、store、layer payload、bytes 和 accounting。
- 现有 sparse managers 首版只接受 explicit layout。

门禁：

- explicit storage 与迁移前分配量和结果一致。
- MLA storage 容量按 576 BF16 values/token/layer 计算。
- slot reuse/free/padding 不泄漏旧数据。
- 所有不支持组合在配置阶段拒绝。

### 阶段 4：MLA operator/provider 与 decode

工作：

- 新增 MlaAttentionOpSpec、registry 和 Triton provider。
- 初始化时绑定 provider。
- 构造显式 workspace 和安全 launch config。
- 打通 absorbed query -> latent decode -> V_b reconstruction。

门禁：

- resolver supported/rejected tests。
- 不在 forward 中 resolve 或动态选择 provider。
- softmax scale 使用 256，而不是 576。
- decode 与 Torch/Transformers component oracle 对齐。
- static padded batch 不越界。

### 阶段 5：完整历史 chunked prefill

工作：

- 保存当前 chunk latent/rope。
- gather 完整 visible history。
- 临时展开 K/V。
- 接现有 256 维 prefill attention kernel。
- 释放 workset。

门禁：

- 单 chunk 与多 chunk 输出对齐。
- 第二及后续 chunk 能看到历史 token。
- ragged batch、prefix length 和不同 chunk boundary 对齐。
- workspace 有界且在不足时明确失败。

### 阶段 6：Tiny MLA/model contract

工作：

- 扩展 tiny-random config/checkpoint generator。
- 建立两层模型：第 0 层 Dense、第 1 层 MoE。
- 与 Transformers 比较 projection、RoPE、router、attention、MLP 和 logits。

门禁：

- prefill logits 对齐。
- 多步 decode greedy token 对齐。
- TP rank-local projection slice 正确。
- 所有 logits finite。

### 阶段 7：最小 packed-expert 抽取

工作：

- 抽取 Qwen3-MoE packed expert 物理执行。
- Qwen3-MoE 使用共享实现，保持语义不变。
- GLM 准备复用相同 packed experts。

门禁：

- Qwen3-MoE 现有 config、loader、provider、TP/EP 测试全部通过。
- 不抽取 router、model block、checkpoint regex 或 MiniMax storage。

### 阶段 8：GLM MoE、模型与 loader

工作：

- 泛化 biased-sigmoid router 到 64/Top-4。
- 实现 Dense/MoE/shared-expert topology。
- 注册 model runner/config/runtime topology。
- 实现 strict loader 和 MTP skip。

门禁：

- tiny end-to-end 对齐。
- loaded/skipped/missing tensor 统计满足精确约束。
- Qwen3-MoE 和 MiniMax router 回归测试通过。

### 阶段 9：Transformers response template

工作：

- 增加 GLM declarative template fallback。
- 加入非流式与流式 OpenAI 响应测试。

门禁：

- 实际调用 Transformers parse_response / get_response_parser。
- thinking、普通 content、多 tool calls、Unicode、跨 chunk tag、畸形输出均覆盖。

### 阶段 10：真实模型验证

顺序：

1. checkpoint 完整性与 loader metadata 检查。
2. HF reference 与 Sparse-vLLM 顺序执行，避免同时占用显存。
3. TP1、eager、vanilla、短上下文 smoke。
4. TP1 component/logit/token parity。
5. 等待四张 GPU 同时空闲。
6. TP4/EP1 smoke。
7. TP1/TP4 greedy token 与 logits 对齐。
8. OpenAI chat/reasoning/tool call smoke。

首轮真实模型建议：

~~~
max_model_len: 2048-4096
prompt: 128-512 tokens
decode: 32-64 tokens
temperature: 0
dtype: bfloat16
decode_cuda_graph: false
enable_prefix_caching: false
vllm_sparse_method: ""
~~~

## 18. 测试矩阵

### 18.1 Kernel tests

- batch：1、2、8。
- local heads：5、10、20。
- context：1、31、32、33、127、128、129、255、256、257、1024、4096。
- slot：contiguous、ragged、non-contiguous、invalid duplicate rejection、padding -1。
- dtype：BF16；其他 dtype 明确拒绝或单独验证。
- stage1/stage2 分别与整体 oracle 对齐。
- workspace 最小容量和不足容量失败。

### 18.2 ComputeView/storage tests

- payload 类型不匹配失败。
- explicit path 行为不变。
- bytes-per-slot 与实际 tensor bytes 一致。
- allocate/free/reuse 不泄漏。
- accounting 覆盖所有 storage tensor。
- 不支持的 sparse/graph/prefix 配置构造前失败。

### 18.3 Component parity

- q_a/q_b projection。
- kv_a norm 和 kv_b split。
- 192+64 interleaved RoPE。
- prefill expanded K/V。
- decode K_b absorption 和 V_b reconstruction。
- biased-sigmoid ids/weights/scaling。
- routed + shared expert 输出。
- Dense/MoE layer topology。

每个测试使用显式 atol/rtol。BF16 kernel 初始可用 3e-2 量级作为探测值，但最终容差必须由误差分布和独立 reference 决定，不能通过不断放宽容差让测试通过。

### 18.4 End-to-end parity

- tiny prompt prefill logits。
- tiny 多步 greedy decode。
- 真实模型首 token logits/top-k margin。
- 真实模型短序列 greedy token。
- TP1 与 TP4 对齐。

非空输出不构成正确性证明。

## 19. 可观测性与实验产物

每次 GPU 验证至少保存：

- code commit 和 dirty status。
- 完整命令与 resolved config。
- Python/Torch/Transformers/Triton/CUDA 版本。
- GPU 型号和 device index。
- model/checkpoint revision/hash/completeness。
- selected MLA provider 与 rejection summary。
- cache layout、每 token 字节数、实际 slot capacity。
- loaded/skipped/missing weight counts。
- raw outputs、parsed outputs、per-sample status、aggregate result。
- stdout/stderr 和错误扫描。

建议通过环境变量配置非系统盘输出根目录：

~~~
${SPARSEVLLM_ARTIFACT_ROOT}/glm-4.7-flash/<run-id>/
~~~

系统盘空间不足时不得把 Hugging Face、Triton、TorchInductor 或实验临时文件写入默认 home cache。

## 20. 风险登记

| 严重度 | 风险 | 最小修正 |
| --- | --- | --- |
| P0 | chunked prefill 只展开当前 chunk，历史不可见 | 先 store，再按 active slots gather 完整历史，然后展开 |
| P0 | LightLLM copy 对 slot=-1 无 mask | vendor 时增加 padding mask 和回归测试 |
| P0 | 用 576 推导 softmax scale | OpSpec 独立记录 qk head dim 256 |
| P1 | latent 冒充 k_cache/v_cache | ComputeView tagged payload |
| P1 | model-specific CacheManager 形成组合爆炸 | CacheManager + storage composition |
| P1 | 原样复制 LightLLM wrapper | 只 vendor 数学核心，本地重写 provider/schedule |
| P1 | runtime kernel failure 后静默 fallback | 仅初始化时 resolver fallback；执行失败直接暴露 |
| P1 | 不支持组合落入 dense/default validation | GLM compatibility 在配置阶段显式拒绝 |
| P2 | prefill workspace 随历史无界增长 | 显式预算、容量估算和 fail-fast |
| P2 | H100 没有固定上游 tuning config | 安全静态 config + 实机数值/性能验证 |
| P2 | MoE 抽取范围过大 | 只抽 packed expert 物理执行 |
| P3 | vendor 来源/修改不清 | README、固定 commit、SPDX、license 和 source mapping |

## 21. 预计文件影响

| 区域 | 文件 |
| --- | --- |
| ComputeView | src/sparsevllm/engine/cache_manager/base.py、所有 view 构造/消费点 |
| Storage | src/sparsevllm/engine/cache_manager/storage/*、standard.py |
| Attention layer | src/sparsevllm/layers/mla_attention.py、attention.py |
| Operator | src/sparsevllm/operators/mla_attention.py、registry tests |
| Kernels | src/sparsevllm/triton_kernel/mla/* |
| Model | src/sparsevllm/models/glm4_moe_lite.py |
| MoE | qwen3_moe.py、共享 packed-expert 模块、router kernel |
| Config/runtime | configs/model.py、configs/runtime.py、method_registry.py |
| Dispatch/warmup | engine/model_runner.py、engine/llm_engine.py |
| Cache factory | engine/cache_manager/base.py 或后续独立 factory |
| Serving | entrypoints/openai/serving/response_parsing.py、api_server.py |
| Tiny/tests | debug/tiny_random.py、新增 kernel/model/parser tests |

## 22. 提交拆分

计划提交：

1. feat: vendor lightllm mla kernels
2. refactor: tag attention payload views
3. feat: add mla cache storage
4. feat: add mla attention provider
5. feat: add mla prefill and decode
6. refactor: share packed moe experts
7. feat: add glm4 moe lite model
8. feat: add glm response template
9. test: validate glm4 runtime parity
10. docs: document glm4 support matrix

每个提交要求：

- git diff --check 通过。
- 对应的 targeted tests 通过。
- 不混入无关重构。
- 不提交失败或不完整的真实实验结果作为成功结论。

ComputeView 迁移与 MLA 引入分开；Qwen MoE 抽取与 GLM model 引入分开；支持矩阵只在真实门禁完成后更新。

## 23. 完成定义

只有以下条件全部满足，才可标记“GLM-4.7-Flash 基础支持完成”：

- Vendor kernel 有固定来源、license、修改说明和独立 oracle tests。
- ComputeView 不再假定唯一 payload 是显式双 K/V。
- 所有原有 explicit-KV 测试保持通过。
- Persistent cache 只保存 512+64 latent/rope，不保存展开式 20-head K/V。
- Chunked prefill 包含完整可见历史。
- Padded decode 不对 slot=-1 写入。
- MLA scale 使用 256 ** -0.5。
- Provider 初始化时绑定，运行时无静默 fallback。
- Tiny model 的 attention、MoE、logits 和多步 decode 与 Transformers 对齐。
- 真实 checkpoint 除精确 MTP/远端 EP allowlist 外无 missing/unexpected weights。
- TP1/TP2/TP4、EP1 真实短序列验证通过；其他 TP/EP 规模不得外推。
- OpenAI 非流式/流式 reasoning 和 tool-call parsing 通过 Transformers parser。
- Artifact 能证明实际执行了目标 MLA provider、latent cache 和 GLM MoE 路径。
- 文档支持矩阵准确区分已实现、已验证和暂不支持。

## 24. 开始执行前检查清单

- [ ] 当前分支同步到目标 main。
- [ ] worktree 干净或已有改动已明确归属。
- [ ] LightLLM source commit 固定。
- [ ] vendor source/license manifest 准备完成。
- [ ] GPU 状态重新检查。
- [ ] 系统盘 cache/output 路径迁移到非系统盘 `${SPARSEVLLM_ARTIFACT_ROOT}`。
- [ ] 首个 kernel commit 的 Torch oracle 写好。
- [ ] ComputeView 迁移单独排期，不与 MLA 功能混合。
- [ ] checkpoint 完整性门禁准备好。
- [ ] 所有首版不支持组合有明确 config rejection。

第一步是独立完成并验证 feat: vendor lightllm mla kernels。该提交通过前，不进入 ComputeView、CacheManager 或模型实现。

## 25. 当前支持状态与验证摘要

当前里程碑已经完成 BF16、eager、vanilla latent MLA，以及
`TP=1/2/4`、`DP=1`、`EP=1` 的基础支持，硬件限定为 NVIDIA H100 80GB
HBM3。支持声明只覆盖这些实测组合。

### 25.1 已落地的架构边界

- MLA kernel 固定来源为 LightLLM commit
  `65c174ee95ac6a6fd36b18b63d0b33d97e76b770`，本地保留 Apache-2.0
  license、逐文件来源映射和修改说明。
- ComputeView 使用 tagged attention payload；explicit K/V 与 MLA latent payload
  通过类型契约区分，不使用互斥 optional tensor 或 metadata 字典旁路。
- CacheManager 继续拥有 slot、生命周期和 logical view；storage strategy 分别管理
  explicit K/V 与每层每 token `512 + 64` BF16 latent/rope cache。
- MLA provider 通过 operator resolver 在初始化时绑定；实机选择
  `triton_h100`，执行失败不会静默切换实现。
- GLM MoE 复用 Qwen3-MoE 的 packed expert 物理执行，只保留 GLM 自己的
  biased-sigmoid router、Dense/MoE topology 和 checkpoint 语义。
- Chat Completions 与 Responses API 均继续调用 Transformers response parser；
  本地只提供 GLM 声明式 template。Terminal EOS 和 stop boundary 在 dispatcher /
  detokenizer 通用边界处理，不在 GLM parser 中写模型特判。

### 25.2 验证结果

| 门禁 | 结果 | 证据 |
| --- | --- | --- |
| CPU regression | `402 passed, 33 skipped, 87 subtests passed` | 完整 GLM/MLA/MoE/OpenAI/scheduler test selection |
| CUDA kernel/operator/model regression | `57 passed` | H100 kernel、operator、attention layer 与模型测试 |
| Tiny multi-chunk TP1 | 17-token prompt、chunk size 8，实际 3 chunks；prefill 与两步 decode argmax 全匹配，max abs diff 分别为 `0.00390625`、`0.00390625`、`0.0048828125` | run manifest、raw comparison 与 stdout |
| 真实 checkpoint TP1 | 48/48 shards、9491 tensors、55.77 GiB、8832 local expert shards、精确跳过 212 个 MTP tensors；prefill + 32 decode steps argmax 全匹配 | run manifest、raw comparison 与 stdout |
| 真实 checkpoint TP2 | 每 rank 28.01 GiB，两个 rank 均加载 48/48 shards、9491 tensors；prefill + 32 decode steps 共 33/33 argmax 全匹配，greedy token 序列与 TP1 相同 | run manifest、raw comparison 与 stdout |
| 真实 checkpoint TP4/EP1 | 每 rank 14.13 GiB，四个 rank 均绑定 `triton_h100` MLA 与 Triton MoE；短序列 33/33、257-token 五段 prefill + decode 9/9 argmax 全匹配 | run manifest、raw comparison 与 stdout |
| TP4 synthetic API ragged + natural LCC | 64/256/768/1536-token synthetic API batch 连续三轮共 12/12 成功且跨轮逐字稳定；另以 1004/1023/1024/1025-token natural LCC 样本对 HF teacher-forced prefill 7/7、decode 27/28，唯一 teacher-forced argmax mismatch 为两个 HF 候选 logit 完全相等的 BF16 tie | raw/parsed responses、per-row comparison 与 request logs |
| TP4 OpenAI client gates | Chat 普通/思考、Chat SSE、Responses、Responses SSE、function tool call、streaming tool call 和 4-way concurrency 共 8/8 client gates 通过；parser 实际委托 Transformers。服务在验证后由 operator 主动停止，teardown 不作为 client gate | command/env manifest、raw parser events、parsed responses、validation status 与 request logs |

完整命令、依赖与设备版本、dirty status、raw/parsed output 和本机 artifact
路径保存在私有实验记录中，不写入公开仓库文档；上表只保留稳定的支持边界和
可复现门禁摘要。

TP1、TP2 和 TP4 的真实 checkpoint 32 步 greedy token 序列全部一致，足以证明
该短序列的独立 greedy rollout 一致。它不等价于严格逐元素 logits parity：
后续步骤及 multi-chunk case 的 BF16 raw logit max abs diff 可到 `9.0`，并带
近似全局平移；自然文本 teacher-forced 测试还观测到一次零 margin tie。当前证据
只支持“非 tie 的 argmax 与短序列 greedy 行为一致”，不支持“所有 logits 数值
严格等价”。

### 25.3 尚未完成的门禁

- CUDA Graph、Prefix Cache/offload、量化、MTP 和所有 sparse method。
- 128K/202K 长上下文容量与吞吐验证。
- `TP=5` 等其他可整除配置，以及 `EP > 1`。
- H100 以外的 GPU 架构。

这些组合在完成独立验证前必须继续 fail fast 或维持未支持声明。
