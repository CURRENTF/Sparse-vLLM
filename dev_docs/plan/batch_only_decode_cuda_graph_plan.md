# Decode Batch-Only CUDA Graph 原生支持计划

状态：阶段 1/2 收口中；阶段 3/4 未开始生产验收

日期：2026-08-25

范围：decode CUDA Graph、graph 输入与 metadata 生命周期、decode attention provider、稀疏 topology path 及相关 metadata

## 1. 目标与结论边界

本计划的目标是让 Sparse-vLLM 在框架和算子层面真正支持 **batch-only decode
CUDA Graph**：对于同一个 batch bucket 和 topology path，真实 context length、请求组合和
稀疏 metadata 可以在 replay 前更新，但不能改变已经 capture 的 host 执行拓扑。

本计划区分两级能力：

```text
Level 1: path-scoped batch-only（近期主要交付）
graph identity = method/runtime contract
               + batch bucket
               + finite topology_path_id
               + sampling topology

Level 2: strict batch-only
graph identity = method/runtime contract
               + batch bucket
               + sampling topology
```

`topology_path_id` 只描述根本不同且数量有限的执行拓扑，例如稀疏方法的 `short` 和
`long` graph family。它不得退化成任意 context bucket。对于 dense 路径，或已经证明
short/long 的 kernel chain、grid、workspace、cache view 和 collective 顺序完全相同的
方法，统一使用一个 path，达到 Level 2。

两级能力都不得再包含：

```text
real context length
1K/2K/4K/32K 等 context bucket
根据当步 context 临时选择的 provider/config
```

首轮允许在 graph 外由 CPU 准备当步 metadata，再拷入地址稳定的 graph input buffer。
因此，本计划解决的是“执行拓扑不依赖真实 context length”，不是一步到位消除 replay
前的所有 CPU 工作。

本轮明确不包含 eviction 的 CUDA Graph 化。现有 cache manager 仍可在 graph 外完成物理
slot 分配、引用计数和释放；本计划只要求当步 forward 使用的物理/逻辑 metadata 能通过稳定
buffer 进入同一个 graph。

## 2. 非目标

本轮不包含：

- graph 内执行 eviction、物理页回收、free-list 更新或容量 admission；
- prefill CUDA Graph；
- 完全 CPU-free 的 decode metadata 构造；
- 强制把拓扑根本不同的 short/long 稀疏路径合并为一张 graph；
- 为了“看起来支持”而只固定少量模型 shape、context length 或设备名；
- 在 replay 过程中捕获异常后静默切换 provider 或 eager；
- 没有 matched benchmark 证据的调优结论；
- 强迫所有现有 provider 支持 batch-only。不能满足契约的 provider 应在绑定阶段明确拒绝。

## 3. 核心约束

### 3.1 动态数据与静态计划必须分离

真实 context length 是每步变化的 GPU 数据：

```text
context_lens: Tensor[batch_capacity], int32, CUDA, stable address
```

capture 时的容量和 launch 计划是静态信息：

```text
context_capacity: int
GraphStableLaunchPlan
```

二者不能继续混用同一个 `max_context_len` 概念。`context_capacity` 只表示已分配的
table/workspace 能覆盖的上界，不得在 replay 时被当作真实长度参与 host routing；
`context_lens` 只描述当步各 row 的真实有效范围。

### 3.2 配置表不是自动的 graph-safe 方案

把热路径中的阈值分支搬到配置表，只有在以下条件下才真正有用：

1. 配置在 capture 前根据静态维度解析一次；
2. 配置不依赖当步真实 context length；
3. replay 期间 kernel 数量、调用顺序、grid、workspace 地址和大小保持不变；
4. 动态 context 只影响 kernel 内的数据访问范围、mask、有效 split 数或工作分配。

如果表项会改变 `constexpr`、`BLOCK_N`、`num_warps`、`num_stages`、grid、shared
memory、workspace shape、kernel variant 或 kernel chain，那么即使“在 kernel 内读取表”，
它仍不能直接解决 batch-only；这些属性在 launch/capture 时已经固定。

因此配置分为两层：

```text
Host static performance profile
  key: device/hardware + model architecture + operator shape + dtype/layout
       + TP topology + batch bucket + configured capacity + topology path
       + graph mode
  value: GraphStableLaunchPlan
  real context length: forbidden

GPU dynamic schedule
  input: context_lens + fixed-address schedule buffers
  output: effective split count/ranges/masks
  invariant: fixed maximum grid and fixed workspace
```

静态 profile 可以复用历史调优结果；GPU schedule 负责在这个静态 envelope 内适应真实
context length。

配置表只在 provider 绑定和 capture 前查询一次：

```text
static spec + DeviceCaps + batch bucket + topology path
  -> lookup GraphStableLaunchPlan
  -> JIT/warmup
  -> allocate fixed workspace
  -> capture
  -> seal
```

每次 replay 不再查询 tile/warp 配置表，只更新固定地址的 `context_lens`、有效 split、
split ranges 和 mask。出现新的真实 context length 不能触发重新选择 tile/warp 或 recapture。

### 3.3 topology path 是有限的静态 graph family

当前 `DecodeCudaGraphKey` 已包含 `graph_path_id`，并通过 `short`/`long` 标识稀疏方法的
graph-stable topology family。近期保留并正式化这条路径，但施加以下约束：

- topology 完全相同的方法返回统一 path，同一 batch bucket 只 capture 一张 graph；
- topology 根本不同的方法允许 `short`/`long` 两个 path，每个 batch bucket 各 capture
  一张 graph；
- `graph_path_id` 必须是方法静态 contract 声明的有限枚举，不能携带具体 context bucket；
- 所有支持的 path 必须在启动期完成 capture 并封存，运行期只选择已有 graph；
- short/long path 内部仍不得根据真实 context length 改变 provider、kernel chain、grid、
  workspace 或 launch config；
- 不能通过不断新增 path id 掩盖 provider 仍在按 context bucket 路由。

path resolver 可以在 graph 外根据方法的共享阈值把同质 batch 分类为 short 或 long，但它
只能选择启动期已经存在的 graph family，不能查询 tile/warp 性能配置、切换未绑定 provider
或触发新 capture。这是 path-scoped 与 strict batch-only 的明确边界。

因此，稀疏方法近期的 graph 数量上界是：

```text
batch bucket count × declared topology path count × sampling topology count
```

而不是 batch bucket count × 任意 context bucket count。

## 4. 当前阻塞点

| 路径 | 当前行为 | 对 batch-only 的阻塞 | 计划处理 |
| --- | --- | --- | --- |
| MiniMax M2 H100 GQA launch profile | `max_context_len > 32768` 时从短配置切到长配置 | 改变 block、GQA tile/warps、grid 和 workspace | 用静态 batch-only plan + fixed-grid dynamic split；不满足性能门槛则 batch-only 绑定其他 provider |
| 通用 Triton paged decode | host `max_context_len` 决定 `block_seq`、`num_seq_blocks` 和 workspace | graph 拓扑与 workspace 随 context 变化 | 生产化固定最大 split provider，实际 split 数由 GPU metadata 控制 |
| 现有 context-independent Triton decode | 固定 `max_kv_splits=16`、少量硬编码 tile，标记为 experimental | 只证明可行性，未形成完整支持域、调优和验证闭环 | 改造成正式 MHA/GQA provider，而不是继续追加特例 |
| FlashInfer paged decode | host `max_context_len` 进入 plan key，并按真实长度重建 indices/indptr | replay 前可能重新 plan，真实长度参与 host 路由 | 验证固定 workspace/plan API；无法证明稳定则明确不支持 batch-only |
| GLM MLA Triton schedule | `max_context_len <= 1024` 选择不同 launch config | context 阈值改变静态 launch plan | batch-only profile 只按 batch/TP/shape 选 envelope，device schedule 决定有效工作 |
| context-independent MLA provider | 用 `8193` 绕过 context 分支 | 魔数只覆盖实验路径，不代表生产支持 | 用正式 profile/capacity contract 替换魔数并覆盖完整支持域 |
| TileLang MLA runtime | context capacity 参与 `num_split`、score mode 和 kernel key | 可能改变编译 variant 和执行链 | 提供固定 plan + device schedule；否则 resolver 阶段拒绝 batch-only |
| graph runner 输入 | runner 直接持有若干 tensor，cache/sparse/provider 各自补充状态 | 生命周期和 ownership 不统一，扩展稀疏算法易继续加位置参数和特例 | 引入统一 registry、typed graph state 和 participant lifecycle |
| 稀疏 short/long 路径 | 短文本类似 vanilla，长文本进入 selection/sparse attention，部分方法拓扑根本不同 | 强制单图需要侵入算法、固定 superset chain，并扩大验证范围 | 默认按 topology path 分别启动期 capture；仅对已证明拓扑相同的方法合并 |
| mixed short/long batch | graph path 是 batch-level contract | 同一 batch 无法同时选择两套根本不同的 graph | 近期由 scheduler 保证 batch 内 path 同质，runner 在 cache mutation 前 fail-fast |

这张表是阶段 0 的初始清单，不是最终支持矩阵。实现前还需逐个 provider 审计是否存在
`.item()`、`tolist()`、按真实长度分配、未预热 JIT、动态输出驱动的 Python 分支，以及
context 相关的第三方 plan/cache key。

### 4.1 阶段 1 provider/context 审计矩阵

下表是当前分支的静态审计结论。“可绑定”只表示 provider 在 capture 前能满足执行
contract；只有标记为“阶段 2 生产验证”的 Qwen3 dense MHA/GQA 组合具有当前阶段的
真实模型和 matched performance 证据。

| provider/path | 真实 context 对 host 拓扑的影响 | batch-only 决策 | 当前证据/后续 owner |
| --- | --- | --- | --- |
| SGL FA3 paged MHA decode | launch topology 依赖 context | `supports()` 在 batch-only 绑定时拒绝 | 保留 eager/bucketed |
| FlashInfer paged MHA decode | plan key、indices/indptr 和 workspace 规划依赖 context | `supports()` 在 batch-only 绑定时拒绝 | 保留 eager/bucketed；固定 plan API 需单独验证 |
| 通用 Triton paged MHA/GQA decode | host 依据 context 选 block/split 并取 workspace view | `supports()` 在 batch-only 绑定时拒绝 | 保留 eager/bucketed |
| context-independent Triton MHA/GQA | tile/warp/max split/workspace 在 `prepare()` 前固定；真实长度只在 GPU 内决定有效 split/range | batch-only 默认原子 provider | 阶段 2 Qwen3-8B/H20 生产验证 |
| MiniMax M2 H100 long-GQA profile | `32768` 阈值会改变 block/tile/warp/grid | batch-only 不绑定该 profile，而绑定上一行的静态 envelope | 阶段 3 补 MiniMax 模型、阈值边界和长上下文证据 |
| context-independent MLA Triton | 固定 grid 原型可绑定 | 暂保留显式 batch-only 原型能力 | 阶段 3 移除魔数并完成 GLM/MLA 生产验收 |
| TileLang/SGL MLA score path | score mode、JIT key 和部分 workspace/plan 尚未证明跨 context 稳定 | 不声明阶段 2 生产支持 | 阶段 3 得出明确支持或拒绝结论 |
| Gemma4 context-independent decode | 固定 grid 原型可绑定，full/window range 由 device metadata 决定 | 暂保留显式 batch-only 原型能力 | 阶段 3 补生产 correctness 和 no-regression gate |

静态审计还确认：新 MHA/GQA provider 的 forward 不执行 `.item()`/`.tolist()`、不重新进入
resolver/plan，workspace 由 provider 持有；运行时 `context_lens` 不参与 Python grid、kernel variant 或
workspace shape 选择。普通 bucketed provider 仍保留原来的 context 分支，两条路由静态 spec
在模型构建时分开，forward 不会在两者之间 fallback。

### 4.2 阶段 1 topology path 和 transition ownership 矩阵

| method 范围 | 最大 path 数 | mixed-batch contract | short→long 分类/状态 owner | 当前阶段结论 |
| --- | ---: | --- | --- | --- |
| vanilla dense | 1：`dense` | 无 short/long 分区 | 无 transition | 阶段 2 已验证 strict batch-only |
| `streamingllm`、`snapkv`、`h2o`、`pyramidkv`、`omnikv`、`quest`、`rkv`、`skipkv` | 2：`short`/`long`（只 capture 配置 context 内可达的 path） | scheduler 只从同一 path queue 取 decode batch；runner 在 cache mutation 前复核并 fail-fast | `method_registry` 定义共享阈值，scheduler 负责分类，方法 cache manager/`SparseController` 负责一次性私有状态准备 | 阶段 1 只冻结 contract；逐方法 transition/prefix/TP 证据属于阶段 4 |
| `deltakv` | 设计上最大 2，当前为 0 个公开 batch-only path | 同上 | DeltaKV cache manager 和 `SparseController` | 配置层明确拒绝 batch-only，不用 bucketed 能力伪装支持 |

path resolver 只使用共享算法阈值选择已启动捕获的有限 family；不查询 provider 调优表，
不携带 context bucket，不允许 lazy capture。阶段 1 建立的 participant 边界只编排 cache/runtime
state，没有把稀疏私有 metadata 的语义 ownership 移入 runner。

## 5. 目标架构

### 5.1 `DecodeGraphContract`

为一次启动期绑定建立显式 contract，至少包含：

```text
shape_policy = batch_only
capability_level = path_scoped | strict
topology_path_id = unified | short | long
batch_capacity
context_capacity
dynamic_context_lens = true
activation/cache dtype
head/page/cache layout
sampling topology
padding contract
provider capability/binding evidence
```

`context_capacity` 是启动 contract/path 的内存和合法输入上界，不是当步请求维度。请求超过
该上界时应在 admission 或 graph 外准备阶段明确失败，不能在 replay 中换图或换 provider。

### 5.2 公共 graph input registry

runner 持有 `DecodeGraphInputRegistry`，统一注册跨模型/方法共享的稳定 buffer：

| slot | 建议 shape | storage owner | semantic owner / value producer | replay 前更新内容 |
| --- | --- | --- | --- | --- |
| `input_ids` | `[batch_capacity]` | runner | runtime state | 当步 token；padding row 使用安全 token |
| `positions` | `[batch_capacity]` | runner | runtime state | 当步 position；padding row 使用安全 position |
| `context_lens` | `[batch_capacity]` | runner | cache/runtime state | 每个 row 的真实长度 |
| `request_indices` | `[batch_capacity]` | runner | cache manager | 请求到 cache metadata row 的映射 |
| `write_slot_mapping` | `[batch_capacity]` | runner | cache manager | 当步 KV 写入 slot |
| `valid_batch_size` 或 `active_mask` | scalar / `[batch_capacity]` | runner | scheduler/runtime state | 区分真实 row 与 padding row |

每个 slot 注册时必须声明：

```text
shape / dtype / device
batch axis
padding policy
source/copy policy
stable-address requirement
```

registry 不接管所有 metadata。它只统一公共输入的地址和复制规则，避免 runner API 随稀疏
方法持续膨胀。

“统一 buffer”不要求把这些 tensor 合并成一块无类型内存。首轮优先采用显式
`DecodeGraphInputs` dataclass；只有出现多个具体消费者共同注册可选公共 slot 的需求后，才
增加轻量 registry。禁止把所有方法私有 tensor 放进一个无限增长的 `dict[str, Tensor]`。

### 5.3 participant lifecycle

cache manager、`SparseController` 和 provider 通过统一 lifecycle 参与 graph：

```python
init_graph_state(contract, topology_path) -> participant_state
prepare_out_graph(step, participant_state) -> None
prepare_in_graph(participant_state) -> None
graph_keepalive_tensors(participant_state) -> Iterable[Tensor]
```

语义如下：

- `init_graph_state`：capture 前分配地址稳定的私有 buffer/workspace，并解析静态计划；
- `prepare_out_graph`：每步在 graph 外构造或更新动态 metadata；允许首轮使用 CPU；
- `prepare_in_graph`：capture/replay 路径内执行固定的 device-side metadata preparation；
- `graph_keepalive_tensors`：显式声明 graph 生命周期内不可释放或换地址的对象。

现有六个位置参数式 `prepare_decode_static(...)` 应迁移为 typed graph state，例如：

```text
prepare_decode_graph_step(seqs, graph_state)
```

方法专属扩展使用有类型的 participant state，不使用一个无限增长的无类型 `dict`。

首轮最小类型结构建议为：

```text
DecodeGraphContract
DecodeGraphInputs
DecodeGraphState
CacheDecodeGraphState
SparseDecodeGraphState
ProviderDecodeGraphState
```

同一个 batch bucket 的 short/long path 可以复用 schema，但首轮分别持有公共输入和私有
workspace，避免两张 graph 因地址共享而产生隐式耦合。模型权重和物理 KV storage 继续共享。

### 5.4 ownership 边界

必须保留现有架构 ownership：

- graph runner：公共 decode 输入、batch padding、capture/replay 和 graph identity；
- cache manager：物理 KV storage、slot/page table、每层物理 cache view；
- `SparseController`：逻辑稀疏选择、跨层 observation、score 协调；
- provider：自己的 kernel plan、schedule buffer、workspace 和物理 weight/layout；
- model/attention layer：只表达稳定算子语义，不添加方法或 provider 特例。

统一 buffer 抽象不意味着 runner 接管 provider workspace，也不意味着把物理 cache metadata
移入 `SparseController`。

每个字段需要区分三个角色：

- storage owner：谁分配并保证地址稳定；
- semantic owner：谁定义字段含义和合法状态；
- value producer：谁在 replay 前或 graph 内写入当步值。

例如 `write_slot_mapping` 的稳定存储可由 runner 分配，但语义和值由 cache manager 负责；
runner 不因此接管物理 slot 生命周期。

### 5.5 padding contract

batch-only graph 必须用 capture bucket 填充不足的真实 batch。padding row 不能只依赖“最后
反正会 slice 掉”，因为它仍会经过 cache write、attention、MoE 和 sampling 链。

建议基础 sentinel 为：

```text
active = false
context_len = 1
request_index = reserved safe row
write_slot = reserved safe slot
position = 0
```

最终 sentinel 需要由绑定的 cache manager 和全部 provider 共同验证，不能直接把上述建议当作
全局事实。验收至少覆盖：无除零、无负索引、无非法 page 访问、无跨请求写入、无 NaN，且
padding row 不改变真实 row 输出。

## 6. 工作流一：改造 context-length 阈值和复用调优信息

### 6.1 先分类每个阈值

每个 context-length 分支必须归入以下一类：

| 类型 | 示例 | batch-only 处理 |
| --- | --- | --- |
| 仅影响 kernel 内 mask/range | 有效 token 上界 | 保留为 GPU 动态数据 |
| 影响有效 split，但最大 grid/workspace 可固定 | split-K 数量 | device schedule 写入有效 split；launch 最大 envelope |
| 影响 tile/warps/grid/variant | MiniMax M2 32K 分支 | capture 前固定一个跨 context plan，或使用固定 superset path |
| 影响 kernel chain/provider | sparse short/long 不同实现 | 默认拆成两个启动期预捕获 topology path；未来有证据时再评估固定 superset |
| 仅用于容量校验 | active slot table width | 改名为 `context_capacity`，只在 graph 外验证 |

### 6.2 复用已有调优结果的优先级

历史上按 context bucket 得到的 winner 不直接变成 replay 时的配置表，而按以下顺序复用：

1. 汇总旧 winner 形成候选集；
2. 对每个 batch bucket，在完整 context 范围验证每个候选的正确性；
3. 使用真实 context 分布和边界点选择一个稳健的静态 plan；
4. 将旧配置中的 split 信息转化为 fixed-grid 内的 device-side effective split；
5. 如果一个 topology path 内的单一 plan 性能不可接受，评估固定 predicated superset；
6. 如果 superset 的空 launch、双 workspace 或编译成本仍不可接受，放弃该 provider/path 的
   batch-only 支持。

固定 predicated superset 的含义是 capture 时始终执行相同 short/long kernel chain，由 GPU
`route_id` 让不命中的路径 no-op。它保持拓扑，但并非默认方案；必须单独报告额外 launch、
workspace 和端到端损失。

这些历史 winner 的作用是形成 capture 前的候选集，而不是让新请求按真实 context 命中表项
并重新 capture。对于拓扑不同的稀疏 short/long path，可以分别选择各自的静态 plan；对于
同一 path，真实 context 只能进入 replay metadata。

### 6.3 profile artifact

每个正式 `GraphStableLaunchPlan` 必须能追溯到 artifact，至少记录：

```text
git revision / provider revision
GPU model and software stack
operator/model shape, dtype and layout
batch bucket, topology path and supported context range
candidate configs and representative context distribution
latency percentiles, workspace bytes and launch count
correctness result
chosen plan and rejection reason
```

测试不冻结某个具体阈值或 tuning 常量。retune 时更新权威 profile/artifact，并重新验证
行为边界。

## 7. 工作流二：统一 graph 输入和 metadata 生命周期

### 7.1 第一阶段迁移

1. 定义 `DecodeGraphContract`、公共 slot schema、padding contract 和 participant protocol；
2. 让 `DecodeCudaGraphRunner` 通过 registry 分配公共 buffer；
3. 把现有 `prepare_decode_static(...)` 迁移到 typed graph state；
4. 先迁移 `StandardCacheManager`，确保 eager 和 bucketed graph 行为不变；
5. 加入地址稳定、copy 边界、padding 和 keepalive 测试；
6. 再迁移每层 sparse/cache metadata，避免在公共 API 中加入方法名分支。

公共输入使用显式 typed dataclass，cache manager、`SparseController` 和 provider 使用各自
typed participant state。runner 只编排 lifecycle，不解析算法私有字段。

### 7.2 稀疏算法接入范围

本轮只统一 forward 所需 metadata，例如：

- 每层 `context_lens` 或有效长度 view；
- active slots/page tables；
- sparse selected indices；
- attention score buffer；
- selected count、有效范围和 enable flags。

本轮不把 eviction 决策、slot 生命周期或物理回收移入 graph。方法的 short/long 转换如果会
改变 kernel chain，默认分别建立 `short` 和 `long` topology path；只有已经证明拓扑完全
相同的方法才合并为 strict batch-only graph。

### 7.3 short/long path、切换与 mixed batch

对存在根本路径差异的稀疏方法：

1. method registry 声明有限的 `short`/`long` topology path 和共享阈值语义；
2. 每个 batch bucket 的两个 path 都在启动期 capture，运行期禁止按需增图；
3. scheduler 保证一个 decode batch 内 path 同质，mixed short/long batch 在 cache mutation
   前 fail-fast；
4. 请求跨过阈值后，下一步进入 long queue，并选择已捕获的 long graph；
5. cache manager/`SparseController` 负责一次性的 short→long 状态初始化，不能在 model
   runner 添加方法名分支；
6. prefix-cache restore 后可能第一步直接进入 long path，不能依赖请求曾经 replay short
   graph；
7. 所有 TP rank 必须得到一致的 path 分类、transition 顺序和 collective topology；
8. short/long graph state 首轮分别拥有地址稳定的私有 metadata/workspace，持久 KV storage
   和模型权重继续共享。
9. 启动 graph budget 必须覆盖全部声明的 batch bucket × topology path；预算不足时启动失败
   或缩小公开支持矩阵，不能退化为运行期 lazy capture。

如果后续证明 long topology 可以通过 mask 正确且高效地处理 short row，可以单独立项合并为
strict batch-only；本轮不以侵入所有稀疏算法实现 fixed superset 为交付前提。

### 7.4 metadata 准备优化原则

batch-only graph 只固定 GPU 地址和执行拓扑，并不会自动消除 replay 前的 CPU metadata
准备成本。首轮优化以低侵入、可验证收益为边界，不重写 scheduler，也不引入新的复杂并发
子系统。metadata 按以下四类处理：

1. **静态 plan**：模型、硬件、provider、tile/warp、workspace envelope 和 topology path 在
   capture 前固定，replay 时不再计算；
2. **最小动态事实**：本 step 的 request/slot id、`context_lens`、active batch、block table
   变更和稀疏选择结果，由 CPU 写入公共 graph input；
3. **GPU 可派生 metadata**：positions、slot mapping、有效 split/range、active mask 等，优先
   从最小动态事实在 GPU 上批量生成，避免 CPU 重复遍历和大量小拷贝；
4. **算法私有 metadata**：仍由 cache manager、`SparseController` 或 provider participant
   持有，runner 只负责统一更新和同步生命周期。

优化前先分段测量 request 遍历、host buffer 填充、cache/sparse/provider prepare、H2D 和等待
时间。只优化能够影响 decode step latency 或 CPU 饱和度的阶段，不能仅以减少 Python 代码
作为收益证据。

### 7.5 持久 buffer、向量化写入和传输

公共 graph input 采用持久、地址稳定的 GPU tensor；需要从 CPU 更新的输入配套持久 pinned
host mirror，并提供 NumPy view 或等价的批量写入接口。每个 replay 只更新 active prefix 或
明确的有效区域，不重新创建 tensor、list、临时 page table 或 workspace。

首轮按以下顺序实现：

1. 复用 host/GPU buffer，消除每 step 分配、dtype 转换和对象重建；
2. 用 NumPy/PyTorch 批量赋值替代逐 request、逐 token 的 Python 小循环；
3. 合并同一生命周期、同一目标 stream 上的小 H2D copy，并为每个字段定义明确的 padding
   策略；
4. 在 CPU 完成某个独立输入后尽早发起 `non_blocking` H2D，与后续 CPU prepare 重叠；
5. replay 在当前 stream 上等待 copy 完成，保持 GPU graph input 地址不变。

`non_blocking` 必须由 pinned memory、独立 CPU 工作和 trace 共同证明存在真实 overlap。当前
计划不引入 UVA buffer pool、逐元素 diff-write、复杂环形 staging 或多阶段 event 协议；这些
机制的实现和验证成本较高，只有后续 profile 证明基础方案仍受 metadata 传输显著阻塞时才
另行立项。

### 7.6 graph 内 GPU 派生 metadata

若某字段能从少量公共输入确定，优先在固定拓扑的 graph 内生成，而不是由 CPU 展开后传入：

- `context_lens` 与静态 plan 生成每个 row/head 的有效 split 和 range；
- request/slot、block table 与 position 生成 slot mapping 或 active slot view；
- active batch、topology path 和 padding contract 生成有效 mask/count；
- 稀疏 selected indices/count 生成 provider 所需的固定容量 view。

这些 preparation kernel 必须使用固定 grid 或 capture 前固定的 launch plan，写入预分配输出，
不得执行 `.item()`、`.cpu()`、动态分配或 host 分支。只有同时减少 CPU 工作或 H2D 数据量，且
kernel 开销低于被替代工作时才迁移；简单标量和已经廉价批量生成的字段仍可由 CPU 直接写入。

### 7.7 异步边界和暂不扩大的范围

第一阶段采用单一明确依赖链：CPU 填充 pinned mirror → 提前异步 H2D → replay stream 等待
copy 完成 → graph replay。异步只用于让已经就绪字段的 H2D 与同一 step 剩余 CPU prepare
重叠，不让下一 step 的 metadata prepare 与当前 graph replay 跨 step 并发。必须保证：

- CPU staging buffer 在对应 copy 完成前不被复写；
- 公共 GPU input 在上一次 replay 最后一次读取完成前不被更新；
- cache mutation、稀疏选择和 TP collective 的依赖顺序保持不变；
- 同步版与异步版连续 replay 的结果完全一致。

本轮不建设通用的多版本 metadata ring、participant read-end 协议或新的后台线程池。也不把
metadata packer 改写为 C++/Rust；现阶段更可能的收益来自减少工作量、批量写入、减少传输和
step 内的有限 overlap，高性能语言重写不属于当前计划。若后续 profile 证明跨 step overlap
存在足够收益，再将双缓冲、读写事件和 participant 生命周期作为独立设计评审，不在本计划中
预埋复杂抽象。

## 8. 工作流三：生产化阻塞 batch-only 的算子

### 8.1 通用 MHA/GQA decode 与 Qwen3 首轮验证

以 Qwen3 vanilla dense GQA + `StandardCacheManager` 作为首个端到端参考，先验证最小统一
buffer、静态 launch plan、fixed-grid dynamic split provider 和 batch-only replay 能形成完整
生产闭环。实现目标仍是有明确支持域的通用 MHA/GQA provider，不在 kernel 或 provider 中
加入 Qwen3 模型名分支：

1. 生产化现有 fixed-grid split-K 原型；
2. 静态 plan 决定最大 split、tile、warps/stages 和 workspace envelope；
3. GPU schedule 根据 `context_lens` 生成每个 row/head 的有效 split/range；
4. kernel grid、workspace 地址和调用链对 context 保持不变；
5. 覆盖 MHA/GQA、FP16/BF16、声明的 head dim、page/cache layout 和 score 输出契约；
6. workspace 在 provider `prepare()`/graph state 初始化时一次分配；
7. Qwen3 覆盖声明的 batch bucket、代表 context、padding、ragged batch 和生产所需 TP 配置；
8. 使用 bench probe 对比同一 Qwen3 配置的 batch-only 与 bucketed/eager 路径；
9. eager/bucketed 模式保留原有 profile，避免扩大行为变化。

现有 `max_kv_splits=16`、按少量 head dim 硬编码 tile 和
`repo_triton_experimental` metadata 只能作为迁移起点，不能作为完成标准。

### 8.2 MiniMax M2、Gemma4 与 MLA

- MiniMax M2：复用已经由 Qwen3 验证的通用 MHA/GQA provider，batch-only 模式不再调用
  `>32768` launch 分支，并覆盖阈值前后和长上下文；MiniMax 不再承担首轮架构验证。
- Gemma4：full/sliding-window decode 共用稳定 launch envelope，device schedule 决定有效
  split 和 window range；不得在 host 根据当步 window/context 换链。
- GLM MLA Triton：保留已有 device scheduling 思路，但移除 `<=1024` 的 replay-time
  launch 切换和 `8193` 魔数；正式 plan 只依赖 batch、TP、head shape 和静态容量契约。
- TileLang MLA：审计 `num_split`、score mode、JIT key 和 workspace。如果无法固定 variant
  和调用链，则在 provider resolver 阶段拒绝 batch-only。
- SGL FA3/FlashInfer：只在固定依赖版本的 API 能证明 plan、workspace、indices/indptr 和
  launch topology 稳定时声明支持；否则保留 eager/bucketed 能力，不做运行时 fallback。

### 8.3 正式 provider 的最低要求

一个 provider 只有同时满足以下条件才能在声明的 topology path 内支持 batch-only：

- 支持域按 shape/dtype/layout/capability 表达，不按几个实验 shape 列白名单；
- `supports()` 能在 capture 前判定兼容性；
- provider 拥有并预分配全部 workspace 和 schedule buffer；
- forward hot path 无真实 context 驱动的 host dispatch、规划或分配；
- 超出支持域时由 resolver/preparation 明确失败；
- 数值结果通过独立 CUDA oracle；
- 同一 graph 跨 context 边界连续 replay 正确；
- 有真实模型集成和 matched performance evidence；
- binding report 能说明选择了哪个静态 plan，以及该 plan 的支持范围和证据。

## 9. 执行顺序与阶段门槛

全部工作在当前分支连续推进，最终作为一个 PR 交付。下面的阶段是实现和验证顺序，不是
branch 或 PR 边界；允许用独立 commit 保留可回退检查点，但不为了阶段本身拆分接口、重复
兼容层或建立多套临时实现。每个阶段通过自己的 correctness 和 bench probe gate 后再进入
下一阶段。

### 阶段 1：最小统一 buffer 和 graph contract

- 冻结 path-scoped/strict batch-only identity、动态/静态 context、padding 和 topology path
  contract；
- 完成 provider/context blocker matrix，并记录稀疏方法的 path 数量、mixed-batch 约束和
  short→long transition owner；
- 落地最小 `DecodeGraphContract`、`DecodeGraphInputs` 和 `DecodeGraphState`；
- 迁移 `DecodeCudaGraphRunner`、runtime state、`StandardCacheManager` 和公共 decode 输入；
- 建立 participant lifecycle 的最小边界，但暂不迁移全部稀疏方法或构建动态通用 registry；
- 复用持久 buffer，并先消除明显的每 step 分配、对象重建和逐项 Python 填充；
- 保持 eager、static eager 和 bucketed graph 的输出、路由和性能行为不变。

阶段 1 的目标是提供可被真实算子消费的最小 typed contract，不要求先完成所有 metadata GPU
派生、异步 overlap 或稀疏 participant。

### 阶段 2：Qwen3 生产纵向闭环

- 以 Qwen3 vanilla dense GQA + `StandardCacheManager` 验证统一 buffer 的第一条生产路径；
- 生产化 fixed-grid dynamic split MHA/GQA provider，形成静态 launch plan/profile artifact；
- 验证静态 tile/warp/max-split/workspace 与动态 `context_lens`、有效 split/range 的分层；
- 覆盖声明的 batch bucket、代表 context、padding、ragged batch、score/no-score 和生产所需
  TP 配置；
- 优先使用 `benchmark/efficiency/bench_probe.py` 跑 Qwen3 batch-only 与 bucketed/eager 的
  matched fixed/churn 端到端对照；
- 根据 bench probe 和 trace 结果，再决定是否启用 pinned mirror、批量 copy、graph 内 GPU
  metadata 派生和 step 内有限异步；
- microbenchmark 用于 kernel config 选择和性能归因，不能替代 Qwen3 端到端 gate。

阶段 2 通过后，统一 buffer、provider ownership 和 batch-only 算子接口视为基本稳定；后续模型
只能扩展 typed contract，不能重新引入模型名分支或真实 context 驱动的 host dispatch。

### 阶段 3：扩展阻塞算子和模型覆盖

- MiniMax M2 复用 Qwen3 已验证的 MHA/GQA provider，移除 batch-only 下的 32K launch
  切换并覆盖长上下文；
- 改造 Gemma4 full/sliding-window schedule、GLM MLA context 配置和魔数；
- 对 TileLang、SGL FA3、FlashInfer 分别形成明确的 batch-only 支持或拒绝结论；
- 每个模型先跑 correctness/graph identity，再用 bench probe 做 matched batch-only 与
  bucketed/eager 对照；
- 不通过运行时 fallback 或新的实验 shape 白名单掩盖缺失能力。

### 阶段 4：稀疏 path 接入和生产硬化

- 完成目标稀疏方法的 typed participant state 和 forward metadata 生命周期；
- 对拓扑相同的方法使用统一 path，对拓扑不同的方法启动期预捕获有限 short/long path；
- 完成 short→long、prefix restore 直接进入 long、TP path 一致性和 mixed-batch 调度验证；
- 用 bench probe 的 matched fixed/churn workload 验证同一稀疏方法的 batch-only 与
  bucketed/eager 行为和性能；
- 移除实验命名、魔数和只覆盖少量 shape 的临时路径；
- 只有 correctness、graph identity、显存和端到端性能证据通过的组合才默认启用。

本阶段仍不包含 eviction graph 化。完成后统一整理当前分支的 commit、验证 artifact、binding
report 和用户文档，形成一个可审查的最终 PR。

## 10. 验证矩阵与硬验收标准

### 10.1 Graph identity 和生命周期

对每个声明支持的模型/方法/provider：

- strict batch-only 每个 batch bucket 只 capture 一个 forward graph；
- path-scoped batch-only 每个 batch bucket × 声明 topology path 只 capture 一个 graph；
- graph 总数不超过 batch buckets × topology paths × sampling topologies；
- 在同一 path 内顺序 replay 代表长度、历史算子阈值前后、最大长度和 mixed-length batch；
- `capture_count` 保持不变，启动计划封存后 `recapture_count == 0`；
- 所有注册 buffer 和 provider workspace 的 `data_ptr()` 保持不变；
- graph key/report 中不存在真实 context bucket；只有拓扑确实不同的方法允许有限
  short/long path；
- replay 前 metadata 更新不会触发 JIT、plan、分配或 provider 切换。

阶段 2 的 Qwen3 首轮验证至少覆盖全部声明 batch bucket、多个代表 context、ragged batch 和
padding，并证明它们在统一 path 内不增图。Qwen3 通过后再扩大模型矩阵。

MiniMax M2 至少覆盖 `32767/32768/32769`；GLM MLA 至少覆盖
`1023/1024/1025`。这些点用于验证跨历史阈值同图 replay，不用于冻结原阈值。

稀疏方法还必须覆盖算法阈值 `threshold-1/threshold/threshold+1`：前两者按方法定义进入
short path，后一者切换到已捕获 long graph；切换不得增加 graph、丢失 cache/score 状态或
改变其他请求状态。

### 10.2 数值正确性

- 每个 GPU kernel 与独立 PyTorch/CUDA reference 对照；
- 覆盖单 row、ragged batch、同一 topology path 内的 mixed context、最大容量、非整 tile
  长度和 padding row；
- MHA/GQA/MLA、FP16/BF16、声明的 head dim/page layout 分别验证；
- score-producing provider 同时验证 output、LSE/score 和调用方需要的副作用；
- graph replay 与同 topology eager 输出满足同一容差；
- 连续 replay 不得出现跨请求污染或 stale metadata。

### 10.3 Padding 和内存安全

- padding row 不产生非法 slot/page 访问；
- 不写入真实请求的 KV 或 score；
- 不产生 NaN/Inf 或除零；
- active mask/sentinel 改变时真实 row 输出不变；
- 超过 `context_capacity` 在 graph 外 fail-fast，且失败不修改 cache/graph 状态。

### 10.4 路由和兼容性

- resolver 测试覆盖支持、缺依赖、错误设备能力、错误 layout 和超容量；
- 不支持 batch-only 的 provider 不会被选中；
- 绑定完成后 forward 不再做 provider capability 判断；
- scheduler 不生成 mixed short/long decode batch，违规输入在 cache mutation 前失败；
- short→long transition、prefix restore 直接进入 long path 和 TP rank path 一致性通过；
- eager 和 bucketed graph 的已有 provider/profile 行为保持不变；
- 模型集成覆盖生产所需 TP 配置和 prefix-cache replay。

### 10.5 metadata 准备与异步正确性

- 同步更新与 step 内有限异步更新在连续 replay、ragged batch 和 padding row 下结果一致；
- host staging 在 H2D 完成前不复用，GPU input 在上一次 graph 读完前不覆盖；
- replay 热路径无 tensor/workspace 动态分配、device-wide synchronize 或 host 回读；
- trace 能区分 CPU prepare、H2D、等待和 replay，并证明 `non_blocking` 是否产生真实 overlap；
- 分别报告 metadata prepare p50/p95、H2D 次数/字节、同步等待时间、prepare-to-replay gap
  和被隐藏的 overlap 时间；
- GPU preparation kernel 的额外 launch/延迟必须小于其节省的 CPU 与传输成本。

### 10.6 性能证据

端到端性能 gate 优先使用
[`benchmark/efficiency/bench_probe.py`](../../benchmark/efficiency/bench_probe.py)；正式运行遵循
[`docs/en/benchmarking/efficiency.md`](../../docs/en/benchmarking/efficiency.md)。标准配置优先用
`scripts/benchmarks/run_efficiency_probe.sh` 完成 GPU 空闲检查和 artifact 组织；batch-only 与
bucketed/eager 的精确 A/B 可以直接调用 bench probe，并通过独立 `--hyper-params` 和输出目录
固定 graph 配置，但必须先手动确认 GPU 空闲。阶段 2 先跑 Qwen3，后续模型和稀疏方法沿用
相同协议。

batch-only、bucketed 和需要时的 eager baseline 必须保持 model revision、prompt/output
length、batch/concurrency、fixed/churn scenario、seed、TP、scheduler budget、warmup 和迭代数
一致，并使用独立输出目录。先运行小规模 smoke，再运行 matched fixed + churn；主要 decode
结论使用足够长的 output，不能用只验证启动和 capture 的短输出代替。

microbenchmark 只用于独立 kernel correctness、候选 config 选择和 bench probe 异常归因；
不能以 isolated kernel 加速替代端到端通过。只有 bench probe 定位出明确可疑 case 后才使用
Nsight 诊断。

保存 bench probe、microbenchmark 和必要 trace 的原始数据与运行配置，至少报告：

```text
kernel latency p50/p95
decode step latency / throughput
kernel launch count
fixed workspace and graph-pool bytes
graph count and bytes by topology path
GPU schedule overhead
metadata prepare p50/p95、H2D bytes/copies 和同步等待
CPU/H2D 与 GPU forward 的有效 overlap
predicated no-op overhead（如使用）
相对现有 bucketed/eager baseline 的结果
```

不能仅以 graph 数减少作为性能通过条件。固定 envelope 若在真实 workload 上造成不可接受的
额外计算或显存，应收窄支持域、换 provider，或明确放弃该组合的 batch-only 支持。

## 11. 完成定义

本计划完成时，应同时满足：

1. graph runner 有统一、可扩展且 ownership 清晰的输入/participant 抽象；
2. `context_lens` 作为动态 GPU 数据贯穿 cache、sparse metadata 和 provider；
3. 支持组合在一个声明 topology path 内的 host 执行拓扑不再依赖真实 context length；
4. Qwen3 首轮纵向闭环通过，且通用 MHA/GQA、MiniMax M2 和目标 MLA/Gemma4 路径完成
   生产级改造；
5. 稀疏方法能通过 typed metadata 接入 path-scoped batch-only graph；拓扑相同的方法可合并为
   strict batch-only，eviction 仍留在 graph 外；
6. 不满足契约的 provider 在启动期得到清晰的 unsupported 结论；
7. correctness、graph identity、内存安全和性能 artifact 均可复现；
8. metadata 更新复用持久 buffer，已消除明显的逐项 Python 热点和不必要传输；异步优化有
   trace 与端到端收益证据，且不依赖新的复杂并发框架或 C++/Rust 重写；
9. eager 与 bucketed 模式没有请求范围外的行为变化。

## 12. 参考实现吸收原则

metadata 优化参考 SGLang revision
`e586a6f2c5f2d1e0626bbe0cb1580d56c12398a2`（Apache-2.0）和 vLLM revision
`e239947777e18071c8053195ce599b6511717f67`（Apache-2.0），重点吸收：

- graph key 以 batch size 为主要动态 shape；
- 公共 forward-batch graph slot registry；
- provider 私有 buffer/workspace 仍由 provider 持有；
- graph 外/graph 内 metadata preparation hooks；
- padding sentinel；
- Triton 固定最大 split workspace + device-side 有效 split；
- pinned CPU mirror、持久 GPU buffer 和非阻塞 copy；
- CPU 侧 NumPy/PyTorch 批量填充，并尽早启动可独立的 H2D；
- slot mapping、positions 等派生 metadata 在 GPU 上批量生成。

不直接复制完整 `ForwardBatch` 或上游框架结构。本仓库继续遵循 cache manager、
`SparseController`、operator/provider 和 model layer 的现有 ownership 边界，只引入完成当前
batch-only contract 所需的最小公共机制。上游更复杂的 UVA/diff-write、staging pool 和并发
协议不属于当前交付范围；是否吸收必须由本仓库 profile 和端到端收益单独证明。
