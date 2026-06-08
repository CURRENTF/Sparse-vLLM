# vLLM / SGLang Prefix Cache 源码调研与 Sparse-vLLM 设计建议

本文只基于本仓库 `/reference` 目录下的 vLLM、SGLang 源码，以及 Sparse-vLLM 当前源码进行分析。目标场景是 agent 类长输入、多请求共享固定系统提示词/工具说明/模板前缀，并且需要兼容 OmniKV、SnapKV、QuEST、DeltaKV 等稀疏 KV 方法。

## 结论

Sparse-vLLM 最适合采用混合式 prefix cache：

1. 匹配层使用 SGLang 风格的 radix/prefix tree。原因是 agent 前缀常常不是固定 block 边界，radix tree 能做任意 page-aligned 最长前缀匹配，并能在调度阶段做 longest-prefix-first 和 in-batch prefix caching。
2. 物理存储层使用 vLLM 风格的 page/block pool、ref count、LRU eviction。原因是 Sparse-vLLM 的 attention kernel 已经通过 `Req -> token slots` 表读 KV，block/page 粒度元数据比 per-token radix value 更容易做到高性能和低 Python 开销。
3. 稀疏方法通过 cache manager hook 接入，不改 `layers/attention.py` 主路径。Sparse-vLLM 当前 attention 已经只依赖 `cache_manager.get_layer_store_view()`、`get_layer_buffer_req_to_token_slots()`、`build_decode_view()` 和 `SparseController.get_read_view()`，这是最适合插入 prefix cache 的边界。
4. Prefix cache 的 payload 必须是 method-aware：vanilla/OmniKV 可以先缓存全量 KV slots；DeltaKV 应缓存 full-layer slots、sparse-layer raw center slots、latent slots、compressed length 和 center metadata；SnapKV/StreamingLLM 应缓存最终保留后的 slots 和逻辑位置映射。

不建议直接移植：

- 纯 vLLM 块哈希：实现简单、高性能，但只能缓存完整 block 的连续前缀，对 agent 模板变化、chunked prefill、中间共享前缀、in-batch 前缀协同不够灵活。
- 纯 SGLang radix cache：命中能力最强，但 value 是 per-token KV indices，Python tree/torch tensor 拼接/节点拆分复杂度高；直接接到 Sparse-vLLM 的 per-layer sparse payload 会变得很重。

推荐路线是：`RadixPrefixDirectory` 负责匹配、命名空间、引用计数和调度提示；`PrefixBlockPool` 负责 GPU page/block 生命周期；各 `CacheManager` 实现 `capture_prefix_payload()` 和 `attach_prefix_payload()`。

## vLLM Prefix Cache 实现

### 核心数据结构

vLLM v1 的 prefix cache 建立在 KV block 之上。`KVCacheBlock` 保存 `block_id`、`ref_cnt`、`block_hash`，并通过 `prev_free_block` / `next_free_block` 进入 free-list。源码见 `reference/vllm/vllm/v1/core/kv_cache_utils.py:116`。

`FreeKVCacheBlockQueue` 是手写双向链表，注释说明其 free queue 顺序就是 eviction order：LRU block 在队首；同一请求释放时反转 block 顺序，使 tail block 先被淘汰。源码见 `reference/vllm/vllm/v1/core/kv_cache_utils.py:165`。

`BlockPool` 保存所有 `KVCacheBlock`，并维护 `cached_block_hash_to_block`，即 block hash 到 block 的索引。源码说明 cached block 可能正在被 running request 使用，也可能在 free queue 中等待被淘汰，见 `reference/vllm/vllm/v1/core/block_pool.py:130`。

### Block Hash 机制

vLLM 对每个完整 block 计算 hash。`hash_block_tokens()` 把 parent block hash、当前 block tokens、extra keys 一起 hash，形成链式 hash，所以相同 block tokens 出现在不同前缀后面不会误命中。源码见 `reference/vllm/vllm/v1/core/kv_cache_utils.py:542`。

extra keys 包括多模态输入、LoRA 名称、cache salt、prompt embeddings hash。cache salt 只加入第一个 block，见 `reference/vllm/vllm/v1/core/kv_cache_utils.py:504`。这对隔离不同租户、不同系统提示词版本、不同 adapter 非常关键。

`get_request_block_hasher()` 只为 full block 生成 hash，不足一个 block 的尾部不会进入 prefix cache。源码见 `reference/vllm/vllm/v1/core/kv_cache_utils.py:638`。

多 KV group 时，vLLM 允许 `hash_block_size` 小于某些 group 的物理 block size，并通过 `BlockHashListWithBlockSize` 把细粒度 hash 懒合并成更大 block 的 hash。源码见 `reference/vllm/vllm/v1/core/kv_cache_utils.py:572` 和 `reference/vllm/vllm/v1/core/kv_cache_utils.py:2083`。

### 命中、复用和提交

`KVCacheManager.get_computed_blocks()` 在 prefix caching 开启时调用 coordinator 查找最长缓存命中；如果 prompt 全部命中，仍强制最多命中 `prompt_length - 1`，因为最后一个 token 必须重新计算 logits。源码见 `reference/vllm/vllm/v1/core/kv_cache_manager.py:196`。

`allocate_slots()` 把 token 区域分为 already computed、new prefix hits、external connector hits、new tokens 和 lookahead tokens。它先处理命中 block，再为需要计算的 token 分配新 block，最后把已经完成的 full blocks 提交到 prefix cache。源码见 `reference/vllm/vllm/v1/core/kv_cache_manager.py:238`。

`BlockPool.touch()` 在 cache hit 被新请求复用时增加 `ref_cnt`；如果 block 原本在 free queue 中，则从 free queue 移除，避免被淘汰。源码见 `reference/vllm/vllm/v1/core/block_pool.py:402`。

`BlockPool.free_blocks()` 降低 ref count，降到 0 的 block 回到 free queue。源码见 `reference/vllm/vllm/v1/core/block_pool.py:419`。

### 淘汰和去重

`BlockHashToBlockMap` 支持一个 hash 对应多个 block，但源码注释明确说当前不会 deduplicate 已存在的相同 block，因为要保持 block table append-only，避免改变已分配 block ids。源码见 `reference/vllm/vllm/v1/core/block_pool.py:34`。

当需要新 block 时，`get_new_blocks()` 从 free queue 弹出 block；如果该 block 还带 hash，就从 prefix cache map 里移除并 reset hash。源码见 `reference/vllm/vllm/v1/core/block_pool.py:333` 和 `reference/vllm/vllm/v1/core/block_pool.py:365`。

`reset_prefix_cache()` 只有在除 null block 外所有 block 都 free 时才成功，否则返回 false。这保证不会把 running request 正在引用的 cache 结构清掉。源码见 `reference/vllm/vllm/v1/core/block_pool.py:454`。

### Hybrid KV Cache

vLLM 的 `HybridKVCacheCoordinator` 对不同 KV cache group 做统一命中长度收敛。它按 spec 分组，full attention 优先查，之后每个 attention type 要么接受当前候选长度，要么缩短候选长度，直到固定点收敛。源码见 `reference/vllm/vllm/v1/core/kv_cache_coordinator.py:422` 和 `reference/vllm/vllm/v1/core/kv_cache_coordinator.py:532`。

这点对 Sparse-vLLM 很重要：OmniKV/DeltaKV/SnapKV 也会让不同层具有不同的最终 KV 形态，prefix cache 不能假设所有层都共享同一种 payload。

### vLLM 方案优缺点

优点：

- Block/page 粒度元数据少，hash lookup O(number_of_blocks)，易保持高吞吐。
- ref count + free-list LRU 很适合 GPU KV pool。
- 多 KV group 设计成熟，可以借鉴到 Sparse-vLLM 的 full layers / sparse layers 分组。
- full-hit 仍重算最后 token，避免 logits 缺失。

缺点：

- 只缓存 full block；prompt 尾部和非 block-aligned 共享前缀不能命中。
- 没有 radix tree 的 in-batch prefix coordination，第一个请求和后续相同前缀请求可能同时重复 prefill。
- 默认不做相同 block 去重，agent 大量同前缀并发时可能产生重复驻留，直到释放后才靠 hash 命中复用。
- 对稀疏 payload 没有直接表达能力；需要扩展 group id / block metadata 才能表达 DeltaKV latent、OmniKV 动态选择等状态。

## SGLang Prefix Cache 实现

### RadixKey 与 TreeNode

SGLang 的 `RadixKey` 保存 token ids、`extra_key` 和 Eagle bigram 模式。`extra_key` 被用于 LoRA、cache salt 等隔离命名空间；源码注释明确说不同 extra_key 的相同 token prefix 不会共享节点。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:56` 和 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:334`。

`RadixKey.match()` 支持按 `page_size` 对齐匹配。`page_aligned()` 会把长度向下截断到 page size 的整数倍。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:112` 和 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:139`。

`TreeNode` 保存 children、parent、key segment、value、lock_ref、last_access_time、host_value、hash_value 和 priority。`value` 是该节点对应 prefix segment 的 KV indices。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:198`。

### Match 与 Insert

`match_prefix()` 先转换 Eagle bigram，再按 page 对齐，然后调用 `_match_prefix_helper()` 找最长 cached prefix；返回的 `device_indices` 是拼接后的 KV indices。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:334`。

`_match_prefix_helper()` 沿 radix tree 走 child，如果命中停在某节点 key 中间，就调用 `_split_node()` 把节点切开，使后续匹配边界更精确。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:619`。

`insert()` 同样按 page 对齐，并在 `_insert_helper()` 中复用已有前缀、拆分节点、创建新节点。新节点会增加 `evictable_size_` 并进入 leaf 状态更新。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:394` 和 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:675`。

### 请求完成与未完成缓存

`cache_finished_req()` 在请求完成时把 committed KV 插入 radix cache。若插入后发现部分前缀已经存在，它会 free 掉重复的 KV indices；未对齐的 tail 也会释放。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:414`。

`cache_unfinished_req()` 支持 chunked prefill：中间 chunk 结束后也可以把当前 prefix 插入树，然后重新 match 获得可能更新后的 shared prefix indices，并写回 request 的 `prefix_indices`。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:461`。

这对 agent 很关键：固定系统提示词很长时，不必等整个请求完成才能让后续请求受益。

### Lock、Eviction 与调度

`inc_lock_ref()` 沿节点到 root 增加 lock_ref；当节点从 unlocked 变为 locked 时，从 evictable 转为 protected。`dec_lock_ref()` 反向释放。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:563`。

`evict()` 只从 evictable leaves 中选节点，按 eviction policy 的 priority 建 heap，free 节点 value，并在父节点变成可淘汰叶子时继续推入 heap。源码见 `reference/sglang/python/sglang/srt/mem_cache/radix_cache.py:534`。

调度策略中，`match_prefix_for_req()` 把 match result 写入 req：`prefix_indices`、`last_node`、`host_hit_length`、`num_matched_prefix_tokens` 等。源码见 `reference/sglang/python/sglang/srt/managers/schedule_policy.py:85`。

`SchedulePolicy` 支持 LPM 和 DFS_WEIGHT 两种 cache-aware policy。LPM 按最长匹配前缀优先；还有一个 waiting queue radix tree 做 in-batch prefix caching，如果多个 waiting 请求共享较长前缀，会临时 deprioritize 后续请求，让先运行的请求把前缀写入 cache。源码见 `reference/sglang/python/sglang/srt/managers/schedule_policy.py:145` 和 `reference/sglang/python/sglang/srt/managers/schedule_policy.py:243`。

调度 admission 不是只看 free tokens，而是看 allocator available size + tree cache evictable size。源码见 `reference/sglang/python/sglang/srt/managers/schedule_policy.py:514`。

### Memory Pool 与 Attention 接入

`alloc_for_extend()` 先基于 `prefix_indices` 构造每个请求已有 prefix，然后只为 extend 部分分配 KV slots，并把 prefix slots + new slots 写入 `req_to_token_pool`。源码见 `reference/sglang/python/sglang/srt/mem_cache/common.py:285`。

`alloc_token_slots()` 和 paged allocation 在分配前会调用 `evict_from_tree_cache()`，即在 OOM 前主动从 prefix cache 中淘汰可释放节点。源码见 `reference/sglang/python/sglang/srt/mem_cache/common.py:155` 和 `reference/sglang/python/sglang/srt/mem_cache/common.py:183`。

`RadixAttention.forward()` 本身不直接关心 radix tree，它通过 forward batch 和 attention backend 使用已经构造好的 cache locations。这种分层与 Sparse-vLLM 当前 attention/cache manager 分层很接近。源码见 `reference/sglang/python/sglang/srt/layers/radix_attention.py:54`。

### HiRadixCache / Hierarchical Cache

SGLang 的 `HiRadixCache` 在 RadixCache 之外增加 host KV pool 和 storage backend。初始化时按 MHA/MLA/DSA KV pool 类型创建 host pool 或 controller；不支持的 pool 会直接报错。源码见 `reference/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:72`。

HiRadixCache 的 eviction 可以先 write-back 到 host，再从 GPU evict；已经 backup 的节点只从 device 移除但保留 host_value。源码见 `reference/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:959`。

这适合超长 agent session，但复杂度明显高于 Sparse-vLLM 近期需要的 GPU prefix cache。

### SGLang 方案优缺点

优点：

- 任意 page-aligned 前缀匹配，适合 agent 系统 prompt、工具 schema、few-shot 模板。
- 支持 chunked prefill 中间插入，使后续请求更早复用。
- in-batch prefix caching 能避免相同前缀并发重复计算。
- `extra_key` 命名空间隔离清晰。
- lock_ref / evictable_size 与调度紧密耦合，不容易误淘汰 running prefix。
- HiCache 提供 GPU/CPU/storage 层级扩展路径。

缺点：

- Python radix tree、节点拆分、torch.cat(prefix indices) 和 per-token value 对长上下文有开销。
- 直接缓存全量 per-token KV indices，对低显存目标不够友好；低显存需要再叠加 HiCache 或稀疏/压缩 payload。
- 对 Sparse-vLLM 的 DeltaKV 这类“一个逻辑 token 可能只有 latent，没有 raw KV”的表示，需要大幅扩展 node value 类型。

## Sparse-vLLM 当前相关结构

Sparse-vLLM 当前没有跨请求 prefix cache。`Sequence` 只记录 token ids、prefill 进度、当前 chunk size、采样参数等，没有 prefix hit / shared prefix 状态。源码见 `src/sparsevllm/engine/sequence.py:16`。

`StandardCacheManager` 为每条 sequence 分配一个 row，`buffer_req_to_token_slots[row, position]` 保存该请求每个逻辑 token 对应的物理 slot。prefill 和 decode 都从 free stack 分配新 slots；`free_seq()` 在完成或抢占时整条释放。源码见 `src/sparsevllm/engine/cache_manager/standard.py:17`、`src/sparsevllm/engine/cache_manager/standard.py:85`、`src/sparsevllm/engine/cache_manager/standard.py:126`。

`CacheManager` 已经提供调度 hook：`reserved_prefill_slots()`、`prefill_step_free_slots()`、`prompt_admission_budgets()`、`prompt_admission_costs()` 等。源码见 `src/sparsevllm/engine/cache_manager/base.py:261`。

`Scheduler` 先 prefill 后 decode，并且用 cache manager 的 admission budgets 决定新 prompt 是否能进入。显存不足时 decode 会 preempt 一个 running seq，释放其 KV 后重新 prefill。源码见 `src/sparsevllm/engine/scheduler.py:200` 和 `src/sparsevllm/engine/scheduler.py:378`。

`Attention.forward()` 是泛化的：先写 KV 到 `get_layer_store_view()` 返回的物理 cache，再通过 `SparseController.get_read_view()` 得到本层应该读的 slot view。源码见 `src/sparsevllm/layers/attention.py:106`。

`SparseController` 已经把 read view 抽象成 `(active_slots, active_indices, req_indices, context_lens, attn_score, temp_slots)`。OmniKV 在 observation layer 后更新后续层 active slots；DeltaKV 会通过 cache manager reconstruct latent KV 到 scratch slots。源码见 `src/sparsevllm/engine/sparse_controller.py:204` 和 `src/sparsevllm/engine/sparse_controller.py:279`。

DeltaKV 已经是多 pool 表示：full layers pool、sparse full-KV pool、latent pool、row-level raw/latent slot maps、compressed lens、center slots。源码见 `src/sparsevllm/engine/cache_manager/deltakv.py:31`。

DeltaKV 的 scheduler accounting 已经 method-aware：full layers 需要保存 prompt + max decode，sparse layers 不需要为完整 prompt 做逻辑保留，并额外 gate future centers budget。源码见 `src/sparsevllm/engine/cache_manager/deltakv.py:424`。

这些结构说明：prefix cache 应该是 cache-manager-first，而不是 attention-kernel-first。

## 哪种实现更适合 Sparse-vLLM

最合适的是：SGLang 的 radix matching + vLLM 的 block pool 生命周期 + Sparse-vLLM 的 method-specific payload。

原因：

1. Agent 固定前缀通常很长且重复，但用户后缀变化大。Radix tree 能在调度阶段按最长共享前缀排序，也能在 chunk boundary 插入中间结果；vLLM block hash 只能在 full block 链上命中。
2. Sparse-vLLM attention kernel 已经读 slot table。只要 prefix cache attach 时把 shared prefix slots 写入当前 seq 的 row，attention kernel 不需要知道 prefix cache 存在。
3. Sparse-vLLM 的稀疏方法各有不同最终 KV 表示。纯 block hash 只能表达“某 block 的 KV 存在”，无法表达 DeltaKV latent、SnapKV keep indices、OmniKV observation/full layers 等元数据。
4. 直接使用 SGLang per-token radix value 会把 method-specific metadata 混入 tree node，后期维护复杂。更好的方式是 tree node 只保存 prefix directory 和 payload handle，payload 由具体 cache manager 定义。

## 推荐架构

### 1. Prefix Directory

新增 `src/sparsevllm/engine/cache_manager/prefix_cache.py`，提供：

- `PrefixCacheKey`: token ids page view + namespace。
- `PrefixNode`: radix segment、children、parent、payload handles、lock_ref、last_access_time、hit_count、priority、byte_cost。
- `PrefixMatchResult`: `hit_len`、`node`、`payload`、`host_hit_len` 预留字段。
- `RadixPrefixDirectory.match()` / `insert()` / `inc_lock_ref()` / `dec_lock_ref()` / `evict()`.

命名空间必须至少包含：

- model path 或 weights version；
- tokenizer identity；
- dtype；
- tensor parallel rank/world size；
- RoPE config；
- sparse method；
- full attention layer set；
- OmniKV obs/full layer config；
- DeltaKV compressor checkpoint/config hash；
- LoRA/adapters；
- cache salt；
- multimodal embedding hash，如果以后支持 VLM。

命名空间错误应 fail fast，不能 warning 后共享。

### 2. Prefix Payload 接口

在 `CacheManager` 增加抽象 hook：

- `prefix_cache_namespace(self) -> tuple`
- `capture_prefix_payload(self, row_idx: int, upto_len: int) -> PrefixPayload`
- `attach_prefix_payload(self, seq: Sequence, payload: PrefixPayload, hit_len: int) -> None`
- `release_prefix_payload(self, payload: PrefixPayload) -> None`
- `prefix_payload_cost(self, payload: PrefixPayload) -> dict[str, int]`

不同 cache manager 的 payload：

- `StandardPrefixPayload`: 每层 dense KV slots 或 page ids。
- `OmniKVPrefixPayload`: full underlying KV slots；可附带 prefix importance sketch，但不要改变 correctness path。
- `SnapKVPrefixPayload`: 每层保留 slots、逻辑位置映射、context lens；如果前缀尚未完成 SnapKV eviction，先按 dense payload 保存。
- `DeltaKVPrefixPayload`: full-layer slots、sparse raw slots、latent slots、compressed lens、center slots、latent-to-father mapping、slot-to-pos metadata。

### 3. 物理 Page Pool

当前 `free_slots_stack` 是 token slot 粒度。Prefix cache 可以先以 token slots 接入，降低改造风险；但长期应该增加 page/block metadata：

- page size 建议 16 或 32 tokens；
- page 保存 slot range 或 slot ids；
- page/block 有 `ref_cnt`、`hash`、`payload_id`、`last_access_time`；
- evict 只释放 `ref_cnt == 0` 的 cached pages；
- request free 时只 dec ref，不直接释放 shared prefix slots。

这样保留 Sparse-vLLM kernel 的 token slot map，又减少 prefix metadata 和 eviction 操作次数。

### 4. 调度流程

新增 prefill admission 前的 prefix match：

1. 新请求进入 waiting 后，scheduler 对 `prompt_token_ids` 做 prefix match。
2. `hit_len = min(match_len, prompt_len - 1)`，与 vLLM 一样，不能让完整 prompt 全命中导致没有最后 token logits。
3. 命中后，cache manager attach payload，把当前 seq 的 row 前 `hit_len` 填成 shared slots，并增加 payload ref。
4. `seq.num_prefilled_tokens = hit_len`，下一次 prefill 只处理 suffix chunk。
5. chunk prefill 结束时，若 prefix 到达 page boundary，可 `cache_unfinished_req` 式插入，供后续请求复用。
6. request finish 或 preempt 时，release shared payload ref；只释放本请求私有 slots。

对于 in-batch prefix caching：

- waiting 队列用轻量 simulated radix tree 检测相同前缀；
- 如果多个请求共享长前缀且当前 cache 未命中，先运行一个 builder request；
- 其他请求临时 deprioritize 或等待 builder chunk commit；
- 这直接借鉴 SGLang `waiting_queue_radix_tree` 的思想。

### 5. 稀疏方法兼容

OmniKV：

- 第一阶段只做 exact KV sharing：prefix hit 后所有层 KV slots 复用，不改变 OmniKV selection。
- Observation layer 在 suffix/decode 仍会产生当前 query 对 cached prefix 的 attention score，后续 sparse layers 使用当前步骤 active slots。
- 创新优化：缓存 prefix 分段 importance sketch，例如每个 page 的累计 attention/hit count，作为 top-k 候选预筛，减少每步从全 prefix 做 top-k 的开销。该优化必须是 opt-in，并可与无 prefix cache 的 OmniKV 输出对齐验证。

DeltaKV：

- Prefix payload 应在 DeltaKV representation 稳定后插入。长 prefill 中可以在每个 compression boundary 插入，而不是等整条请求结束。
- 命中后 attach 的不是全量 dense slots，而是 full-layer slots + sparse raw center slots + latent slots + compressed lens。
- decode/prefill 读时继续走 `deltakv_reconstruct()`；scratch temp slots 仍是 per-request/per-layer 临时资源，不进入 prefix payload。
- center reservation budget 要把 shared prefix centers 从“新请求成本”里扣掉，否则 admission 会过于保守。

SnapKV / PyramidKV / StreamingLLM：

- 如果 prefix 已完成最终 eviction，payload 只保存 keep 后 slots 和逻辑位置映射。
- 如果 prefix 还在 prefill 中间态，先插 dense/staging payload，最后 chunk 完成后替换为 sparse payload。
- 替换必须是原子操作：新 payload 构建成功后再切换 tree node handle，避免中间失败污染 cache。

QuEST：

- QuEST 的 page selection 与 query 相关，prefix cache 应只复用底层 KV 和 page metadata，不缓存 query-specific selected pages。
- 可以缓存 page-level statistics，加速 candidate page scoring。

## 面向低显存的创新方向

1. Sparse-aware prefix payload：full layers exact，sparse layers 保存 method-native 压缩表示。对 DeltaKV 直接保存 latent；对 SnapKV 保存 keep set；对 OmniKV 可先 exact，后续加 opt-in compaction。
2. Prefix value density eviction：淘汰优先级不只看 LRU，还看 `saved_prefill_flops / resident_bytes`、hit_count、system-prefix priority。agent 系统 prompt 应有更高保留优先级。
3. In-batch prefix builder：同一批 waiting 请求共享长前缀时，只让一个请求先计算并提交 prefix chunk，其他请求随后直接命中，降低冷启动并发浪费。
4. Prefix cold compaction：缓存热度下降或显存压力上升时，把非 full layer 的 old prefix 从 dense KV 转为 DeltaKV latent 或 SnapKV summary。该功能会改变 vanilla exactness，只能对 sparse methods 开启。
5. Prefix page sketch：为每个 cached prefix page 保存轻量统计，如 attention max/mean、hit_count、最近访问层，用于 OmniKV/QuEST 的候选预筛。
6. Hierarchical prefix cache：先实现 GPU-only；后续参考 SGLang HiRadixCache 增加 CPU pinned tier。长 agent 固定系统 prompt 可常驻 CPU，GPU 只保留最近高热 page。
7. Cache namespace manifest：每个 run 保存 prefix cache namespace manifest，包含模型、tokenizer、sparse config、compressor hash、cache salt。实验复现时可以明确说明 cache 是否可复用。

## 分阶段落地建议

### Phase 0: 观测与接口

- 给 `Sequence` 增加 `prefix_cache_hit_len`、`prefix_cache_node`、`prefix_cache_payload_id`。
- 给 `CacheManager` 增加 prefix hook 的默认 NotImplemented 实现。
- 只记录 match/insert/evict 指标，不改变执行路径。

### Phase 1: GPU-only dense prefix cache

- 实现 radix directory + token/page ref count。
- 先支持 vanilla 和 OmniKV exact KV sharing。
- 命中后设置 `num_prefilled_tokens = hit_len`，并复用 shared slots。
- full prompt hit 只命中到 `prompt_len - 1`。
- 支持 cache salt / sparse namespace。

### Phase 2: Chunked prefill 与 in-batch

- 在 chunk boundary 插入 page-aligned prefix。
- 调度器增加 LPM 排序和 in-batch prefix builder。
- admission budget 改为 `free + evictable - protected`。

### Phase 3: Sparse payload

- DeltaKV payload：full slots + raw slots + latent slots + compressed lens + center metadata。
- SnapKV/StreamingLLM payload：keep slots + logical position map。
- OmniKV prefix page sketch 作为可选加速，不影响 baseline path。

### Phase 4: 低显存增强

- sparse-only cold compaction；
- CPU pinned tier；
- hit-value-aware eviction；
- prefix cache event log，用于 benchmark 可复现。

## 必要测试

1. Vanilla exactness：同一 prompt 在 prefix cache on/off 下 greedy 输出一致。
2. Full hit 边界：`prompt_len` 全缓存时只命中到 `prompt_len - 1`，仍能产生 logits。
3. Namespace 隔离：不同 cache salt、LoRA、sparse config、DeltaKV checkpoint 不能共享。
4. Chunked prefill：长系统 prompt 第一个 chunk 插入后，第二个请求能命中该 chunk。
5. Ref count：两个请求共享 prefix，一个完成后另一个仍能 decode；两者都完成后 prefix 才可淘汰。
6. Eviction：显存不足时只淘汰 unlocked cached prefix，不动 running request。
7. DeltaKV：prefix hit 后 `deltakv_reconstruct()` 路径与无 prefix cache 的同方法输出一致或在既有 sparse 误差范围内一致。
8. OmniKV：prefix hit 不改变 observation layer 更新 active slots 的语义。
9. Scheduler：in-batch builder 不饿死短请求；被 deprioritize 的请求最终能运行。
10. Reproducibility：报告每次 benchmark 的 cache hit tokens、evicted tokens、cache namespace、page size、sparse method。

## 最小可行设计摘要

最小可行版本不需要 CPU offload、不需要 cold compaction、不需要改变 attention kernels。

只需要：

- radix directory 做 page-aligned match/insert；
- cache manager 能 attach shared slots 到当前 `buffer_req_to_token_slots`；
- slots 有 ref count，`free_seq()` 不释放 shared prefix slots，只 dec ref；
- scheduler 在 prefill admission 前查 prefix，命中后跳过 prefix tokens；
- chunk boundary 插入 prefix；
- namespace 包含 sparse config 和 cache salt。

这样就能先解决 agent 固定系统提示词重复 prefill 的最大问题，同时为 OmniKV/DeltaKV 的 method-aware payload 留出扩展空间。
