可以，把计划重写成 **vanilla full attention + OmniKV + Quest 三者一起支持** 的版本。

核心设计改成：

`vanilla` 和 `omnikv` 共享同一套 token-block prefix cache，实现放在 `StandardCacheManager`。  
`quest` 复用同一个 prefix cache index，但 attach/materialize/free/evict 以 page 为单位，额外维护 `buffer_req_to_page_slots` 和 `metadata_cache`。  
不支持会 drop / prune / compress 物理 KV 的方法，例如 `streamingllm`、`snapkv`、`pyramidkv`、`deltakv*`。

### **1. 支持范围**

配置上允许：

```python
vllm_sparse_method = ""          # vanilla full attention
vllm_sparse_method = "vanilla"   # alias，normalize 后还是 ""
vllm_sparse_method = "omnikv"
vllm_sparse_method = "quest"
```

不允许：

```python
streamingllm
snapkv
pyramidkv
deltakv*
attention-sink
```

因为这些方法会物理删除、裁剪、压缩或重建 KV，和“缓存完整 prefix KV block”的语义不一致。

新增一个常量比较清楚：

```python
PREFIX_CACHE_SUPPORTED_METHODS = {"", "omnikv", "quest"}
```

配置校验：

```python
if enable_prefix_caching:
    if vllm_sparse_method not in PREFIX_CACHE_SUPPORTED_METHODS:
        raise ValueError(
            "prefix caching only supports vanilla, omnikv, quest"
        )
```

注意：`"vanilla"` 在 `method_registry.py` 里已经会 normalize 成 `""`，所以内部只需要处理 `""`。

### **2. 总体架构**

不要做三套 prefix cache。建议拆成两层：

第一层：通用 `PrefixCacheIndex`。

它负责：

prefix block hash  
longest-prefix lookup  
block chain 恢复  
refcount  
LRU  
eviction 候选管理  
统计信息

第二层：cache-manager 适配。

`StandardCacheManager` 负责 token-slot 级别 attach / materialize / free。  
`OmniKVCacheManager` 继续继承 `StandardCacheManager`，基本不需要单独写逻辑。  
`QuestCacheManager` 负责 page-slot 级别 attach / materialize / free，并复用 Quest page metadata。

结构大概是：

```text
PrefixCacheIndex
    |
    +-- StandardCacheManager
    |       +-- vanilla full attention
    |       +-- OmniKVCacheManager
    |
    +-- QuestCacheManager
            +-- token slots
            +-- page slots
            +-- metadata_cache
```

### **3. 新增配置项**

在 `src/sparsevllm/config.py` 增加：

```python
enable_prefix_caching: bool = False
prefix_cache_block_size: int | None = None
prefix_cache_max_blocks: int | None = None
prefix_cache_salt: str = ""
prefix_cache_cache_decode_blocks: bool = False
```

首版强制：

```python
prefix_cache_cache_decode_blocks = False
```

也就是只缓存 prompt prefill 产生的完整 KV blocks，不缓存 decode 生成 token 的 blocks。

block size 规则：

```python
if vllm_sparse_method == "quest":
    prefix_cache_block_size = quest_chunk_size
else:
    prefix_cache_block_size = prefix_cache_block_size or 16
```

Quest 首版建议强制：

```python
prefix_cache_block_size == quest_chunk_size
```

因为 Quest 的 KV 管理单位是 page，metadata 也是 page 粒度。这样一个 prefix cache block 正好等于一个 Quest page，实现最稳。

vanilla / OmniKV 用普通 token block，默认 16 比较合适，也和常见 vLLM block 粒度接近。

### **4. cache fingerprint 必须隔离三种方法**

即使 token prefix 一样，也不要让 vanilla、OmniKV、Quest 共享同一个 cache entry。

原因是 prefix cache 复用的是已经计算好的 KV，不只是 token ids。不同执行模式下，即使大部分情况下 KV 可能相同，首版也应该保守隔离，避免引入隐蔽 correctness bug。

hash fingerprint 建议包含：

```python
{
    "model": config.model,
    "model_type": config.hf_config.model_type,
    "dtype": str(config.hf_config.torch_dtype),
    "tp_size": config.tensor_parallel_size,
    "method": config.vllm_sparse_method,  # "", "omnikv", "quest"
    "block_size": resolved_prefix_cache_block_size,
    "salt": config.prefix_cache_salt,

    # conservative method config
    "chunk_prefill_accel_omnikv": config.chunk_prefill_accel_omnikv,
    "num_top_tokens": config.num_top_tokens,
    "num_top_tokens_in_prefill": config.num_top_tokens_in_prefill,
    "num_sink_tokens": config.num_sink_tokens,
    "num_recent_tokens": config.num_recent_tokens,
    "full_attn_layers": config.full_attn_layers,
    "obs_layer_ids": config.obs_layer_ids,

    # quest-specific
    "quest_chunk_size": config.quest_chunk_size,
    "quest_skip_layers": config.quest_skip_layers,
}
```

可以稍微保守一点，多包含一些参数。这样会少一些跨配置复用，但 correctness 更稳。

不要把 `rank` 放进 fingerprint，因为 TP 多 rank 需要用同一个 block hash chain。可以包含 `tp_size`，但不要包含 `rank`。

### 

### **5. 新增通用模块**

**`prefix_cache.py`**

新增文件：

```text
src/sparsevllm/engine/prefix_cache.py
```

核心数据结构：

```python
@dataclass
class PrefixCacheBlock:
    key: bytes
    parent_key: bytes | None
    block_size: int
    logical_block_idx: int

    # Standard / OmniKV 使用
    slots: torch.Tensor | None = None       # [block_size], int32, cuda

    # Quest 使用
    page_slot: int | None = None
    page_slots: torch.Tensor | None = None  # optional: [1] or [num_pages]

    ref_count: int = 0
    last_access: int = 0
    child_keys: set[bytes] = field(default_factory=set)
```

通用 index：

```python
class PrefixCacheIndex:
    def lookup_longest_prefix(
        self,
        token_ids: list[int],
        *,
        max_usable_tokens: int,
    ) -> tuple[int, bytes | None, int]:
        ...

    def get_chain(
        self,
        last_key: bytes,
        num_blocks: int,
    ) -> list[PrefixCacheBlock]:
        ...

    def insert_block(
        self,
        block: PrefixCacheBlock,
    ) -> PrefixCacheBlock | None:
        ...

    def touch_chain(self, blocks: list[PrefixCacheBlock]) -> None:
        ...

    def evict_until_freeable(
        self,
        needed_blocks: int,
    ) -> list[PrefixCacheBlock]:
        ...

    def evictable_blocks(self) -> int:
        ...
```

hash 使用 SHA256：

```python
block_key = sha256(
    parent_key
    + packed_block_token_ids
    + extra_fingerprint_bytes
).digest()
```

不要用 Python 内置 `hash()`，因为它受进程 hash seed 影响，不稳定。

### **6. usable hit length 规则**

prefix cache 只缓存完整 block。

并且首版不支持“完整 prompt 全部命中后直接进入 decode”，因为 KV cache 只缓存 K/V，不缓存最后一个 prompt token 的 logits。

所以 usable hit length 应该是：

```python
usable_hit_len = floor((prompt_len - 1) / block_size) * block_size
```

例如：

```text
prompt_len = 128, block_size = 16 -> usable_hit_len = 112
prompt_len = 129, block_size = 16 -> usable_hit_len = 128
prompt_len = 15,  block_size = 16 -> usable_hit_len = 0
```

这样至少会重新 prefill 一个 suffix token，保证能拿到首个 generated token 的 logits。

### 

### **7. 修改**

**`Sequence`**

在 `src/sparsevllm/engine/sequence.py` 增加 prefix cache 状态：

```python
self.prefix_cache_enabled = False
self.prefix_cache_hit_len = 0
self.prefix_cache_hit_blocks = 0
self.prefix_cache_hit_last_key: bytes | None = None
self.prefix_cache_block_size = 0
self.prefix_cache_method = ""
```

`__getstate__()` / `__setstate__()` 需要把这些字段传给 worker。

这里不要传所有 block keys。只传：

```python
hit_last_key
hit_blocks
hit_len
block_size
method
```

原因是 128K prompt、block size 16 时有 8000 个 blocks。如果把每个 block key 都通过 IPC 传给 worker，会让序列化成本变得很高。

worker rank 通过本地 `PrefixCacheIndex.get_chain(hit_last_key, hit_blocks)` 恢复完整 block chain。

### **8. 修改 Scheduler：调度前做 prefix lookup**

在 `CacheManager` base class 增加 hook：

```python
def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
    return

def clear_prefix_cache_hit(self, seq: Sequence) -> None:
    return

def decode_step_free_slots(self) -> int:
    return int(self.num_free_slots)
```

在 `Scheduler.schedule()` 处理 waiting seq 时，`remaining_prefill_tokens` 前增加：

```python
if seq.num_prefilled_tokens == 0 and seq.num_completion_tokens == 0:
    self.memory_oracle.refresh_prefix_cache_hit(seq)
```

首版建议只对：

```python
seq.num_completion_tokens == 0
```

的 fresh prompt 做 prefix lookup。对于已经生成过 token 后被 preempt 的 sequence，不要首版强行 prefix cache，因为那属于 continuation replay，和 prompt-only prefix caching 是两件事。

`remaining_prefill_tokens(seq)` 修改为基于虚拟 prefilled 长度：

```python
virtual_prefilled = max(seq.num_prefilled_tokens, seq.prefix_cache_hit_len)
remaining = seq.num_prompt_tokens - virtual_prefilled
```

当 prompt admission 成功后：

```python
seq.num_prefilled_tokens = seq.prefix_cache_hit_len
```

这样 `_prepare_prefill()` 看到的 `start_idx` 就直接从 prefix hit 之后开始。

### **9. prompt admission cost 修正**

vanilla / OmniKV：

```python
cost = seq.num_prompt_tokens - seq.prefix_cache_hit_len
```

Quest：

```python
suffix_len = seq.num_prompt_tokens - seq.prefix_cache_hit_len
cost = ceil(suffix_len / quest_chunk_size) * quest_chunk_size
```

Quest 必须按 page 取整，因为它的实际分配单位是 page，不是单 token。

对应 hook：

```python
def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
    if not prefix_enabled:
        return {"slots": original_cost}

    suffix_len = seq.num_prompt_tokens - seq.prefix_cache_hit_len

    if method == "quest":
        return {"slots": ceil_div(suffix_len, page_size) * page_size}

    return {"slots": suffix_len}
```

同时 admission budget 应该把可 eviction 的 prefix cache 算进去：

```python
free_for_admission = num_free_slots + prefix_cache_evictable_slots
```

decode 阶段也要改：

```python
logical_free_count = self.memory_oracle.decode_step_free_slots()
```

否则 free slots 为 0、但有 prefix cache 可驱逐时，scheduler 会误以为必须 preempt 活跃请求。

### **10. StandardCacheManager：同时支持 vanilla 和 OmniKV**

这是三方法里最重要的一块。

`StandardCacheManager` 当前服务于 vanilla full attention；`OmniKVCacheManager` 只是继承它。因此只要 prefix cache 逻辑放在 `StandardCacheManager`，vanilla 和 OmniKV 就能共享核心实现。

需要新增状态：

```python
self.prefix_cache = PrefixCacheIndex(...)
self.seq_id_to_prefix_blocks: dict[int, list[PrefixCacheBlock]] = {}
self.seq_id_to_materialized_blocks: dict[int, list[PrefixCacheBlock]] = {}
self.seq_id_to_cached_ranges: dict[int, list[tuple[int, int]]] = {}
self.prefix_runtime_states: dict[int, PrefixRuntimeState] = {}
self.pending_prefix_blocks: dict[int, list[PendingPrefixBlock]] = {}
```

runtime state：

```python
@dataclass
class PrefixRuntimeState:
    parent_key: bytes | None
    next_logical_block_idx: int
    pending_tokens: list[int]
    pending_slots: list[torch.Tensor]
```

pending block：

```python
@dataclass
class PendingPrefixBlock:
    key: bytes
    parent_key: bytes | None
    logical_block_idx: int
    slots: torch.Tensor
    token_ids: list[int]
```

### **11. Standard attach 流程**

在 `StandardCacheManager._prepare_prefill()` 的每个 seq 开头：

```python
if enable_prefix_caching and seq.prefix_cache_hit_len > 0:
    self._attach_prefix_cache_if_needed(seq)
```

`_attach_prefix_cache_if_needed(seq)` 做：

1. 如果 `seq.seq_id` 已经有 row，说明已经 attach 过，直接返回。
2. 用 `seq.prefix_cache_hit_last_key` 和 `seq.prefix_cache_hit_blocks` 从本地 prefix index 找 chain。
3. 如果找不到完整 chain，fail fast。不要 fallback，因为 worker 可能已经没有完整 prompt tokens。
4. 分配 row，但不分配新 KV slots。
5. 把 cached slots 写入：

```python
buffer_req_to_token_slots[row_idx, block_start:block_end] = block.slots
```

6. 设置：

```python
row_seq_lens[row_idx] = seq.prefix_cache_hit_len
```

7. 每个 block：

```python
block.ref_count += 1
```

8. 记录 cached ranges：

```python
seq_id_to_cached_ranges[seq_id].append((block_start, block_end))
```

这样后续 `_allocate(seq_id, chunk_size)` 会从 `hit_len` 之后继续分配 suffix slots。

vanilla 的 full attention read view 会自然读到完整 prefix + suffix。  
OmniKV 的 sparse read view 也会自然基于同一个 `buffer_req_to_token_slots` 工作。

### **12. Standard materialize 流程**

在 `_prepare_prefill()` 中，除了分配 suffix slots，还要把当前 chunk 的 token ids 和 slot mapping 喂给 runtime state。

每凑满一个完整 prefix block，就生成一个 `PendingPrefixBlock`。

注意：不能在 `_prepare_prefill()` 里立刻插入 prefix cache。此时 KV 还没写入所有 layers。

新增 hook：

```python
def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
    ...
```

在 `ModelRunner.run()` 中：

```python
self.sparse_controller.post_forward(seqs, is_prefill)
self.cache_manager.on_forward_end(seqs, is_prefill)
reset_context()
```

`on_forward_end()` 只在 `is_prefill=True` 时 materialize prompt blocks。

流程：

```python
for pending_block in pending_prefix_blocks[seq_id]:
    block = PrefixCacheBlock(...)
    inserted = prefix_cache.insert_block(block)

    if inserted is block:
        block.ref_count = 1  # 当前 seq 正在引用它
        seq_id_to_materialized_blocks[seq_id].append(block)
        seq_id_to_cached_ranges[seq_id].append((start, end))
    else:
        # duplicate block，首版不强制 row switch
        # 当前 seq 继续使用自己刚算出来的 slots，free_seq 时正常释放
        pass
```

首版可以不做 duplicate row switch。这样实现简单，correctness 稳。后续优化可以在 duplicate 时把 row mapping 切到已有 cached slots，并立即释放重复 slots。

### **13. Standard free_seq 改造**

当前 `StandardCacheManager.free_seq()` 会把 row 里的所有 slots 都释放。prefix caching 后不能这么做。

新逻辑：

1. 找到 row。
2. 找到 `cached_ranges`。
3. 对 attached prefix blocks 和 materialized blocks 做：

```python
block.ref_count -= 1
```

4. 只释放不属于 cached ranges 的 slots。
5. 清理 row mapping。
6. row 归还 `free_rows`。

伪代码：

```python
cur_len = row_seq_lens[row_idx]
cached_ranges = merge_ranges(seq_id_to_cached_ranges.pop(seq_id, []))

uncached_segments = complement_ranges(0, cur_len, cached_ranges)

for start, end in uncached_segments:
    slots = buffer_req_to_token_slots[row_idx, start:end]
    push_to_free_slots_stack(slots)

for block in seq_id_to_prefix_blocks.pop(seq_id, []):
    block.ref_count -= 1

for block in seq_id_to_materialized_blocks.pop(seq_id, []):
    block.ref_count -= 1
```

必须保证：

cached block 还在 prefix cache 里时，不能把它的 physical slots 放回 `free_slots_stack`。

### **14. Standard eviction**

prefix cache 占用的是同一个 KV slot pool。

在 `_allocate()` 和 `_allocate_batch()` 前增加：

```python
self._evict_prefix_cache_until_free(size)
```

eviction 规则：

只 eviction：

```python
block.ref_count == 0
```

并且首版建议只 eviction leaf block，避免把 parent evict 掉后留下 orphan child。

也就是说：

```python
can_evict(block):
    return block.ref_count == 0 and len(block.child_keys) == 0
```

evict 一个 Standard block 时：

```python
free_slots_stack[ptr:ptr + block_size] = block.slots
_num_free_slots += block_size
prefix_cache.remove(block.key)
```

如果 parent 的 child set 变空，parent 后续也可作为 leaf 被 eviction。

### **15. OmniKV 适配**

`OmniKVCacheManager` 不需要重新实现 prefix cache。

当前结构：

```python
class OmniKVCacheManager(StandardCacheManager):
    def __init__(...):
        super().__init__(...)
```

保留这个结构即可。

需要注意的是 hash fingerprint 必须包含：

```python
method = "omnikv"
chunk_prefill_accel_omnikv
num_top_tokens
num_top_tokens_in_prefill
num_sink_tokens
num_recent_tokens
full_attn_layers
obs_layer_ids
```

这样 vanilla 和 OmniKV 不会共享 cache entry。

OmniKV 的注意力 kernel 不需要改。prefix cache attach 后，`buffer_req_to_token_slots[row, :hit_len]` 已经指向 cached prefix slots。OmniKV 后续的 `SparseController.get_read_view()` 仍然可以基于这个 row table 生成 active slots。

### **16. QuestCacheManager：page-level prefix cache**

Quest 需要单独适配，因为它除了 token slots，还有：

```python
buffer_req_to_page_slots
metadata_cache[0/1, layer, page_slot, kv_head, head_dim]
```

首版强制：

```python
prefix_cache_block_size == quest_chunk_size
```

这样 prefix block 就是 Quest page。

Quest attach 做：

1. 通过 `hit_last_key + hit_blocks` 取 chain。
2. 分配 row。
3. 对每个 cached block/page：

```python
buffer_req_to_page_slots[row_idx, page_idx] = block.page_slot
buffer_req_to_token_slots[row_idx, start:end] = (
    block.page_slot * page_size + page_offsets
)
row_seq_lens[row_idx] += page_size
block.ref_count += 1
```

4. 不需要重新写 metadata。
5. cached page 的 `metadata_cache` 已经存在于原 page slot 上。

Quest materialize 做：

在 prefill suffix 中，每完成一个完整 page，就记录 `PendingPrefixBlock`。

`on_forward_end()` 之后插入 prefix cache：

```python
block.page_slot = page_slot
block.slots = page_slot * page_size + page_offsets
```

Quest 的 `on_kv_stored()` 已经在每层写完 KV 后更新 page metadata。`on_forward_end()` 发生在所有层 forward 结束后，所以此时 page KV 和 metadata 都已经可复用。

Quest free_seq 做：

1. 按 page_idx 找 cached pages。
2. cached pages 只 decrement refcount，不归还 `free_pages_stack`。
3. uncached pages 归还 `free_pages_stack`。
4. 清空：

```python
buffer_req_to_token_slots[row_idx, :]
buffer_req_to_page_slots[row_idx, :]
row_seq_lens[row_idx]
```

Quest eviction 做：

```python
if block.ref_count == 0 and block is leaf:
    free_pages_stack[_num_free_pages] = block.page_slot
    _num_free_pages += 1
    prefix_cache.remove(block.key)
```

不需要清空 `metadata_cache`。只要 page slot 不在任何 row mapping 中，就不会被读取。debug 模式可以选择填 NaN，但不是必要逻辑。

### **17. Scheduler admission 对 Quest 的特殊修正**

Quest 当前 `num_free_slots` 返回：

```python
_num_free_pages * page_size
```

但是实际分配是 page 粒度。因此 prefix caching 版本应该让 Quest override：

```python
def prompt_admission_cost(self, seq):
    suffix_len = seq.num_prompt_tokens - seq.prefix_cache_hit_len
    return ceil_div(suffix_len, self.page_size) * self.page_size
```

以及：

```python
def reserved_prefill_slots(...):
    return sum(ceil_div(remaining, page_size) * page_size)
```

否则有些 prompt suffix 长度不是 page size 整数倍时，scheduler 会低估 Quest 实际 page 占用。

### **18. 多 rank / tensor parallel 一致性**

Rank 0 的 scheduler 做 lookup，并把 hit 信息写进 `Sequence`：

```python
seq.prefix_cache_hit_len
seq.prefix_cache_hit_blocks
seq.prefix_cache_hit_last_key
```

每个 worker rank 本地都有自己的 `PrefixCacheIndex`，用同样的 hash chain 找对应 block。

要求：

```python
if worker cannot find hit_last_key or chain length mismatch:
    raise RuntimeError
```

不要 fallback 到 recompute。因为 worker 收到的 `Sequence.token_ids` 可能只有 suffix chunk，不一定有完整 prompt。

prefix cache insert / eviction 必须在所有 ranks 上同序发生。由于每个 rank 处理同一批 seq、同一批 token、同一套 block hash，正常情况下 index 结构一致；只是 physical slots 是各 rank 自己的。

### **19. decode CUDA graph 策略**

首版建议先支持 eager path，把 decode CUDA graph 标成 experimental 或直接禁用：

```python
if enable_prefix_caching and decode_cuda_graph:
    logger.warning("prefix caching + decode_cuda_graph is experimental")
```

更稳妥的首版可以先：

```python
if enable_prefix_caching and decode_cuda_graph:
    raise ValueError("prefix caching with decode_cuda_graph will be enabled after validation")
```

原因不是语义冲突，而是 debug 成本高。prefix caching 改的是 row mapping 和 prefill 起点；decode graph 主要依赖静态 metadata tensor 地址。理论上可以支持，但建议在 eager correctness 完成后再打开。

### **20. 文件级改动计划**

新增：

```text
src/sparsevllm/engine/prefix_cache.py
```

修改：

```text
src/sparsevllm/config.py
src/sparsevllm/method_registry.py
src/sparsevllm/engine/sequence.py
src/sparsevllm/engine/scheduler.py
src/sparsevllm/engine/model_runner.py
src/sparsevllm/engine/cache_manager/base.py
src/sparsevllm/engine/cache_manager/standard.py
src/sparsevllm/engine/cache_manager/omnikv.py
src/sparsevllm/engine/cache_manager/quest.py
```

基本不需要改：

```text
src/sparsevllm/layers/attention.py
src/sparsevllm/engine/sparse_controller.py
src/sparsevllm/triton_kernel/*
```

除非只是加 debug/profiler，不应该把 prefix caching 逻辑塞进 attention kernel。

### **21. 推荐实现顺序**

第一步：配置和 hash index。

实现：

```python
enable_prefix_caching
prefix_cache_block_size
prefix_cache_max_blocks
prefix_cache_salt
PREFIX_CACHE_SUPPORTED_METHODS = {"", "omnikv", "quest"}
PrefixCacheIndex
PrefixCacheBlock
```

先写 CPU 单测验证：

```text
相同 tokens -> 相同 block hash
不同 salt -> 不命中
vanilla / omnikv / quest -> 互不命中
parent hash 不同 -> child hash 不同
完整 prompt 命中时仍保留 tail token/block 重新 prefill
```

第二步：Sequence + Scheduler。

实现：

```text
Sequence prefix hit 字段
IPC 序列化字段
refresh_prefix_cache_hit hook
remaining_prefill_tokens 修正
prompt_admission_costs 修正
on_prompt_admitted 后设置 num_prefilled_tokens = hit_len
```

这一步可以用 fake cache manager 单测，不需要 GPU。

第三步：StandardCacheManager 支持 prefix cache。

先只跑 vanilla：

```python
vllm_sparse_method = ""
enable_prefix_caching = True
```

实现：

```text
_attach_prefix_cache_if_needed
_prepare_prefill 中记录 pending blocks
on_forward_end materialize
free_seq refcount-aware
_allocate / _allocate_batch 前 eviction
```

通过 vanilla 后，OmniKV 会自然复用大部分逻辑。

第四步：打开 OmniKV。

只需要放开配置 allowlist，并补 OmniKV correctness tests：

```python
vllm_sparse_method = "omnikv"
enable_prefix_caching = True
```

重点测：

```text
chunk_prefill_accel_omnikv = False
chunk_prefill_accel_omnikv = True
共享长 prefix + 不同 suffix
重复相同 prompt
batch 内 hit/miss 混合
```

第五步：Quest page-level 接入。

实现 Quest 的：

```text
page attach
page materialize
page refcount
page eviction
metadata_cache 复用
Quest admission cost page-align
```

重点测：

```text
prefix hit 后 buffer_req_to_page_slots 没有 -1
cached prefix page 的 metadata_cache 可被 build_decode_view 使用
partial suffix page 不进入 prefix cache
LRU eviction 不释放 active cached pages
```

第六步：统计和 profiler。

建议加：

```text
prefix_cache_lookup_requests
prefix_cache_hit_requests
prefix_cache_hit_tokens
prefix_cache_hit_blocks
prefix_cache_materialized_blocks
prefix_cache_evicted_blocks
prefix_cache_live_blocks
prefix_cache_evictable_blocks
prefix_cache_pinned_blocks
prefix_cache_duplicate_blocks
```

profiler scope：

```text
prefix_cache_lookup
prefix_cache_attach
prefix_cache_materialize
prefix_cache_evict
```

### **22. 必须守住的 correctness invariants**

这些建议直接写成 debug assertions：

```text
prefix hit length 必须是 block_size 整数倍
prefix hit length 必须小于 prompt_len
cached block 必须在所有 layer KV 写完后才可见
cached block ref_count > 0 时不可 eviction
free_seq 不可释放 shared cached slots
Quest cached block 必须同时有 token slots 和 page slot
Quest block size 必须等于 quest_chunk_size
worker 找不到 rank0 指定的 prefix chain 时 fail fast
vanilla / omnikv / quest cache entry 必须 hash 隔离
prefix caching disabled 时现有行为完全不变
```

### **23. 三方法合并后的代码量估计**

三方法一起支持，比“OmniKV only”多不了太多，因为 vanilla 主要是顺带支持，Quest 才是额外成本。

我估计：

```text
核心实现：1,100–1,700 行
含测试：2,000–3,000 行
```

拆分大概是：

```text
prefix_cache.py                         300–450 行
config / registry                         50–90 行
sequence.py                               50–90 行
scheduler.py                             120–220 行
base.py / model_runner.py                 60–100 行
standard.py                              300–500 行
omnikv.py                                  0–30 行
quest.py                                 350–600 行
tests / smoke tests                      800–1,300 行
docs / README                             80–150 行
```

额外支持 vanilla 的成本很低，大概几十行核心代码，因为它已经走 `StandardCacheManager`。真正新增复杂度来自 Quest 的 page slot 和 metadata 生命周期。

### **24. 最终目标效果**

支持后：

```python
Config(
    vllm_sparse_method="",
    enable_prefix_caching=True,
)
```

支持 vanilla full attention prefix cache。

```python
Config(
    vllm_sparse_method="omnikv",
    enable_prefix_caching=True,
)
```

支持 OmniKV prefix cache。

```python
Config(
    vllm_sparse_method="quest",
    enable_prefix_caching=True,
    quest_chunk_size=16,
    prefix_cache_block_size=16,
)
```

支持 Quest prefix cache。

重复系统 prompt、RAG 固定模板、agent tool description 这类 workload 中，第二个及后续请求只需要 prefill 未命中的 suffix。对于 128K prompt、block size 16 的完整 prefix hit，实际 prefill 会从 128K 降到最后 1–16 个 token 左右，具体取决于 prompt 长度是否正好落在 block 边界上。