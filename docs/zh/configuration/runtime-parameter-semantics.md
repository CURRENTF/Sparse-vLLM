# Runtime 参数语义与审计

本文档是本仓库 runtime 和 benchmark 参数的唯一事实来源，面向两类读者：

- 需要运行或比较实验，同时避免静默改变方法语义的用户。
- 后续维护仓库、需要避免重复旧参数错误的维护者。

范围包括影响 inference 行为、准确率、吞吐量、capacity 或模型加载的 runtime 与实验参数，覆盖仓库自有 DeltaKV/HF path、Sparse-vLLM path、LLaVA-OneVision visual-cache path 和主要 benchmark 入口。仅当本仓库暴露 vendored baseline 参数时才记录其内部细节。

## 1. 总体规则

新配置尽可能使用规范名称：

```json
{
  "sparse_method": "deltakv",
  "deltakv_checkpoint_path": "/path/to/compressor",
  "decode_keep_tokens": 2048,
  "sink_keep_tokens": 8,
  "recent_keep_tokens": 128,
  "full_attention_layers": "0,1,2,8,18",
  "deltakv_neighbor_count": 4,
  "deltakv_center_ratio": 0.1,
  "deltakv_latent_dim": 256
}
```

随后添加 backend-specific speed 和 capacity 参数：

```json
{
  "engine_prefill_chunk_size": 4096,
  "gpu_memory_utilization": 0.9,
  "max_num_batched_tokens": 8192,
  "max_num_seqs_in_batch": 8,
  "max_decoding_seqs": 8
}
```

HF DeltaKV 使用显式 HF 参数名：

```json
{
  "hf_prefill_chunk_size": 32768
}
```

重要规则：

- Legacy runtime name 会在 public runtime/API/CLI 边界被拒绝。新 runtime config 不要使用 `chunk_prefill_size`、`num_top_tokens`、`model_cls`、`compressor_path`、`vllm_sparse_method`、`deltakv_path`、`seq_chunk_size` 或 `k_neighbors`。
- `engine_prefill_chunk_size` 是 Sparse-vLLM scheduler chunking；`hf_prefill_chunk_size` 是 HF wrapper/model chunking。
- `decode_keep_tokens=0.17` 在部分 HF path 中表示 ratio，但对 Sparse-vLLM 无效。运行 Sparse-vLLM 前应把 ratio 转换为显式 token count。
- HF 和 Sparse-vLLM 都使用 `sparse_method` 作为 public method selector。
- HF 和 Sparse-vLLM 都使用 `deltakv_checkpoint_path` 作为 public DeltaKV checkpoint path。
- Sparse-vLLM prefix-cache control 是直接的 engine config field：`enable_prefix_caching`、`prefix_cache_block_size`、`prefix_cache_max_blocks` 和 `prefix_cache_salt`。它们不是 HF 参数。
- `compressor_token_group_size` 用于 compressor token grouping；`deltakv_neighbor_count` 用于 selected reference/prototype count。
- LLaVA 的 `--deltakv_checkpoint_path none` 加 `visual_uniform_keep` 不是 learned DeltaKV，而是 visual-token uniform-pruning baseline。

## 2. Runtime 参数流

共有五个主要 runtime 入口。

| 入口 | 参数容器 | 规范化 | 主要使用者 |
| --- | --- | --- | --- |
| `scripts/benchmarks/bench_sparse_vllm.py` | `--hyper_params` JSON | `normalize_runtime_params(..., backend="sparsevllm")` | `sparsevllm.Config`, `Scheduler`, `CacheManager`, `SparseController` |
| `sparsevllm-openai-server` / `python -m sparsevllm.entrypoints.openai.api_server` | CLI flag 加 OpenAI JSON request body | CLI engine kwarg 传给 `LLM(..., **kwargs)`，后者通过 `normalize_runtime_params(..., backend="sparsevllm")` 规范化；request sampling param 直接构造 `SamplingParams` | `LLMEngine`, `AsyncEngineDispatcher`, `/v1/completions` |
| `benchmark/long_bench/pred.py` 和 `benchmark/math_bench/pred.py` | `--hyper_param` JSON 或文件 | `get_generate_api(...)` 在 merge 后规范化 | HF wrapper 或 Sparse-vLLM engine |
| `benchmark/scbench/run_scbench.py` DeltaKV branch | `--hyper_param` JSON dict | `get_generate_api(...)` 规范化 | HF wrapper |
| `benchmark/multimodal/visual_cache/run_visual_cache.py` | 专用 CLI 参数 | 无 global normalizer；构造 `config.deltakv_infer_config` | LLaVA wrapper 和 `KVQwen2Config` |

核心文件：

- `src/deltakv/configs/runtime_params.py`：规范 alias mapping 和 conflict check。
- `src/deltakv/configs/model_config_cls.py`：HF custom config default 和 `set_infer_args`。
- `src/deltakv/get_chat_api.py`：路由到 HF 或 Sparse-vLLM，并解析 `sparse_method`。
- `src/sparsevllm/config.py`：Sparse-vLLM dataclass default 和 engine config validation。
- `src/sparsevllm/engine/scheduler.py`：prefill/decode scheduling 与 admission。
- `src/sparsevllm/engine/cache_manager/base.py`：method-to-cache-manager routing。
- `src/deltakv/modeling/kv_cache.py`：standard DeltaKV 的 HF cache 行为。
- `src/deltakv/modeling/origin_residual_quant_cache.py`：partial direct residual-quant ablation。
- `src/deltakv/modeling/all_origin_residual_quant_cache.py`：all-layer direct residual-quant ablation。

## 3. 规范 Alias Map

normalizer 只接受规范 public runtime name。它把这些名称映射到 backend-native internal field，并对 legacy public key 抛出 `ValueError`。HF config object 内仍存在 internal field，但它们不是有效的 user-facing runtime 参数。

| 规范 key | HF target | Sparse-vLLM target | 含义 |
| --- | --- | --- | --- |
| `sparse_method` | `model_cls` | `vllm_sparse_method` | 方法 selector。 |
| `deltakv_checkpoint_path` | top-level `compressor_path` | `deltakv_path` | DeltaKV checkpoint 目录。 |
| `decode_keep_tokens` | `num_top_tokens` | `decode_keep_tokens` | Decode-time important-token budget。 |
| `prefill_keep_tokens` | `num_top_tokens_in_prefill` | 不支持 | HF prefill/finalization important-token budget。Sparse-vLLM 的 prefill 相关 budget 使用 `decode_keep_tokens`。 |
| `sink_keep_tokens` | `num_sink_tokens` | `num_sink_tokens` | 始终保留的 prefix token。 |
| `recent_keep_tokens` | `num_recent_tokens` | `num_recent_tokens` | 始终保留的 recent tail token。 |
| `full_attention_layers` | `full_attn_layers` | `full_attn_layers` | 保持 full 的 layer，或方法对应的 observation anchor。 |
| `deltakv_neighbor_count` | 相同 | `deltakv_k_neighbors` | Reference/prototype neighbor 数。 |
| `deltakv_center_ratio` | `cluster_ratio` | `cluster_ratio` | Reference center 的 fraction 或 stride-derived rate。 |
| `deltakv_latent_dim` | `kv_compressed_size` | `kv_compressed_size` | DeltaKV latent width。 |
| `deltakv_latent_quant_bits` | `kv_quant_bits` | `kv_quant_bits` | cached DeltaKV-style state 的 quantization bit。 |
| `hf_prefill_chunk_size` | `chunk_prefill_size` | 无 | HF wrapper/model chunk size。 |
| `engine_prefill_chunk_size` | 无 | `chunk_prefill_size` | Sparse-vLLM scheduler chunk size。 |
| `visual_token_prune_only` | 相同 | 无 | LLaVA visual-token-only cache dropping/pruning。 |
| `visual_token_keep_ratio` | 相同 | 无 | LLaVA 保留 eligible visual token 的 ratio。 |
| `enable_prefix_caching` | 无 | 相同 | 为支持的方法启用 Sparse-vLLM prefix KV reuse。 |
| `prefix_cache_block_size` | 无 | 相同 | Prefix-cache hash/materialization block size；QuEST 之外默认为 16。 |
| `prefix_cache_max_blocks` | 无 | 相同 | Live prefix-cache block 的可选上限；只淘汰 unreferenced leaf block。 |
| `prefix_cache_salt` | 无 | 相同 | 额外 fingerprint salt，用于隔离本应不共享的 cache entry。 |

被拒绝的 legacy runtime name：

| Legacy key | 替代项 | 旧名称的问题 |
| --- | --- | --- |
| `model_cls`, `vllm_sparse_method` | `sparse_method` | Backend-specific method selector name 泄漏到 shared config。 |
| `compressor_path`, `deltakv_path` | `deltakv_checkpoint_path` | Backend-specific checkpoint name 使跨 backend config 产生歧义。 |
| `chunk_prefill_size` | `hf_prefill_chunk_size` 或 `engine_prefill_chunk_size` | 同一拼写在两个 backend 表示不同的 speed/capacity 行为。 |
| `num_top_tokens`, `num_top_tokens_in_prefill` | `decode_keep_tokens`；仅 HF 使用 `prefill_keep_tokens` | Count/ratio/per-layer 语义随 backend 不同。 |
| `num_sink_tokens`, `num_recent_tokens`, `tail_token_size` | `sink_keep_tokens`, `recent_keep_tokens` | Internal cache name 泄漏到实验 config。 |
| `full_attn_layers` | `full_attention_layers` | Internal layer-routing name 泄漏到 shared config。 |
| `seq_chunk_size` | `compressor_token_group_size` | 它描述 token grouping，但也曾被用作 cluster-neighbor fallback。 |
| `k_neighbors`, `deltakv_k_neighbors` | `deltakv_neighbor_count` | Backend/internal name 隐藏了 selected-reference 含义。 |
| `cluster_ratio` | `deltakv_center_ratio` | DeltaKV-specific center/prototype rate。 |
| `kv_compressed_size` | `deltakv_latent_dim` | 表示 latent width，不是 token count。 |
| `kv_quant_bits` | `deltakv_latent_quant_bits` | 被量化的对象随方法而异；shared config 中名称必须明确。 |
| `deltakv_visual_compress_only` | `visual_token_prune_only` | 名称包含 “DeltaKV” 和 “compress”，但无 checkpoint path 实际是 uniform pruning。 |
| `deltakv_visual_keep_ratio` | `visual_token_keep_ratio` | 与旧名称绑定。 |

## 4. 方法路由

### 4.1 HF `sparse_method`

`get_generate_api(..., backend="hf")` 使用 public `sparse_method`，将其映射到 internal HF wrapper class。支持的仓库自有或路由值包括：

| `sparse_method` | 主要行为 | Checkpoint 要求 |
| --- | --- | --- |
| `auto` | 普通 HF `AutoModelForCausalLM`；可选 chunked-forward monkey patch。 | 无 compressor。 |
| `deltakv` | Standard DeltaKV HF wrapper。 | 可选，但 learned compressor 需要 checkpoint。 |
| `full_deltakv` | Full-layer DeltaKV compression wrapper。 | 通常依赖 checkpoint。 |
| `origin_residual_quant` | 对 full-attention layer 直接执行 token-space residual quant；sparse layer 使用 standard path。 | 取决于 config，可不提供 checkpoint；但 cluster metadata 可以从 checkpoint config 加载。 |
| `all_origin_residual_quant` | 对每个 layer 直接执行 token-space residual quant；要求 `use_cluster=True`。 | reconstruction path 不需要 learned compressor。 |
| `snapkv` | HF SnapKV wrapper。 | 无 DeltaKV checkpoint。 |
| `pyramidkv` | HF PyramidKV wrapper。 | 无 DeltaKV checkpoint。 |
| `omnikv` | 加载 DeltaKV wrapper，并设置 `use_compression=False` 和 `use_cluster=False`。 | 无 DeltaKV checkpoint。 |
| `quest` | Patch Quest baseline attention。 | 无 DeltaKV checkpoint。 |
| `palu`, `kivi`, `adakv`, `kvzip` | Baseline adapter。 | 取决于 baseline。 |

例如，`sparse_method="deltakv"` 映射到 standard HF DeltaKV wrapper，因为 Triton variant 是 Sparse-vLLM-specific implementation choice。

### 4.2 Sparse-vLLM `sparse_method`

`backend="sparsevllm"` 使用 public `sparse_method`。engine 在内部将规范化值存为 `vllm_sparse_method`。

已知 Sparse-vLLM method string：

| 方法 | Cache manager 行为 |
| --- | --- |
| `""` 或规范 `vanilla` | Standard dense cache manager。 |
| `streamingllm`, `attention-sink`, `attention_sink` | StreamingLLM cache manager；alias 规范为 `streamingllm`。 |
| `snapkv` | SnapKV cache manager。 |
| `h2o` | 使用累计归一化 token-importance 评分及 heavy-hitter + recent 物理淘汰的 H2O cache manager。Prefill 使用 attention mass；优化后的 decode 使用 max-reduced raw QK logit，再按 token 归一化。 |
| `pyramidkv` | 带 PyramidKV layer budget 的 SnapKV cache manager。 |
| `omnikv` | OmniKV cache manager。 |
| `quest` | Quest cache manager。 |
| `rkv`, `r-kv`, `r_kv` | R-KV cache manager，支持 physical decode eviction、query-cache attention importance scoring 和 key-redundancy scoring。 |
| `skipkv`, `skip-kv` | SkipKV cache manager，支持 physical decode eviction 和 sentence-aware redundancy signal。 |
| `deltakv` | 维护中的 compressor-backed DeltaKV runtime。 |
| `deltakv-less-memory`, `deltakv-less-memory-cudagraph` | 为旧 config 和 regression manifest 保留的 legacy alias。它们规范为 `deltakv`；cudagraph alias 还会请求 decode CUDA Graph。 |

Sparse-vLLM 对仓库当前 validation run 使用的 DeltaKV-family path 支持 text-only Qwen3。Qwen3 DeltaKV 改动对 alignment 敏感：报告 benchmark result 前，必须通过 HF-vs-Sparse logits 检查验证 qk-norm、RoPE theta/dtype、sparse-reference storage 和 full-layer KIVI view。

新命令使用 `sparse_method="deltakv"`。真实 Sparse-vLLM DeltaKV run 需要匹配的 `deltakv_checkpoint_path`；缺失 checkpoint 仅适用于显式设置 `allow_missing_deltakv_path=True` 的构造测试。

### 4.3 Sparse-vLLM Prefix Cache

Prefix cache 是 Sparse-vLLM engine feature，不是通用 HF runtime feature。它在同一个 live engine process 内跨 request 复用 request-context KV block，process 重启后不会保留。

支持的方法：

| `sparse_method` | Prefix-cache storage unit | 说明 |
| --- | --- | --- |
| `vanilla` / `""` | token block | 使用 `StandardCacheManager`。 |
| `omnikv` | token block | 复用 `StandardCacheManager`；fingerprint 仍包括 `omnikv` 设置。 |
| `quest` | QuEST page | 要求 `prefix_cache_block_size == quest_chunk_size`。 |

当 `enable_prefix_caching=true` 时，不支持的方法会快速失败：StreamingLLM、attention-sink alias、SnapKV、PyramidKV 和所有 DeltaKV-family 方法。这些方法会以无法等价复用完整 request-context KV prefix 的方式物理剪枝、压缩、重建或重映射 KV。

参数语义：

| 参数 | 含义 |
| --- | --- |
| `enable_prefix_caching` | Boolean 或显式 true/false 字符串。启用 scheduler lookup 及 cache-manager attach/materialize/free/evict hook。 |
| `prefix_cache_block_size` | 正整数或 `null`。vanilla/OmniKV 默认为 16。QuEST 解析为 `quest_chunk_size`；任何不同的显式值都会被拒绝。 |
| `prefix_cache_max_blocks` | 可选正整数上限。设置后，插入时只淘汰 unreferenced leaf block；不淘汰 referenced block。 |
| `prefix_cache_salt` | 折叠进 cache fingerprint 的字符串。用于有意隔离不应共享 cache entry 的 run。 |

正确性约束：

- Cache key 包含 model path、model type、dtype、TP size、sparse method、block size、salt 和 method-specific sparse setting。
- 不直接使用 full-prompt hit；至少重新计算一个 suffix token，从而正常产生第一个 generated token 的 logits。
- 只有对应 forward step 的所有模型 layer 都写入 KV 后才 materialize block。Prompt 和 decode input token 追加到同一 block-size buffer；完整 block 插入 prefix cache，request free 时丢弃不完整 trailing block。
- Active cached block 采用 refcount，仍被引用时不能淘汰或返回 free-slot/page pool。
- 当 `decode_cuda_graph_capture_sampling=false` 时，`vanilla`、`omnikv` 和 `quest` 支持 prefix cache 与 `decode_cuda_graph=true` 组合。`tensor_parallel_size>1` 时，每个 rank 保留 rank-local mirrored prefix cache，具有稳定 logical block ID 和 rank-local KV payload。

API serving 以 `--kebab-case` engine flag 传入：

```bash
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=2346 \
PYTHONPATH=src .venv/bin/python -m sparsevllm.entrypoints.openai.api_server \
  --model /models/Qwen2.5-7B-Instruct-1M \
  --served-model-name qwen25-7b-1m \
  --sparse-method vanilla \
  --enable-prefix-caching true \
  --prefix-cache-block-size 16 \
  --engine-prefill-chunk-size 4096 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768
```

Benchmark JSON 使用相同 snake_case field name：

```json
{
  "sparse_method": "vanilla",
  "enable_prefix_caching": true,
  "prefix_cache_block_size": 16,
  "prefix_cache_max_blocks": 4096,
  "engine_prefill_chunk_size": 4096
}
```

测量 prefix-cache 收益时，应向同一个 engine process 发送具有相同 prompt prefix 的重复 request。为每个 sample 新建 engine 的 benchmark 无法测量 cache reuse。

## 5. Unknown-Key 行为

这一点很重要，因为“命令运行成功”不代表“参数已生效”。

| Backend | 当前行为 |
| --- | --- |
| HF DeltaKV custom config | `set_infer_args` 只在 config 具有对应 attribute 时应用 key。Unknown key 会记录 `There is NO <key> in Custom Config!`，通常被忽略。 |
| Sparse-vLLM | `LLMEngine.__init__` 把 kwarg 过滤为 `sparsevllm.Config` dataclass field。默认对 unknown key 抛出 `ValueError`；仅当 `allow_unknown_config_keys=True` 时记录并忽略。 |
| LLaVA visual script | 构建选定的 `infer_config`；无关 CLI 参数不进入 config。 |
| SCBench DeltaKV branch | 复制 `hyper_param`，移除 `sparse_method` 和 `cuda_device`，再把剩余参数发给 `get_generate_api`。 |

除非 shared mega-config 已规范化并检查 ignored key，否则不要跨 backend 使用。

## 6. 影响准确率的参数

### 6.1 Token Keep Budget

| 参数 | HF 行为 | Sparse-vLLM 行为 | 风险 |
| --- | --- | --- | --- |
| `decode_keep_tokens` | 在 token-selection helper 中，可以是 integer count 或不大于 `1.0` 的 float ratio；部分 HF wrapper 也允许使用 list/tuple 或逗号字符串表示 per-observation-layer budget。 | 必须是显式 integer-like count；normalizer 拒绝 ratio-style float。 | 把 HF 的 `0.17` 复制到 Sparse-vLLM 是错误的。 |
| `prefill_keep_tokens` | Prefill/finalization token selection budget；HF wrapper 也可能允许 list/逗号字符串。 | 不支持。Sparse-vLLM 的 prefill/finalization budget 复用 `decode_keep_tokens`。 | 传给 Sparse-vLLM 时默认是 unknown config key。 |
| `sink_keep_tokens` | Cache wrapper 和稀疏方法保留的 prefix token。 | Cache manager 和 sparse controller 保留的 prefix token。 | 直觉通常相同，但 storage layout 不同。 |
| `recent_keep_tokens` | `CustomConfigMixin` 还会复制到 internal recent/tail buffer。 | Scheduler/cache manager 直接使用。 | 跨 backend 应保持同步。 |

HF ratio 语义来自 `src/deltakv/modeling/token_select.py`。Sparse-vLLM 使用 engine/cache planning，没有 target context length 时无法安全解释 ratio。应显式转换：

```text
int(131072 * 0.17) = 22282
```

### 6.2 Layer Routing

| 参数 | HF 行为 | Sparse-vLLM 行为 |
| --- | --- | --- |
| `full_attention_layers` | `CustomConfigMixin` 解析为列表。Standard DeltaKV 将其用作 full-attention layer 和 selection anchor。 | 内部解析为列表。Sparse-vLLM 根据该值派生 observation layer。 |
| `snapkv_num_full_layers` | HF SnapKV 相关参数。 | SnapKV manager 可以保留前部 full layer。 |

特殊情况：

- `full_attention_layers` 字符串相同不足以保证 backend parity，还需检查所选方法如何解释它。

OmniKV full layer 可以离线自动选择：

```bash
PYTHONPATH=$PWD:$PWD/src python scripts/analysis/select_omnikv_full_layers.py \
  --model-path <MODEL_DIR> \
  --longbench-root <LONGBENCH_ROOT> \
  --config-dir benchmark/long_bench/config \
  --dataset narrativeqa \
  --output-dir <OUTPUT_DIR> \
  --num-full-layers 6 \
  --num-samples 32 \
  --topk 2048 \
  --random-decode-points-per-sample 8 \
  --num-sink-tokens 0 \
  --num-recent-tokens 32 \
  --prefill-chunk-size 512 \
  --torch-dtype bfloat16 \
  --device cuda
```

selector 写入 `<OUTPUT_DIR>/selected_full_layers.json`；在 OmniKV Sparse-vLLM config 中使用其中的 `full_attention_layers` 值，例如：

```json
{
  "sparse_method": "omnikv",
  "full_attention_layers": "0,2,4,11,16,22",
  "decode_keep_tokens": 4096,
  "recent_keep_tokens": 32,
  "sink_keep_tokens": 0,
  "engine_prefill_chunk_size": 512
}
```

这是 offline calibration step，不是自动 `LLM(...)` runtime mode。设置 `full_attention_layers` 后，Sparse-vLLM 从中派生 internal observation layer。`observation_layers` 不是受支持的 runtime key。

### 6.3 SnapKV、H2O、PyramidKV、OmniKV、Quest

| 参数 | 主要使用者 | 含义 |
| --- | --- | --- |
| `snapkv_window_size` | HF SnapKV/PyramidKV 和 Sparse-vLLM SnapKV/DeltaKV-SnapKV | Local observation/recent window。 |
| `pool_kernel_size` | HF token selection 和部分 Sparse-vLLM 方法 | Score smoothing kernel；使用者因方法而异。 |
| `h2o_decode_budget` | Sparse-vLLM H2O | 最后一个 prefill chunk 结束后、以及 decode 超预算时保留的 token 总数。该总数同时包含 heavy hitter 和 recent token；H2O 不额外强制 sink budget。默认值：`4096`。 |
| `h2o_prefill_budget` | Sparse-vLLM H2O | 非最后 prefill chunk 结束后保留的较高 token 总预算，必须不小于 `h2o_decode_budget`。默认值：`8192`。 |
| `h2o_recent_ratio` | Sparse-vLLM H2O | 两种总预算中为最近物理 token 保留的比例；剩余预算按累计 attention mass 选择。必须严格位于 0 和 1 之间。默认值：`0.5`。 |
| `h2o_prefill_score_window` | Sparse-vLLM H2O | 归一化 prefill score kernel 使用的当前 chunk 尾部 query token 数，范围为 1 到 128。默认值：`128`。 |
| `pyramid_layer_ratios` | Sparse-vLLM PyramidKV | 显式 per-KV/full-attention-layer keep ratio。对于 mixed-attention model，legacy full Transformer-layer list 会投影到 KV layer。 |
| `pyramidkv_start_layer`, `pyramidkv_start_ratio`, `pyramidkv_least_layer`, `pyramidkv_least_ratio` | HF 和 Sparse-vLLM PyramidKV-style path | 自动生成 budget schedule；layer position 按 KV/full-attention layer 计数。 |
| `quest_chunk_size` | Sparse-vLLM Quest | Quest page size。 |
| `sink_keep_tokens` + `decode_keep_tokens` + `recent_keep_tokens` | Sparse-vLLM Quest | Quest token budget，在 config 构造时一次性派生。 |
| `chunk_size` | HF Quest adapter | HF 上的 Quest chunk/page size。 |
| `decode_keep_tokens` | HF Quest adapter | HF 上的 Quest token budget。 |

Quest 很能体现“相同研究思想，不同参数 surface”。HF config：

```json
{"backend": "hf", "sparse_method": "quest", "decode_keep_tokens": 1024, "chunk_size": 16}
```

使用一个 total budget，而 Sparse-vLLM 将相同 total 表示为：

```json
{"backend": "sparsevllm", "sparse_method": "quest", "sink_keep_tokens": 0, "decode_keep_tokens": 992, "recent_keep_tokens": 32, "quest_chunk_size": 16}
```

Sparse-vLLM 拒绝已移除的 `quest_token_budget` input，而不是静默忽略。上例派生的 total budget 同为 1024 token。

### 6.4 R-KV

Sparse-vLLM R-KV 在 cache manager 中保留小型 per-layer query cache，用它在 decode eviction 时为 candidate KV token 评分。

| 参数 | 主要使用者 | 含义 |
| --- | --- | --- |
| `rkv_compression_interval` | SparseController | 两次 R-KV decode eviction 之间的 generated-token buffer size。 |
| `rkv_observation_tokens` | RKVCacheManager | R-KV observation window 使用的 recent query state 数。与 `rkv_compression_interval` 独立，默认为 `8`，且必须 `<= 128` 并 `<= rkv_compression_interval`。 |
| `rkv_alpha` | RKVCacheManager | 论文 joint score lambda：`alpha * importance - (1 - alpha) * redundancy`。 |
| `rkv_redundancy_window` | RKVCacheManager | 用于 key redundancy 评分的 candidate window。`0` 对完整 candidate set 评分；正数选择 trailing-window approximation。 |

R-KV importance score 通过 shared prefill score kernel，根据 cached observation query 和当前 K cache 计算。decode attention-score buffer 不是 R-KV 的唯一事实来源。

## 7. DeltaKV HF Cache 语义

HF DeltaKV 行为由 `KVQwen2Config`、`KVQwen3Config` 和 `KVLlamaConfig` 控制，三者都使用 `CustomConfigMixin`。

`CustomConfigMixin` 的重要 default：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `deltakv_latent_dim` | `128` | Latent KV width 的 public name，内部存为 `kv_compressed_size`。 |
| `compressor_token_group_size` | `1` | Non-cluster compressor reference 的 token group size。 |
| `deltakv_neighbor_count` | `1` | Cluster/ref neighbor count，不再从 token group size fallback。 |
| `layer_chunk_size` | `1` | 历史 layer grouping；standard runtime 在 compression path 中 assert 它保持为 `1`。 |
| `recon_mode` | `delta_in_latent` | Standard compressor residual mode。 |
| `ref_mode` | `avg` | Non-cluster compression 的 chunk reference mode。 |
| `use_compression` | `False` | Standard cache 是否使用 learned compressor；checkpoint config 可能覆盖。 |
| `use_cluster` | `True` | 是否选择 cluster/prototype path。 |
| `deltakv_center_ratio` | `0.1` | Center sampling ratio；HF cluster path 内部实现为 `cluster_step=max(1, int(1/ratio))`。 |
| `stride_alpha` | `0.0` | Dynamic center stride growth；`0.0` 表示 fixed stride。 |
| `deltakv_latent_quant_bits` | `0` | `2` 或 `4` 在支持的 path 启用 packed quantized storage。 |
| `hf_prefill_chunk_size` | `100000000` | HF wrapper chunk size；较大 default 对许多 prompt 实际避免 manual chunking。 |

### 7.1 Standard `CompressedKVCache`

文件：`src/deltakv/modeling/kv_cache.py`。

当 `use_cluster=False`：

- `use_compression=True` 存储 learned latent residual。
- `compress()` 按 `compressor_token_group_size` 对 token 分组，使用 mean reference 等 chunk base，并存储 `compressor_down(kv) - compressor_down(base)`。
- Reconstruction 使用 `compressor_up(comp_kv) + base`。
- 如果 `use_compression=False` 且 `deltakv_latent_quant_bits=2` 或 `4`，buffer 中存储的 historical KV 是直接量化 raw KV，不是 learned DeltaKV latent。
- 如果 `use_compression=False` 且 `deltakv_latent_quant_bits=0`，直接存储 historical key/value tensor。

在 full-attention layer 中，standard `CompressedKVCache` 当前只在 main update path 中保留 sink 加 buffer，不压缩该 layer。

### 7.2 Standard `ClusterCompressedKVCache`

文件：`src/deltakv/modeling/kv_cache.py`。

当 `use_cluster=True`：

- Sink token 作为初始 center 插入。
- 使用 `deltakv_center_ratio` 从 history 中采样更多 center，可选使用 `stride_alpha` 的 dynamic stride。
- 每个 token 使用 `cluster_metric` 和 `cluster_on_kv` 选择最多 `deltakv_neighbor_count` 个 center。
- 当 `use_compression=True` 时，cluster compression 存储 learned latent residual：

```text
compressor_down(token_kv) - compressor_down(mean(selected_centers))
```

- 如果 `deltakv_latent_quant_bits=2` 或 `4`，量化的是 latent `comp_kv`，不是原始 full KV，也不是 token-space residual。
- 当 `use_compression=False` 时，同一 cache 可以运行 direct residual-quant path：

```text
residual = token_kv - mean(selected_centers)
```

- 该 direct path 中，`deltakv_latent_quant_bits=2` 或 `4` 直接量化 token-space residual。compression 或 reconstruction 不使用 learned `compress_down` 或 `compress_up` module。
- 对 batched left-padding run，direct path 能识别 padding：padding token 使用 invalid position 存储，不能成为有效 reference center。

当前 LLaVA benchmark 通过 method `deltakv` 和真实 checkpoint 暴露该路径。无 checkpoint 的多模态 run 属于独立 `visual_uniform_keep` baseline，不是 DeltaKV。

### 7.3 `origin_residual_quant`

文件：

- `src/deltakv/modeling/origin_residual_quant_cache.py`
- `src/deltakv/modeling/qwen2/qwen2_origin_residual_quant_inference.py`
- 等价 Qwen3/Llama wrapper。

行为：

- Full-attention layer 直接存储 token-space residual：

```text
residual = token_kv - reference
```

- `deltakv_latent_quant_bits=2` 或 `4` 时量化 residual。
- Sparse layer 通过委托 `super().update(...)` 继续使用原始 DeltaKV cache path。
- Clustered variant 选择 cluster center 作为 reference，但存储 token-space residual，而不是 learned latent residual。

该路径适合 ablation；如果非 full sparse layer 仍走 standard path，它并不表示“所有位置都没有 compressor”。

### 7.4 `all_origin_residual_quant`

文件：

- `src/deltakv/modeling/all_origin_residual_quant_cache.py`
- `src/deltakv/modeling/qwen2/qwen2_all_origin_residual_quant_inference.py`
- 等价 Qwen3/Llama wrapper。

行为：

- 要求 `use_cluster=True`。
- 对每个 layer 应用 token-space residual quantization。
- 在 cache update path 中删除/忽略 `compressor_down` 和 `compressor_up`。
- Reconstruction：

```text
reconstructed_kv = dequantized_residual + mean(selected_centers)
```

该路径最符合“使用 cluster/ref token，但不使用 learned compressor，直接量化 residual”。

## 8. 已移除的 `seq_chunk_size` 与拆分语义

`seq_chunk_size` 是最容易误解的 DeltaKV HF 参数，现已从 public runtime/training 参数中移除。

它至少有四种语义或历史用途：

| Code path | 实际用途 |
| --- | --- |
| `CompressedKVCache.compress()` 中的 non-cluster compression | 把 KV reshape 为 chunk 并计算 chunk reference base 的 token group size，现为 `compressor_token_group_size`。 |
| E2E compressor training/inference helper | Sequence reference 的 token group size。 |
| `origin_residual_quant` full-layer path | 构建 token-space residual base 的 token group size。 |
| `CustomConfigMixin.finalize_cluster_args()` | Cluster neighbor 的 legacy fallback，现已移除。 |

最后一种用途在名称语义上错误。`seq_chunk_size` 看起来表示“compression chunk 包含多少 sequence token”，但在 cluster inference 中可能静默变为“每个 token 平均多少 center”。

当前行为：

```python
if config.use_cluster and config.deltakv_neighbor_count is None:
    raise ValueError("deltakv_neighbor_count is required")
```

实践规则：

- Token grouping 使用 `compressor_token_group_size`。
- Cluster/reference neighbor count 使用 `deltakv_neighbor_count`。
- 历史 checkpoint `config.json` 可以在 load 时内部迁移，使旧权重仍可测试。这属于 artifact schema migration，不代表接受旧 public runtime 参数。

## 9. DeltaKV Sparse-vLLM 语义

Sparse-vLLM DeltaKV 不是 HF wrapper 的逐行移植。它具有 engine-owned scheduling、physical cache slot、compressed slot map 和 method-specific cache manager。

关键 Sparse-vLLM public name 与 internal field：

| Public 参数 | Internal field | 默认值 | 行为 |
| --- | --- | --- | --- |
| `sparse_method` | `vllm_sparse_method` | `vanilla` / `""` | Sparse method selector。 |
| `deltakv_checkpoint_path` | `deltakv_path` | `None` | Compressor 权重/config 的 DeltaKV checkpoint path。 |
| `deltakv_neighbor_count` | `deltakv_k_neighbors` | `4` | Reconstruction 使用的 center 数。 |
| `deltakv_center_ratio` | `cluster_ratio` | `0.1` | Center/prototype rate 和 capacity input。 |
| `cluster_metric` | 相同 | `l2` | Center scoring metric。 |
| `deltakv_latent_dim` | `kv_compressed_size` | `128` | Latent dimension。 |
| `deltakv_latent_quant_bits` | `kv_quant_bits` | `4` | DeltaKV-style state 的 quantization bit。 |
| `deltakv_full_pool_reserve_ratio` | 相同 | `0.1` | 为 full-KV pool 保留的 KV memory 比例。 |
| `deltakv_sparse_decode_backend` | 相同 | `auto` | Sparse decode backend。`Config` 构造时，如果已安装 `flash_attn`，`auto` 解析为 `fa2`，否则为 `custom`。 |
| `deltakv_triton_gather_heads_per_program`, `deltakv_triton_reconstruct_heads_per_program` | 相同 | `4` | Gather/reconstruct kernel 的 Triton grouping control；不控制 materialized sparse-view kernel。 |
| `deltakv_triton_materialize_block_tokens` | 相同 | `16` | Materialized sparse-view kernel 的 token block size。 |
| `allow_unknown_config_keys` | 相同 | `False` | 显式允许忽略 unknown Sparse-vLLM config key。 |
| `allow_missing_deltakv_path` | 相同 | `False` | 缺少 compressor 权重时，仅供构造测试使用的 escape hatch。可报告 benchmark run 不得使用。 |

`bitsandbytes` 是声明的 package dependency，因为 4-bit 和 8-bit loading path 会在 runtime 导入它。普通 `pip install -e .` 应安装该依赖；手动维护的环境需显式包含。

`benchmark/microbench.py` 在每个 result row 记录 input `engine_hyper_params` 和构造后的 `resolved_engine_config`。通过 `resolved_engine_config.deltakv_sparse_decode_backend` 审计 `auto` backend run 实际使用 `fa2` 还是 `custom`。

### 9.1 基于 Compressor 的 DeltaKV

方法 `deltakv` 是维护中的 compressor-backed DeltaKV runtime，真实 run 要求 `deltakv_checkpoint_path`。

### 9.2 Legacy `deltakv-less-memory*` Alias

文件：

- `src/sparsevllm/engine/cache_manager/deltakv_runtime.py`
- `src/sparsevllm/engine/cache_manager/deltakv_less_memory.py`
- `src/sparsevllm/engine/cache_manager/deltakv_less_memory_cuda_graph.py`

保留历史 `deltakv-less-memory` 名称，使旧 config 和 regression manifest 仍能加载。它们规范为 public `deltakv` runtime；该 runtime 初始化 compressor module 并要求 `deltakv_path`。`deltakv-less-memory-cudagraph` alias 还会设置 `decode_cuda_graph=True`。

精简 runtime 支持两种存储组合：

| Full-layer storage | Sparse-layer storage |
| --- | --- |
| `full_layer_kv_quant_bits=0` | `deltakv_latent_quant_bits=0` |
| 启用 full-layer KIVI 的 `full_layer_kv_quant_bits=4` | `deltakv_latent_quant_bits=4` |

其他 bit 组合会快速失败。`deltakv_neighbor_count`、`deltakv_center_ratio`、`full_attention_layers`、`sink_keep_tokens`、`recent_keep_tokens` 和 `decode_keep_tokens` 仍影响 center/reference selection、full-layer routing 和 sparse-view budget。

Sparse-vLLM smoke command 示例：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv \
  --output_len 4 \
  --temperature 0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"deltakv_checkpoint_path":"<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor","deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_dim":256,"deltakv_latent_quant_bits":4,"full_layer_kv_quant_bits":4,"enable_full_layer_kivi_quant":true,"deltakv_full_pool_reserve_ratio":0.2}'
```

与 HF 的重要差异：

- `deltakv_neighbor_count` 映射到 `deltakv_k_neighbors`，不是旧 `k_neighbors`。
- `deltakv_center_ratio` 影响算法行为和 memory planning。
- `decode_keep_tokens` 必须是 integer budget。
- `full_attention_layers` 可能派生 observation layer。
- 默认情况下，unknown Sparse-vLLM config key 和缺失 DeltaKV compressor path 会快速失败。`allow_missing_deltakv_path` 只用于构造测试，不得用于可报告 run。
- Sparse-vLLM 方法可以在 prefill/decode 中物理淘汰或重映射 cache slot，与 HF `DynamicCache` wrapper 不同。

## 10. Speed 与 Capacity 参数

这些参数不只影响速度。在 Sparse-vLLM 中，它们可能改变 admission、queueing，以及 benchmark 是否测量预期 batch。

| 参数 | Backend | 含义 |
| --- | --- | --- |
| `engine_prefill_chunk_size` | Sparse-vLLM | `all_chunked` 下每个 step、每个 sequence 调度的最大 prefill chunk。不要为 `long_bs1full_short_batch` 设置；该 policy 从 `long_prefill_offload_threshold` 派生 chunk size。 |
| `hf_prefill_chunk_size` | HF | Long input forward 的 wrapper/model chunk size；较大值通常表示“不 chunk”。 |
| `max_model_len` | Sparse-vLLM | Prompt 加 generated token 的 engine hard capacity，影响 allocation 和 request validation。 |
| `long_prefill_offload_threshold` | Sparse-vLLM | `long_bs1full_short_batch` 下 complete batched short prefill 与 isolated chunked RawKV offload 的精确边界。默认 `98304` token（96K），同时成为该 policy 的 `chunk_prefill_size`。 |
| `max_num_batched_tokens` | Sparse-vLLM | 一个 step 的 aggregate scheduler token 上限；memory heuristic 可能降低。`all_chunked` 允许小于 `engine_prefill_chunk_size`；`long_bs1full_short_batch` 至少规范到 `long_prefill_offload_threshold`，使边界 prompt 能原子容纳。 |
| `max_num_seqs_in_batch` | Sparse-vLLM | Prefill/decode step 中最大 active sequence 数。 |
| `max_decoding_seqs` | Sparse-vLLM | Decode queue 中最大 sequence 数。 |
| `gpu_memory_utilization` | Sparse-vLLM | Cache planning 使用的 GPU 总显存比例。 |
| `tensor_parallel_size` | Sparse-vLLM | TP rank/process 数。 |
| `num_kvcache_slots` | Sparse-vLLM | 可选显式 KV slot override。 |
| `enable_prefix_caching` | Sparse-vLLM | 仅为 vanilla、OmniKV 和 QuEST 启用 prefix KV reuse。 |
| `prefix_cache_block_size` | Sparse-vLLM | Prefix-cache block size；QuEST 必须等于 `quest_chunk_size`。 |
| `prefix_cache_max_blocks` | Sparse-vLLM | Prefix cache 的可选 live-block 上限。 |
| `prefix_cache_salt` | Sparse-vLLM | Cache isolation 使用的附加 fingerprint salt。 |
| `admission_wave_size` | `scripts/benchmarks/bench_sparse_vllm.py` | 仅 benchmark 使用的 staged admission。 |
| `wave_decode_gap_steps` | `scripts/benchmarks/bench_sparse_vllm.py` | 加入下一 wave 前，仅 benchmark 使用的 delay。 |
| `max_decode_steps_after_full` | `scripts/benchmarks/bench_sparse_vllm.py` | Full admission 后，仅 benchmark 使用的 decode window 上限。 |
| `enable_profiler` | Sparse-vLLM | 启用仓库 profiler；`PROFILER_SVLLM` 环境变量也会启用。 |
| `throughput_log_interval_s` | Sparse-vLLM | 周期性 throughput logging interval。 |

### 10.1 为什么必须拆分 `chunk_prefill_size`

HF 语义：

- wrapper 使用它拆分 long prompt forward。
- 很大的值通常表示避免 wrapper chunking。
- 不保留 scheduler-owned KV pool。

Sparse-vLLM 语义：

- 控制 scheduler step size。
- 参与 warmup length。
- 影响 long/short bucket 分类。
- 影响 max batched token 和 cache admission。
- memory heuristic 估计 activation headroom 不足时，可能使 engine assert。

不要跨 backend 复制相同数值。

### 10.2 Benchmark Queueing 与 Wave Admission

`scripts/benchmarks/bench_sparse_vllm.py` 报告：

- TTFT。
- Prefill throughput。
- Decode throughput。
- ITL。
- Average active batch size。
- Peak memory。
- 相同 length 和 batch size 下相对 vanilla 的 speedup。

run 存在 queued request 时，打印的 BS 列带 `*`，表示请求的 batch size 在整个测量期间没有始终完全 active。

Wave admission 用于 prefill eviction 后可以容纳更多 decode request 的方法：

```bash
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 24 \
  --methods snapkv \
  --admission_wave_size 6 \
  --wave_decode_gap_steps 0 \
  --max_decode_steps_after_full 64 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":4096}'
```

公平速度比较应对齐：

- 相同 model 和 tokenizer；
- 相同 prompt length；
- 相同 output length；
- 相同 sampling setting；
- 相同 batch admission policy；
- 相同 actual full-admission status；
- 相同 decode measurement scope。

## 11. `scripts/benchmarks/bench_sparse_vllm.py` 特定说明

应用 `--hyper_params` 前，benchmark 设置稳定 default：

```json
{
  "enforce_eager": true,
  "gpu_memory_utilization": 0.8,
  "engine_prefill_chunk_size": 4096,
  "tensor_parallel_size": 1
}
```

它针对 backend `sparsevllm` 验证规范名称，再将规范 kwarg 传入 `LLM(...)`。

重要行为：

- `method=vanilla` 强制 `sparse_method="vanilla"`。
- 从 `--hyper_params` 取出 `max_model_len`，再覆盖为 `length + output_len + 100`，使 benchmark case 保持一致。
- 未提供时，`max_num_seqs_in_batch` 和 `max_decoding_seqs` 默认为请求 batch size。
- `--hyper_params` 接受 inline JSON 或 `@file.json`。
- 已移除 `--gpu_util`、`--chunk_size`、`--tp` 和 `--enforce_eager` 等旧 helper flag。请使用 `--hyper_params`。

## 12. 模型加载与量化参数

HF `get_generate_api` 通过 `src/deltakv/quantization.py` 路由模型加载设置。

| 参数 | HF 行为 | 通过 `infer_config` 的 Sparse-vLLM 行为 |
| --- | --- | --- |
| `torch_dtype` | 从 runtime config pop，控制模型 dtype。 | 除非之后添加同名 Sparse-vLLM config field，否则忽略。 |
| `load_in_4bit` | 构建 BitsAndBytes 4-bit load config。 | Sparse-vLLM config filtering 忽略。 |
| `load_in_8bit` | 构建 BitsAndBytes 8-bit load config。 | Sparse-vLLM config filtering 忽略。 |
| `quant_skip_modules` / `llm_int8_skip_modules` | 额外排除在 BnB quantization 之外的 module。 | 忽略。 |
| `bnb_4bit_compute_dtype`, `bnb_4bit_use_double_quant`, `bnb_4bit_quant_type`, `bnb_4bit_quant_storage` | BnB 4-bit 选项。 | 忽略。 |
| `llm_int8_threshold`, `llm_int8_enable_fp32_cpu_offload`, `llm_int8_has_fp16_weight` | BnB 8-bit 选项。 | 忽略。 |

默认 skip list 有意包含 `compress_down`、`compress_up`、`k_compress_down` 和 `v_compress_up` 等 compressor module。

不要混淆 base-model quantization 与 KV-cache quantization：

- `load_in_4bit` 在 load 时量化模型权重。
- `deltakv_latent_quant_bits=2` 或 `4` 以方法特定方式量化 cached KV/latent/residual state。

## 13. LLaVA-OneVision Visual-Cache 参数

主要文件：

- `benchmark/multimodal/visual_cache/run_visual_cache.py`
- `src/deltakv/modeling/llava_onevision_deltakv.py`
- `docs/zh/benchmarking/multimodal/README.md`

当前实现的 benchmark 方法：

| Method label | 实际行为 |
| --- | --- |
| `vanilla` | Standard `LlavaOnevisionForConditionalGeneration`。 |
| `deltakv` | 带真实 learned compressor checkpoint 的 LLaVA-OneVision DeltaKV wrapper，使用 checkpoint config 加 CLI keep budget。 |
| `visual_uniform_keep` | 使用 DeltaKV wrapper 基础设施，但没有 compressor、cluster 或 ref token。均匀保留 visual token，丢弃其余部分。 |
| `visual_uniform_keep_int4` | 相同的 uniform visual token keep path，加上 retained visual KV 的 direct int4 storage。 |

重要参数：

| 参数 | 含义 |
| --- | --- |
| `visual_token_prune_only` | 将 cache dropping/pruning 限制在 visual token。 |
| `visual_token_keep_ratio` / CLI `--visual_keep_ratio` | Uniform subsampling 保留 eligible visual token 的比例。 |
| `--quantize_visual_kv` | 在无 checkpoint fallback 中设置 `deltakv_latent_quant_bits=4`。 |
| `--deltakv_center_ratio` | Compressor-backed `deltakv` 的 center/prototype sampling ratio。 |
| `--deltakv_neighbor_count` | Compressor-backed `deltakv` 选择的 ref center 数。 |
| `recent_keep_tokens`, `sink_keep_tokens`, `full_attention_layers` | 传入 text config，影响 cache buffer 行为。 |
| `decode_keep_tokens`, `prefill_keep_tokens` | 为 wrapper compatibility 保留；当前 visual uniform path 不使用 SnapKV-style attention scoring 进行 pruning。 |

当前限制：

- HF cache update 中，`visual_token_prune_only` 当前对大于 1 的 batch size 抛出错误。
- `visual_uniform_keep` 显式设置 `use_compression=False` 和 `use_cluster=False`。
- 无 checkpoint 的 “LLaVA visual keep10” 仍不使用 cluster/ref token，而是 uniform pruning。

## 14. OpenAI-Compatible Serving

OpenAI-compatible online serving 入口：

```bash
sparsevllm-openai-server \
  --model /path/to/local/Qwen2.5-1.5B-Instruct \
  --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

如果 active virtual environment 中的 console script 尚未刷新，等价 module 入口为：

```bash
python -m sparsevllm.entrypoints.openai.api_server \
  --model /path/to/local/Qwen2.5-1.5B-Instruct \
  --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

`--model` 是传给 `sparsevllm.Config.model` 的本地模型目录；`--served-model-name` 是 request JSON 接受的外部 OpenAI API model ID。二者可以不同；request 必须使用 served name。

### 14.1 Serving CLI 参数

serving 入口具有专用 server flag：

| CLI flag | 默认值 | 含义 |
| --- | --- | --- |
| `--model` | 必填 | Sparse-vLLM 加载的本地模型目录。 |
| `--served-model-name` | `--model` 值 | 通过 `/v1/models` 暴露并由 `/v1/completions` 接受的 model ID。 |
| `--host` | `0.0.0.0` | Uvicorn bind host。 |
| `--port` | `8000` | Uvicorn bind port。 |
| `--engine-kwargs` | 未设置 | Sparse-vLLM engine kwarg 的 JSON object 或文件路径。 |
| `--request-log-dir` | 未设置 | Per-request JSON log 的可选目录。 |
| `--response-parser` | 未设置 | 可选 Chat Completions 和 Responses output parser。`qwen3` 和 `minimax_m2` 把模型输出拆为 reasoning、content 和 tool call；加载的 tokenizer 选择匹配 response template。 |

`/v1/models` entry 还会广告 engine 的有效 `max_model_len`。vLLM-compatible client 使用该扩展发现真实 context window，而不是视为未知。smart router 报告服务同一模型的 healthy worker 中最小的 context window。

`/livez` 报告 process liveness。fatal engine-step error 后，`/health` 和 `/readyz` 报告 traffic readiness 并返回 HTTP 503。CLI server 随后以 status 1 退出，使外部 supervisor 能替换 process 和 CUDA context。smart router 在 routing 前探测 worker readiness，移除 failed worker，并在 restarted worker ready 后自动接纳。它有意不 replay interrupted request。per-GPU worker 和 router service template 参见 `deploy/systemd/README.md`。

附加 `--kebab-case` flag 被解析为 Sparse-vLLM engine kwarg。Public runtime control 应使用 `normalize_runtime_params(..., backend="sparsevllm")` 接受的规范 semantic key。也可以传入 `max_model_len`、`max_num_seqs_in_batch`、`gpu_memory_utilization` 和 `throughput_log_interval_s` 等非 legacy `src/sparsevllm/config.py` field。即使 serving parser 能识别 Section 3 中 legacy public name 的拼写，engine 初始化仍会拒绝。`--engine-kwargs` 和显式 CLI engine flag 设置同一 key 时，startup 失败，不会静默选择一个值。

示例：

```bash
sparsevllm-openai-server \
  --model /models/Qwen2.5-1.5B-Instruct \
  --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 32768 \
  --max-num-seqs-in-batch 8 \
  --gpu-memory-utilization 0.9 \
  --sparse-method snapkv \
  --sink-keep-tokens 64 \
  --recent-keep-tokens 512
```

重要 serving default：

| Engine 参数 | Serving 默认值 | 说明 |
| --- | --- | --- |
| `sparse_method` / `vllm_sparse_method` | `""` | 未显式设置时使用 dense/vanilla path。 |
| `tensor_parallel_size` | `1` | Multi-GPU TP 使用 `CUDA_VISIBLE_DEVICES=...` 加 `--tensor-parallel-size N`。 |
| `gpu_memory_utilization` | `0.8` | 继承自 `Config`。 |
| `max_model_len` | `128000` | 继承自 `Config`；prompt length 加 `max_tokens` 必须能容纳。 |
| `max_num_batched_tokens` | `65536` | 继承自 `Config`。 |
| `max_num_seqs_in_batch` | `32` | 继承自 `Config`。 |
| `max_decoding_seqs` | `64` | 继承自 `Config`。 |
| `engine_prefill_chunk_size` / `chunk_prefill_size` | `8192` | 仅用于 `all_chunked`。CLI 使用语义明确的 `--engine-prefill-chunk-size`。 |
| `long_prefill_offload_threshold` | `98304` | 仅用于 `long_bs1full_short_batch`，同时决定该 policy 的 chunk size。 |
| `enable_prefix_caching` | `false` | 传入 `--enable-prefix-caching true` 启用 prefix KV reuse。 |
| `prefix_cache_block_size` | vanilla/OmniKV 为 `16`，QuEST 为 `quest_chunk_size` | 使用 `--prefix-cache-block-size`；QuEST 拒绝与 `quest_chunk_size` 不同的值。 |
| `prefix_cache_max_blocks` | 未设置 | 可选 cache capacity 上限。 |
| `prefix_cache_salt` | `""` | 可选 cache fingerprint isolation salt。 |
| `throughput_log_interval_s` | serving 中为 `0.0` | server 默认关闭周期性 `Avg TP` log，改为 per-request logging。传入 `--throughput-log-interval-s 10` 可重新启用周期性 throughput log。 |

OpenAI server 当前有意不暴露 DeltaKV-family sparse method。规范化 method 以 `deltakv` 开头时，server startup 会快速失败。在 serving correctness 和 memory 行为得到验证前，使用离线实验入口运行这些方法。

### 14.2 `/v1/completions` Request 参数

已实现的 endpoint 是 OpenAI-style text completions：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
  }'
```

Streaming 使用 SSE：

```bash
sparsevllm-openai-client \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "San Francisco is a" \
  --max-tokens 7 \
  --temperature 0
```

raw HTTP stream 保持标准 SSE（`data: {...}` frame，以 `data: [DONE]` 结束）。helper client 解析 frame，只打印 incremental text。

Online serving 要求 Hugging Face fast tokenizer backend 支持 `DecodeStream`。Sparse-vLLM 为每个 request 保留独立的 visible 和 raw incremental decoder，因此切分 multi-byte Unicode character 的 byte-level token 会先 buffer，直到字符完整。这统一适用于 Completions、Chat Completions 和 Responses streaming；拼接后的 text delta 与对应 non-streaming final text 一致。不支持的 slow tokenizer 会显式失败，不会 fallback 到不安全的 per-token decoding 或 replacement-character filtering。

同时提供 chat completions：

```bash
sparsevllm-openai-client \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --chat \
  --prompt "Explain Sparse-vLLM in one sentence."
```

支持的 JSON field：

| Field | 默认值 | 含义 |
| --- | --- | --- |
| `model` | 必填 | 提供 `--served-model-name` 时必须与其相等，否则必须等于 `--model` 值。 |
| `prompt` | 必填 | 字符串、token ID 列表、字符串列表或 token ID 列表的列表。 |
| `max_tokens` | `16` | 映射到 `SamplingParams.max_tokens`，必须为正数。 |
| `max_completion_tokens` | `null` | 仅 Chat 使用的 OpenAI-compatible `max_tokens` alias；显式同时设置且值不同时快速失败。 |
| `temperature` | `1.0` | 映射到 `SamplingParams.temperature`；`0` 表示 greedy sampling。 |
| `top_p` | `1.0` | 映射到 `SamplingParams.top_p`，必须在 `(0, 1]`。 |
| `top_k` | `0` | 映射到 `SamplingParams.top_k`；`0` 禁用 top-k filtering。 |
| `n` | `1` | 当前只支持 `1`。 |
| `stream` | `false` | `true` 返回 `text/event-stream` chunk，以 `data: [DONE]` 结束。 |
| `ignore_eos` | `false` | 即使生成 EOS，仍继续到 `max_tokens`。 |
| `stop` | `null` | 字符串或字符串列表；返回 completion 中不包含 stop text。 |
| `logprobs` | `null` | `/v1/completions` 支持不大于 5 的非负整数；返回 sampled-token logprob 和最多指定数量的 top logprob。 |

Unknown JSON field 会被拒绝，而不是静默忽略。这比部分 OpenAI-compatible server 更严格，但可以避免接受不影响研究结果的参数。

`stop` 和 `logprobs` 可分别使用。同时设置会快速失败，因为 text-level stop trimming 可能使返回 token logprob 与 visible output 不一致。

`/v1/chat/completions` 支持相同 sampling field，另加 `messages`。message role 必须是 `developer`、`system`、`user`、`assistant` 或 `tool`。支持 string content 和 text-only content-part list；拒绝 unknown nested message field。assistant message 可包含兼容扩展 `reasoning_content` 和 OpenAI function `tool_calls`；tool result message 必须使用 `tool` role 和匹配的 `tool_call_id`。这些 field 传给 Hugging Face chat template，使历史 reasoning、call 和 result 能 round-trip 到下一个 prompt。Role-specific field 和 malformed function call object 会 validation fail，不会被忽略。由于大多数本地 tokenizer template 不定义独立 developer role，`developer` 会渲染为 `system`。tokenizer 提供 chat template 时，server 使用 `apply_chat_template(..., add_generation_prompt=True)`；否则使用简单 role-prefixed prompt。

Chat request 可将 `reasoning_effort` 设为 `none`、`minimal`、`low`、`medium`、`high` 或 `xhigh`；`none` 映射到 `enable_thinking=false`，其他值映射到 `true`。直接 `enable_thinking` field 和 `"chat_template_kwargs": {"enable_thinking": false}` 仍供 Qwen3-style template 使用。遵循 vLLM Chat API contract，`chat_template_kwargs` 是开放 JSON object，值直接传给 tokenizer template。兼容的 top-level `preserve_thinking` field 会规范到 `chat_template_kwargs.preserve_thinking`，使 Qwen-family client 在 template 支持时 replay 历史 reasoning。相同值的重复 control 可以接受；冲突值、已知 thinking control 的非 boolean 值，或 tokenizer 无 chat template 时提供 template kwarg，都会快速失败。

Chat function tool 接受 OpenAI nested function schema 和兼容 flat Responses form。effective tool 通过 tokenizer 的 `tools` kwarg 传入。`tool_choice` 支持 `null`、`"auto"` 和 `"none"`；`none` 会从 generation prompt 移除 tool。named/required choice 和 `parallel_tool_calls=false` 会显式失败，因为尚未实现对应 generation constraint。只有 tool 生效时，server 才解析 Qwen-style `<tool_call>`/`<tool_calls>` 和 MiniMax M2 `<minimax:tool_call><invoke ...>` output；server 本身不执行 tool。thinking 与 tool calling 同时启用时，应启用匹配的 `--response-parser`；未启用时，reasoning-only output 保持 raw `content`。

使用 `--response-parser qwen3` 或 `--response-parser minimax_m2` 时，non-streaming Chat response 把本地 raw reasoning 拆到 Sparse-vLLM `message.reasoning_content` 扩展，把 visible answer 放入 `message.content`。function call 使用 OpenAI `message.tool_calls` 和 `finish_reason="tool_calls"`。Streaming 使用 `delta.reasoning_content` 传递本地 raw reasoning，使用标准 indexed `delta.tool_calls` chunk 传递 function name 和 argument delta。state machine 解析跨 chunk reasoning tag 和 tool JSON；malformed 或 unclosed output 会显式报告。未启用 reasoning parser 时，raw reasoning text 保留在 `content`，保持旧行为。

Chat `logprobs=true` 启用 sampled-token logprob，`top_logprobs` 控制最多 20 个 top alternative。reasoning 或 tool output parsing 激活时拒绝 logprob，因为无法相对拆分/隐藏的 Chat field 真实表示 raw generated token position。`/v1/completions` 仍是 raw prompt endpoint，不增加 server-side thinking switch；需要时 client 可自行在 prompt 中加入 `/think` 或 `/no_think` 等 marker。

`/v1/responses` 是独立 item-based input/output endpoint。首个实现支持 text input、text-only message item、`function_call_output` input item、function tool schema、`reasoning.effort`、non-streaming response 和 Responses SSE streaming。`max_output_tokens` 映射到 `SamplingParams.max_tokens`；`temperature`、`top_p` 和 `top_k` 直接映射到 sampling param。`tool_choice` 仅支持 `null` 或 `"auto"`；`parallel_tool_calls=false` 和 `reasoning.summary` 在语义实现前显式失败。`stream=true` 返回 Responses semantic SSE event，而不是 Chat Completions chunk。

为兼容 client，可以接受 `store=false`（或省略）和非空 `prompt_cache_key`。Sparse-vLLM 不持久化 response object，因此 `store=true` 显式失败。`prompt_cache_key` 作为 cache-grouping hint 保留在 request log 中，但不改变渲染后的模型 prompt，也不替代 Sparse-vLLM exact-prefix cache matching。

启用 `--response-parser qwen3` 或 `--response-parser minimax_m2` 时，`/v1/responses` 把以 `<think>` 开头的模型输出解析为 Sparse-vLLM extension reasoning item，随后是 assistant message 或 function call item。该扩展为复现暴露本地模型 reasoning text，不声称等价于 OpenAI-hosted reasoning token（后者不以 raw text 暴露）。未启用 parser 时，generated text 原样返回为 `output_text`。`reasoning.effort="none"` 映射到 `enable_thinking=false`，其他 effort 值映射到 `enable_thinking=true`。与显式 `chat_template_kwargs.enable_thinking` 冲突时快速失败。streaming 模式中，`response.reasoning_text.delta` 是本地 raw reasoning text 的 Sparse-vLLM extension event，不是 OpenAI-hosted raw reasoning token field。

tokenizer chat template 支持时，function tool 通过 `tools` kwarg 传入。server 规范 OpenAI function tool schema，适配为加载的 tokenizer 所需 Qwen 或 MiniMax template shape，并把显式 Qwen `<tool_call>...</tool_call>` 或 MiniMax `<minimax:tool_call><invoke ...>` output 解析为 Responses `function_call` item。server 不执行 tool；application 必须执行，再把 result 作为 `function_call_output` input item 发回。Streaming tool call 发送 `function_call` output item，以及 `response.function_call_arguments.delta` 和 `response.function_call_arguments.done` event。

Prefix-cache matching 接受完整 `chat` 和 `response` selector。worker 使用真实 generation 相同的 endpoint prompt helper 渲染，因此 message、instruction、tool、reasoning control 和 chat template kwarg 一致参与 cache-match key。smart router 使用这些完整 selector，而不是仅根据 message 近似 Chat request。

server 对每个 `/v1/completions` request 记录一条 request-start line，以及一条 request-finish 或 request-cancel line；不会记录每个 generated token。

## 15. Benchmark 入口

### 15.1 LongBench 与 MathBench

`benchmark/long_bench/pred.py` 和 `benchmark/math_bench/pred.py`：

- 从 `infer_config={"max_model_len": args.max_model_len}` 开始。
- Merge `--hyper_param`（JSON 文件或 inline JSON）。
- 调用 `get_generate_api(...)`。
- 传入 `temperature`、`top_p` 和 `top_k` 等 generation 参数。

`get_generate_api` 返回后的 backend 差异：

| Generation kwarg | HF path | Sparse-vLLM wrapper path |
| --- | --- | --- |
| `max_new_tokens` | 使用。 | 映射到 `SamplingParams.max_tokens`。 |
| `max_tokens` | 非主要参数。 | `max_new_tokens` 不存在时作为 fallback。 |
| `do_sample` | 使用。 | 只决定 temperature 是否变为 `0.0`。 |
| `temperature` | 使用。 | 使用；greedy 设为 `0.0`，很小 sampling 值 clamp 到 `1e-5`。 |
| `top_p` | HF generation 使用。 | 当前 wrapper 不转发。 |
| `top_k` | HF generation 使用。 | 当前 wrapper 不转发。 |
| `eos_token_id` | HF generation 使用。 | 当前 wrapper 不转发。 |
| `past_key_values` | Manual HF path 可以使用。 | 为 signature compatibility 接受，但忽略。 |

### 15.2 NIAH

`benchmark/niah/test_niah.py` 根据 function argument 手动构建 `infer_config`。现在使用 `sparse_method`、`hf_prefill_chunk_size`、`engine_prefill_chunk_size`、`gpu_memory_utilization` 和 `use_cluster` 等规范 key。

这便于快速实验，但也增加某个 backend 忽略另一个 backend key 的可能。新增 NIAH run 时使用规范名称。

### 15.3 SCBench

`benchmark/scbench/run_scbench.py` 的 DeltaKV branch 支持：

```text
deltakv, full_deltakv, origin_residual_quant, all_origin_residual_quant,
snapkv, pyramidkv, palu, quest
```

它复制 `hyper_param`，提取 `deltakv_checkpoint_path`，pop `sparse_method` 和 `cuda_device`，再调用 `get_generate_api(..., return_model=True)`。该 path 面向 HF；Sparse-vLLM-style engine 参数通常不适用。

## 16. 已知歧义与推荐名称

| 已移除 legacy name | 问题 | 规范名称 |
| --- | --- | --- |
| `seq_chunk_size` | 部分 path 表示 token grouping，cluster path 中曾 fallback 为 `k_neighbors`。 | grouping 使用 `compressor_token_group_size`；neighbor 使用 `deltakv_neighbor_count`。 |
| `chunk_prefill_size` | 同一名称表示 HF wrapper chunking 或 Sparse-vLLM scheduler chunking。 | `hf_prefill_chunk_size` 或 `engine_prefill_chunk_size`。 |
| `num_top_tokens` | 随 HF path 可能是 count、ratio 或 per-layer list；已从 Sparse-vLLM config 移除。 | `decode_keep_tokens`，并显式记录 ratio conversion。 |
| `num_top_tokens_in_prefill` | 已移除的 Sparse-vLLM prefill budget；仍是 HF internal field。 | 仅 HF 使用 `prefill_keep_tokens`；Sparse-vLLM 使用 `decode_keep_tokens`。 |
| `compressor_path` | HF top-level path；未规范化时 Sparse-vLLM 忽略。 | `deltakv_checkpoint_path`。 |
| `deltakv_path` | Sparse-vLLM path；未规范化时 HF 忽略。 | `deltakv_checkpoint_path`。 |
| `model_cls` | 仅 HF 的 method selector。 | 使用 `sparse_method` 表达 portable intent。 |
| `vllm_sparse_method` | 仅 Sparse-vLLM 的 method selector。 | 使用 `sparse_method` 表达 portable intent。 |
| `tail_token_size` | 历史 HF recent-buffer name。 | `recent_keep_tokens`。 |
| `kv_quant_bits` | 不同 path 量化不同对象。 | `deltakv_latent_quant_bits`，并记录量化对象。 |
| `deltakv_visual_compress_only` | 即使 uniform pruning，名称也写 DeltaKV/compress。 | `visual_token_prune_only`。 |
| `deltakv_visual_keep_ratio` | 与误导性的 visual compress name 绑定。 | `visual_token_keep_ratio`。 |

## 17. Alignment Workflow

### 17.1 准确率 Alignment

比较准确率前，对齐：

- model path；
- tokenizer path；
- backend；
- method family；
- checkpoint path；
- keep budget；
- sink/recent budget；
- full/observation layer；
- cluster/ref setting；
- latent width 和 quantization mode；
- sampling 参数；
- prompt formatting 和 truncation。

确定性检查尽可能使用 greedy decode。

HF 示例：

```bash
python benchmark/long_bench/pred.py \
  --model_path <MODEL> \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path <COMPRESSOR> \
  --hyper_param '{"decode_keep_tokens":0.17,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18","hf_prefill_chunk_size":32768}'
```

使用显式 count 的 Sparse-vLLM 示例：

```bash
python benchmark/long_bench/pred.py \
  --model_path <MODEL> \
  --backend sparsevllm \
  --hyper_param '{"sparse_method":"deltakv","deltakv_checkpoint_path":"<COMPRESSOR>","decode_keep_tokens":22282,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18","engine_prefill_chunk_size":512,"gpu_memory_utilization":0.9}'
```

### 17.2 速度 Alignment

比较速度前，区分三类设置：

| 类别 | 示例 | 是否应对齐 |
| --- | --- | --- |
| Sparse policy | `decode_keep_tokens`, `recent_keep_tokens`, `full_attention_layers`, `deltakv_center_ratio` | 方法 parity 需要。 |
| Engine capacity | `gpu_memory_utilization`, `engine_prefill_chunk_size`, `max_num_batched_tokens` | 测量相同 engine regime 时对齐；测量最大 capacity 时调优。 |
| Benchmark policy | `batch_sizes`, `admission_wave_size`, `max_decode_steps_after_full` | 必须随结果报告。 |

Full-attention baseline 示例：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src:$PYTHONPATH \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 6 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization":0.95,"engine_prefill_chunk_size":4096}'
```

Sparse wave-admission 示例：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src:$PYTHONPATH \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 24 \
  --methods snapkv \
  --admission_wave_size 6 \
  --wave_decode_gap_steps 0 \
  --max_decode_steps_after_full 64 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":4096}'
```

## 18. 环境变量

环境变量不由 `normalize_runtime_params` 规范化。应将其视为 process-level switch，并始终随 benchmark result 记录。

| 变量 | 范围 | 作用 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | 所有 GPU run | 选择 physical GPU；multi-worker script 把 local rank 映射到此列表。 |
| `PYTHONPATH` | 所有仓库 run | 从 source 运行时应包含 `$PWD/src`。 |
| `LOG_LEVEL` | 仓库 logging | 控制 `deltakv` 和 `sparsevllm` logger verbosity。 |
| `PROFILER_SVLLM` | Sparse-vLLM | 启用仓库 profiler，等价于 `enable_profiler=True`。 |
| `CUDA_SYNC_SVLLM` | Sparse-vLLM profiler | 在 profiler timing point 周围同步 CUDA，会增加开销，仅用于 profiling。 |
| `SPARSEVLLM_MASTER_PORT` | Sparse-vLLM TP | Spawned TP worker 的 master port。 |
| `SPARSEVLLM_DEBUG_SLOTS` | Sparse-vLLM scheduler/cache | 打印额外 cache-slot admission/debug 信息。 |
| `SPARSEVLLM_DELTAKV_L2_BLOCK_N/M/D/NUM_WARPS` | Sparse-vLLM DeltaKV kernel | 覆盖 DeltaKV L2 selection 的 Triton block/warp tuning。 |
| `SPARSEVLLM_DELTAKV_STANDALONE_TEMP_SLOTS` | DeltaKV standalone/snapkv | 覆盖临时 reconstruction slot reservation。 |
| `SPARSEVLLM_DELTAKV_STANDALONE_DECOMPRESS_CHUNK_TOKENS` | DeltaKV standalone/snapkv | 覆盖 reconstruction chunking size。 |
| `OMNIKV_ASSERT` | OmniKV fused kernel | 启用额外 assertion。 |
| `USE_ADVSEL` | Sparse-vLLM DeltaKV | 启用 experimental advanced selection path。 |
| `MANUAL_GEN_CHUNK_PREFILL_SIZE` | HF generation wrapper | 在选定 HF path 强制 manual prompt chunking。 |
| `BAN_EOS` | HF generation wrapper | 在已实现位置屏蔽 generation 的 EOS token。 |
| `NOT_SKIP_SPECIAL_TOKENS` | HF generation decode | 在 decoded output 中保留 special token。 |
| `ENABLE_HF_GEN` | HF generation path | 在 `get_generate_api` 中强制模型 `.generate(...)` path。 |
| `KVZIP_DEBUG` | KVzip adapter | 打印 KVzip debug memory/cache 信息。 |
| `DEBUG` | benchmark/HF path | 在多个脚本中启用额外 prompt/cache debug print。 |
| `DELTAKV_OUTPUT_DIR` | LongBench/MathBench/SCBench | Benchmark prediction/log 的 output root。 |
| `DELTAKV_DATA_DIR` | LongBench/MathBench | Dataset root fallback。 |
| `DELTAKV_LONGBENCH_DATA_DIR` | LongBench | LongBench-specific dataset root override。 |
| `DELTAKV_OUTPUT_BASE` | NIAH | NIAH output root。 |
| `ENABLE_THINKING` | MathBench | 控制 tokenizer chat template thinking mode。 |
| `LOCAL_RANK` | compressor 训练 | Distributed training local rank override。 |
| `FORCE_QWEN` | compressor 训练 | 在 `train_compressor.py` 中强制 Qwen2 model branch。 |
| `ANALYSIS` | compressor training/analysis | 在 cluster training code 中启用额外 analysis output。 |
| `MSE_DETACH`, `NTP_DETACH` | compressor 训练 | Loss-gradient detaching 的 ablation switch。 |
| `REMOVE_COMP`, `REMOVE_REF` | HF cache ablation | 在选定 cache path 移除 compressor 或 reference component。 |
| `COPY_ON_GPU` | SCBench DeltaKV wrapper | 在 `DeltaKVGreedySearch` 中把 copy 保留在 GPU。 |

Profiling 示例：

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=$PWD/src:$PYTHONPATH \
LOG_LEVEL=DEBUG \
PROFILER_SVLLM=1 \
CUDA_SYNC_SVLLM=1 \
python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 6 \
  --methods deltakv \
  --hyper_params '{"sparse_method":"deltakv","deltakv_checkpoint_path":"<COMPRESSOR>","engine_prefill_chunk_size":4096,"decode_keep_tokens":4096,"gpu_memory_utilization":0.9}'
```

Benchmark output/data 示例：

```bash
DELTAKV_OUTPUT_DIR=<OUTPUT_ROOT> \
DELTAKV_DATA_DIR=<DATA_ROOT> \
PYTHONPATH=$PWD/src:$PYTHONPATH \
python benchmark/long_bench/pred.py \
  --model qwen25-deltakv \
  --model_path <MODEL> \
  --backend hf \
  --sparse_method deltakv \
  --deltakv_checkpoint_path <COMPRESSOR> \
  --hyper_param '{"hf_prefill_chunk_size":32768,"decode_keep_tokens":0.17,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18"}'
```

## 19. 安全 Config Checklist

启动 run 前：

- Shared config 使用 `sparse_method` 和 `deltakv_checkpoint_path`。
- 使用 `hf_prefill_chunk_size` 或 `engine_prefill_chunk_size`；新 shared config 不要使用裸 `chunk_prefill_size`。
- `use_cluster=True` 时显式设置 `deltakv_neighbor_count`。
- 使用 Sparse-vLLM 时，把所有 ratio budget 转换为 token count。
- 使用 prefix cache 时，`sparse_method` 必须是 `vanilla`、`omnikv` 或 `quest`；generated decode input token 在形成完整 prefix-cache block 后默认进入 cache；使用 `decode_cuda_graph` 时保持 `decode_cuda_graph_capture_sampling=false`。
- QuEST prefix cache 应让 `prefix_cache_block_size` 等于 `quest_chunk_size`，或省略前者。
- 使用 LLaVA 无 checkpoint path 时，将其标为 visual uniform pruning，不要标为 DeltaKV compressor inference。
- 记录 `deltakv_latent_quant_bits=2` 或 `4` 量化的是 latent state、raw KV 还是 residual。
- 检查 log 中被忽略的 unknown key。
- 对 throughput，记录 run 是否 queue，以及 decode throughput 来自 full admission 还是 fallback scope。
