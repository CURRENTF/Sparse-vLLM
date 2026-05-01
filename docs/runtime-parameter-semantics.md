# Runtime Parameter Semantics

This document defines the canonical runtime parameter names used to align
DeltaKV/HF inference and Sparse-vLLM inference.

The goal is not to remove legacy keys immediately. The goal is to make the
accuracy-affecting parameters and speed/capacity parameters explicit, so the same
configuration does not silently mean different things on different backends.

## Summary

Use the canonical names below in new configs:

```json
{
  "sparse_method": "deltakv",
  "deltakv_checkpoint_path": "/path/to/compressor",
  "decode_keep_tokens": 2048,
  "prefill_keep_tokens": 4096,
  "sink_keep_tokens": 8,
  "recent_keep_tokens": 128,
  "full_attention_layers": "0,1,2,8,18",
  "deltakv_neighbor_count": 4,
  "deltakv_center_ratio": 0.1,
  "deltakv_latent_dim": 256
}
```

Then add backend-specific speed/capacity knobs:

```json
{
  "engine_prefill_chunk_size": 512,
  "gpu_memory_utilization": 0.9,
  "max_num_batched_tokens": 8192,
  "max_num_seqs_in_batch": 8,
  "max_decoding_seqs": 8
}
```

or, for HF DeltaKV:

```json
{
  "hf_prefill_chunk_size": 32768
}
```

## What Changed

The repo now normalizes canonical aliases before constructing either runtime:

- Sparse-vLLM: `LLMEngine.__init__` normalizes kwargs before filtering them into
  `sparsevllm.Config`.
- HF DeltaKV: `CustomConfigMixin.set_extra_args()` normalizes kwargs before
  applying them to the HF config object.
- `get_generate_api()` normalizes `infer_config`, `model_cls`, and
  `compressor_path` before choosing the backend.
- `scripts/bench_sparse_vllm.py` normalizes `--hyper_params` before launching
  the Sparse-vLLM engine.

Legacy keys still work. If a canonical key and its legacy target are both
provided with different values, normalization raises a `ValueError`.

Sparse-vLLM now rejects ratio-style keep budgets such as
`"decode_keep_tokens": 0.17` or `"num_top_tokens": 0.17`. Sparse-vLLM requires
explicit token counts. HF DeltaKV still supports ratio-style budgets where the
HF token selection code supports them.

## Accuracy-Affecting Parameters

These parameters change what tokens are visible or reconstructed. They must be
matched before comparing accuracy.

| Canonical key | Sparse-vLLM legacy key | HF DeltaKV legacy key | Actual function |
| --- | --- | --- | --- |
| `sparse_method` | `vllm_sparse_method` | `model_cls` | Selects the sparse method/runtime path. |
| `deltakv_checkpoint_path` | `deltakv_path` | top-level `compressor_path` | Loads the DeltaKV compressor weights/config. |
| `decode_keep_tokens` | `num_top_tokens` | `num_top_tokens` | Decode-time top/important token budget. |
| `prefill_keep_tokens` | `num_top_tokens_in_prefill` | `num_top_tokens_in_prefill` | Prefill/finalization top/important token budget. |
| `sink_keep_tokens` | `num_sink_tokens` | `num_sink_tokens` | Always-visible prefix/sink tokens. |
| `recent_keep_tokens` | `num_recent_tokens` | `num_recent_tokens` / `tail_token_size` | Always-visible recent tail tokens. |
| `full_attention_layers` | `full_attn_layers` | `full_attn_layers` | Layers that stay dense/full, or observation anchors for mixed methods. |
| `observation_layers` | `obs_layer_ids` | not supported | Explicit Sparse-vLLM observation layers. |
| `deltakv_neighbor_count` | `deltakv_k_neighbors` | `k_neighbors` | Number of reference/neighbor full-KV slots per latent token. |
| `deltakv_center_ratio` | `cluster_ratio` | `cluster_ratio` | Fraction/stride controlling retained center/reference tokens. |
| `deltakv_latent_dim` | `kv_compressed_size` | `kv_compressed_size` | Compressed latent KV width. |
| `deltakv_latent_quant_bits` | `kv_quant_bits` | `kv_quant_bits` | Quantization bits for DeltaKV-style cached state. |

### Multimodal Visual-Token Pruning Names

For LLaVA-OneVision experiments, use the explicit names below:

| Canonical key | Legacy key | Actual function |
| --- | --- | --- |
| `visual_token_prune_only` | `deltakv_visual_compress_only` | Restrict cache dropping/pruning to visual tokens. This does not imply learned DeltaKV compression. |
| `visual_token_keep_ratio` | `deltakv_visual_keep_ratio` | Fraction of eligible visual KV tokens kept by the current visual-token pruning path. |

The no-checkpoint LLaVA path with `visual_token_prune_only=true` and
`use_compression=false` is a visual-token uniform-pruning baseline, not DeltaKV
cluster/compressor inference.

### Keep Budget Rule

HF DeltaKV selection supports both integer counts and ratio-style floats in some
paths:

```json
{"decode_keep_tokens": 0.17}
```

Sparse-vLLM uses engine/cache planning and must receive explicit token counts:

```json
{"decode_keep_tokens": 22282}
```

For a 131072-token prompt, a 17% budget is approximately:

```text
int(131072 * 0.17) = 22282
```

This conversion should be done deliberately and recorded in the run config.

## Speed And Capacity Parameters

These parameters should be matched when measuring speed, but they do not define
the sparse selection policy itself.

| Canonical / current key | Backend | Actual function |
| --- | --- | --- |
| `engine_prefill_chunk_size` | Sparse-vLLM | Engine scheduler prefill chunk size. Maps to legacy `chunk_prefill_size`. |
| `hf_prefill_chunk_size` | HF DeltaKV | Model/wrapper prefill chunk size. Maps to legacy `chunk_prefill_size`. |
| `gpu_memory_utilization` | Sparse-vLLM | Fraction of GPU memory used for KV/cache planning. |
| `max_num_batched_tokens` | Sparse-vLLM | Scheduler cap for tokens in one prefill step. |
| `max_num_seqs_in_batch` | Sparse-vLLM | Scheduler cap for total active sequences in a step. |
| `max_decoding_seqs` | Sparse-vLLM | Scheduler cap for concurrent decode sequences. |
| `tensor_parallel_size` | Sparse-vLLM | Number of tensor-parallel worker ranks. |
| `admission_wave_size` | benchmark script | Adds requests in waves; useful when sparse methods can host more decode requests after prefill eviction. |
| `wave_decode_gap_steps` | benchmark script | Optional decode-step gap before admitting the next wave. |
| `max_decode_steps_after_full` | benchmark script | Optional cap for the full-admission decode measurement window. |

### Why `chunk_prefill_size` Is Not A Portable Name

The legacy key `chunk_prefill_size` exists on both sides but does not mean the
same operational thing.

- HF DeltaKV: mostly model/wrapper-side manual chunking.
- Sparse-vLLM: engine scheduling, memory planning, and prefill admission.

New configs should use:

- `hf_prefill_chunk_size` for HF DeltaKV.
- `engine_prefill_chunk_size` for Sparse-vLLM.

The normalizer maps both to the backend-native `chunk_prefill_size` only after
the backend is known.

## Method And Checkpoint Routing

### Sparse-vLLM

Canonical:

```json
{
  "sparse_method": "deltakv-triton-v4",
  "deltakv_checkpoint_path": "/path/to/compressor"
}
```

Normalized Sparse-vLLM config:

```json
{
  "vllm_sparse_method": "deltakv-triton-v4",
  "deltakv_path": "/path/to/compressor"
}
```

### HF DeltaKV

Canonical:

```json
{
  "sparse_method": "deltakv",
  "deltakv_checkpoint_path": "/path/to/compressor"
}
```

Normalized HF routing:

```json
{
  "model_cls": "deltakv",
  "compressor_path": "/path/to/compressor"
}
```

The HF backend cannot run every Sparse-vLLM method string directly. The
normalizer maps common Sparse-vLLM DeltaKV variants such as
`deltakv-triton-v4` to HF `model_cls="deltakv"` for API compatibility, but the
implementation details are still backend-specific.

## Alignment Workflow

### 1. Normalize The Same Config On Both Backends

Use the same canonical JSON as the source of truth. Then inspect the normalized
result for each backend:

```python
from deltakv.configs.runtime_params import normalize_runtime_params

cfg = {
    "sparse_method": "deltakv",
    "deltakv_checkpoint_path": "/path/to/compressor",
    "decode_keep_tokens": 2048,
    "prefill_keep_tokens": 4096,
    "sink_keep_tokens": 8,
    "recent_keep_tokens": 128,
    "full_attention_layers": "0,1,2,8,18",
    "deltakv_center_ratio": 0.1,
    "deltakv_neighbor_count": 4
}

print(normalize_runtime_params(cfg, backend="hf"))
print(normalize_runtime_params(cfg, backend="sparsevllm"))
```

If the helper raises, fix the config before running evaluation.

### 2. Accuracy Alignment

Accuracy comparisons should match:

- base model path,
- tokenizer path,
- compressor checkpoint,
- sparse method family,
- keep budgets,
- sink/recent budgets,
- full/observation layers,
- DeltaKV latent width and neighbor count,
- sampling settings (`temperature=0` for deterministic checks).

For backend-to-backend comparisons, run the same task set twice, once with
`--backend hf` and once with `--backend sparsevllm`, using the same canonical
accuracy config. Convert ratio budgets to explicit token counts before the
Sparse-vLLM run.

Example HF-style LongBench run:

```bash
python benchmark/long_bench/pred.py \
  --model_path <MODEL> \
  --backend hf \
  --model_cls deltakv \
  --compressor_path <COMPRESSOR> \
  --hyper_param '{"decode_keep_tokens":0.17,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18","hf_prefill_chunk_size":32768}'
```

Example Sparse-vLLM run with explicit token count:

```bash
python benchmark/long_bench/pred.py \
  --model_path <MODEL> \
  --backend sparsevllm \
  --hyper_param '{"sparse_method":"deltakv-triton-v4","deltakv_checkpoint_path":"<COMPRESSOR>","decode_keep_tokens":22282,"prefill_keep_tokens":4096,"sink_keep_tokens":8,"recent_keep_tokens":128,"full_attention_layers":"0,1,2,8,18","engine_prefill_chunk_size":512,"gpu_memory_utilization":0.9}'
```

### 3. Speed Alignment

Speed comparisons should separate:

- sparse policy parameters, which affect visible/reconstructed tokens,
- engine capacity parameters, which affect admission, batching, and OOM behavior,
- benchmark admission policy, such as wave admission.

For Sparse-vLLM throughput, prefer `scripts/bench_sparse_vllm.py` because it
reports TTFT, prefill throughput, decode throughput, ITL, average active batch,
and memory.

Full-attention capacity baseline:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src:$PYTHONPATH \
python scripts/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 6 \
  --methods vanilla \
  --hyper_params '{"gpu_memory_utilization":0.95,"engine_prefill_chunk_size":4096}'
```

Sparse wave-admission throughput:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src:$PYTHONPATH \
python scripts/bench_sparse_vllm.py \
  --model_path <MODEL> \
  --lengths 131072 \
  --batch_sizes 24 \
  --methods snapkv \
  --admission_wave_size 6 \
  --wave_decode_gap_steps 0 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":4096,"sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":4096,"prefill_keep_tokens":4096}'
```

The benchmark marks queued full-attention runs with `*` in the `BS` column.
For sparse methods, `AdmissionWave`, `FullAdmit`, and `DecodeScope` tell you
whether the reported decode throughput came from the full-admission window.

## Legacy Compatibility

The following legacy keys are still accepted:

- `vllm_sparse_method`
- `deltakv_path`
- `compressor_path`
- `model_cls`
- `num_top_tokens`
- `num_top_tokens_in_prefill`
- `num_sink_tokens`
- `num_recent_tokens`
- `tail_token_size`
- `full_attn_layers`
- `obs_layer_ids`
- `deltakv_k_neighbors`
- `k_neighbors`
- `cluster_ratio`
- `kv_compressed_size`
- `kv_quant_bits`
- `chunk_prefill_size`

Prefer the canonical keys in new configs. Use legacy keys only when reproducing
old runs exactly.
