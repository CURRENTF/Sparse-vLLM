# Runtime Parameter Semantics

Sparse-vLLM has one inference backend: the native engine under
`src/sparsevllm/`. Runtime parameter names are identical in `LLM(...)`,
`Config`, JSON configs, benchmark manifests, and internal code. The engine does
not maintain a public-to-internal alias layer.

## Public names

Use semantic public names in commands, JSON configs, and benchmark manifests:

| Canonical name | Meaning |
| --- | --- |
| `sparse_method` | Sparse method selector. |
| `prefill_sparse_method` | Orthogonal prefill-attention algorithm selector. It does not replace the cache/decode `sparse_method`. |
| `deltakv_checkpoint_path` | DeltaKV compressor checkpoint path. |
| `engine_prefill_chunk_size` | Maximum scheduled prefill chunk. |
| `sink_keep_tokens` | Fixed sink-token budget. |
| `recent_keep_tokens` | Recent-token budget. |
| `full_attention_layers` | `auto` (default), a comma-separated string, or a list of full-layer indices. `auto` resolves an exact method/model-name profile; catalog entries can be shared by OmniKV and DeltaKV or scoped to one method. |
| `deltakv_neighbor_count` | Number of DeltaKV reference neighbors. |
| `deltakv_center_ratio` | DeltaKV reference-center ratio. |
| `deltakv_latent_dim` | Compressor latent width. |
| `deltakv_latent_quant_bits` | Quantization bits for latent state. |
| `deltakv_latent_quant_group_size` | Latent quantization group size. |
| `gpu_memory_utilization` | Fraction of GPU memory available to the engine. |
| `decode_graph` | Enable decode CUDA Graph execution. |

The `auto` full-layer matcher compares the final model path/repository segment
case-insensitively and also recognizes Hugging Face cache paths such as
`models--org--model/snapshots/...`. It never uses substring matching. An
unregistered OmniKV or DeltaKV model fails with an explicit calibration error.
When a catalog entry lists both methods, they intentionally share the same
full-layer anchors; an explicit layer list still overrides `auto`.

Legacy aliases such as `sparse_method`, `deltakv_checkpoint_path`,
`engine_prefill_chunk_size`, `sink_keep_tokens`, `recent_keep_tokens`,
`full_attention_layers`, `deltakv_neighbor_count`, `deltakv_center_ratio`,
`deltakv_latent_dim`, `deltakv_latent_quant_bits`, `deltakv_latent_quant_group_size`,
`device_memory_utilization`, and `decode_graph*` are not accepted. Unknown
names fail at the engine boundary instead of being rewritten.

## Token budgets

Native Sparse-vLLM token budgets are explicit integer token counts. Ratio-style
values such as `decode_keep_tokens=0.17` are rejected because their meaning
depends on a target context length. Convert ratios before launching a run.

`quest_token_budget` is not a public input. QuEST derives its total selection
budget from `sink_keep_tokens`, `decode_keep_tokens`, and
`recent_keep_tokens`.

## Prefill scheduling

Prefill policy ownership lives in `src/sparsevllm/method_registry.py`:

- `all_chunked` uses `engine_prefill_chunk_size` for regular chunked batching.
- `long_bs1full_short_batch` runs qualifying full-prefill requests atomically
  and uses `long_prefill_offload_threshold` for its long-request boundary.

Do not duplicate method policy decisions in benchmark scripts. Runtime reports
should record the resolved method, policy, chunk size, context length, batch
size, and checkpoint path.

## Prefill sparsity

`prefill_sparse_method` selects the prefill attention algorithm independently
from `sparse_method`. `flashprefill_v2` supports `vanilla`, `omnikv`, `quest`,
`snapkv`, and `h2o` on explicit-KV MHA models, including their supported
prefix-cache modes. MLA latent models reject this prefill method during
configuration. H2O defaults to `prefill_sparse_method="h2o_prefill"`;
explicitly selecting `flashprefill_v2` changes only the prefill attention
computation. H2O still
collects its method-owned posthoc scores and performs its normal prefill KV
compaction. The cache manager owns that physical lifecycle, while the prepared
prefill provider consumes its view without inspecting the cache method name. See
[FlashPrefill V2](../features/flashprefill-v2.md) for the validated kernel
contract and required calibration parameters.

## Runtime invariant validation

`validate_runtime_invariants=False` is the serving fast-path default. Set it to
`True` in allocator, eviction, and CUDA Graph diagnostics to enable expensive
internal checks such as MLA slot range/uniqueness, H2O decode context bounds,
and H2O/SnapKV cross-layer metadata alignment. The option is resolved when the
engine is initialized and is independent of `enable_profiler`, so profiling
does not silently enable debug work.

## DeltaKV

Reportable DeltaKV inference requires a compatible compressor checkpoint.
For registered models, its default `full_attention_layers=auto` consumes the
same model profile as OmniKV:

```python
from sparsevllm import LLM

llm = LLM(
    "/path/to/Qwen3-4B-Instruct-2507",
    sparse_method="deltakv",
    deltakv_checkpoint_path="/path/to/compressor",
    full_attention_layers="auto",
    decode_keep_tokens=2048,
    recent_keep_tokens=128,
    sink_keep_tokens=8,
    engine_prefill_chunk_size=16384,
)
```

The native implementation, cache metadata, loader, and kernels live under
`src/sparsevllm/`. Compressor training is maintained in
[CURRENTF/DeltaKV](https://github.com/CURRENTF/DeltaKV).

## Benchmark adapter

Text benchmarks share `benchmark/model_adapters/sparsevllm.py`. It accepts the
same public parameter names, constructs the native engine, and exposes a small
generation callable for LongBench, MathBench, NIAH, and RULER core. SCBench uses
its native `sparsevllm` attention type. There is no `--backend hf` option.
