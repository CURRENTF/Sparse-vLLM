# Runtime Parameter Semantics

Sparse-vLLM has one inference backend: the native engine under
`src/sparsevllm/`. Benchmark entrypoints normalize their public parameters and
construct `sparsevllm.LLM`; this repository does not provide an HF DeltaKV
backend or reference implementation.

## Public names

Use semantic public names in commands, JSON configs, and benchmark manifests:

| Public name | Native engine field | Meaning |
| --- | --- | --- |
| `sparse_method` | `vllm_sparse_method` | Sparse method selector. |
| `deltakv_checkpoint_path` | `deltakv_path` | DeltaKV compressor checkpoint path. |
| `engine_prefill_chunk_size` | `chunk_prefill_size` | Maximum scheduled prefill chunk. |
| `sink_keep_tokens` | `num_sink_tokens` | Fixed sink-token budget. |
| `recent_keep_tokens` | `num_recent_tokens` | Recent-token budget. |
| `full_attention_layers` | `full_attn_layers` | Comma-separated full-layer indices. |
| `deltakv_neighbor_count` | `deltakv_k_neighbors` | Number of DeltaKV reference neighbors. |
| `deltakv_center_ratio` | `cluster_ratio` | DeltaKV reference-center ratio. |
| `deltakv_latent_dim` | `kv_compressed_size` | Compressor latent width. |
| `deltakv_latent_quant_bits` | `kv_quant_bits` | Quantization bits for latent state. |
| `deltakv_latent_quant_group_size` | `kv_quant_group_size` | Latent quantization group size. |

`src/sparsevllm/configs/runtime_params.py` performs this mapping at the engine
boundary. Internal names such as `vllm_sparse_method`, `deltakv_path`, and
`chunk_prefill_size` are rejected when supplied through the public normalizer.
Conflicting aliases fail instead of silently choosing a value.

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

## DeltaKV

Reportable DeltaKV inference requires a compatible compressor checkpoint:

```python
from sparsevllm import LLM

llm = LLM(
    "/path/to/model",
    sparse_method="deltakv",
    deltakv_checkpoint_path="/path/to/compressor",
    full_attention_layers="0,1,3,9,13,16,21,28",
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
generation callable for LongBench, MathBench, NIAH, and RULER-VT. SCBench uses
its native `sparsevllm` attention type. There is no `--backend hf` option.
