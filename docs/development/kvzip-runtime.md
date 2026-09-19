# KVzip runtime contract

`sparse_method="kvzip"` applies the existing `kvzip_global` maintenance scoring
policy to an ordinary completed prompt. This token-shared variant differs from
[upstream KVzip](https://github.com/snu-mllab/KVzip)'s non-uniform per-head cache.
It requires no external scoring checkpoint.

For each replay chunk, teacher-force the reconstruction instruction, preceding
context and current source chunk after the full original prompt. Each layer's
scorer normalizes probabilities over original prompt keys, averages over replay
queries, and takes the maximum over query heads. Reduce by maximum across layers,
chunks and attention TP ranks. Keep the highest-scoring prompt positions with
earlier positions breaking ties, then restore their original order. There are no
separate sink/recent reservations and no decode-time eviction.

## Ownership and compatibility

- `KVzipRuntime` owns chunk orchestration, the global score accumulator and final
  logical selection. The ordinary prefill output supplies the first token.
- `KVzipCacheManager` reuses SnapKV's layer slot allocator, physical compaction,
  decode views and batch-only graph state. It owns temporary replay slots and
  tiled scorer workspace. A failed replay releases its temporary suffix before
  propagating the error. Original prompt tokens accumulate on the host across
  chunk-only TP messages and are released after compaction or sequence free.
  No replay token enters request history or sampling.
- `AuxiliaryPrefillRequest` is a narrow unsampled-forward hook. ModelRunner
  copies the sequence, prepares its existing row and restores the caller's
  forward context. Attention, scheduler and controller contain no KVzip branches.
- Completed requests are processed serially after a possibly batched/chunked
  prefill. Incomplete rows stay dense. Generated KV appends to retained rows.
  Preemption reconstructs the original prompt and reapplies compression.
- Qwen2/Qwen3/Llama explicit KV and attention TP use the same mechanism. Recurrent,
  MLA, shared-KV, MoE, sparse-prefill overrides, prefix reuse and offload are
  rejected until their auxiliary-forward lifecycles are implemented.
- Decode reuses ordinary prepared providers and SnapKV's `unified` batch-only
  graph inputs. Reconstruction and compaction finish outside graph replay;
  changing retained lengths changes metadata, not graph identity.

## Cost and reuse

Let `L` be prompt length, `C` source chunk size, `R` maximum replay length,
`A` layers, `H` local query heads, `D` head dimension and `K` retained tokens.
Original prefill remains dense. Reconstruction takes `ceil(L/C)` extra forwards,
each with attention/scoring cost `O(A H R L D)` plus ordinary model projections
and feed-forward work. Final stable selection is `O(L log L)`; slot-table
compaction is `O(A L)` and does not copy retained KV. Decode reads `K+t` positions
after `t` generated tokens. There is no claim of an end-to-end speedup.

The temporary KV peak adds at most `engine_prefill_chunk_size` slots per layer,
shared across serial replays. That reserve is excluded from admission, prefill
and decode-window capacity, independently of persistent prompt reservations.
The temporary startup cache includes this reserve in addition to its profiling
workload and accounts for the same payload and slot metadata as the allocator.
Prompt plus replay must fit the logical row capacity. Model activation peaks
remain within the ordinary prefill chunk and batch limits.

The score accumulator is float32 `[L]`; the reused layer output is float32
`[1,L+R]`. Host source history costs `O(L)` integer entries per active prefill
request on each rank. The scorer preallocates bounded partial/global LSE tiles, at most
`4 * H_padded * R_padded * (ceil(max_model_len / block_n)+1)` bytes,
plus a float32 `[H,max_model_len]` query-reduction workspace. Additional
`64 * A * max_model_len` bytes conservatively cover score, index/sort and
compaction intermediates. Cache sizing subtracts this workspace bound before
allocating production KV. For `A=32,H=32,D=128,Lmax=32768,Rmax=2048`, the bound
is approximately 205 MiB, excluding model activations and temporary KV slots.

Reused symbols are `prefill_score_fwd` / `PrefillScoreWorkspace` for the exact
existing probability policy and `SnapKVCacheManager.free_part_slots` for physical
eviction. Stable device `argsort` matches the maintenance API's host tie rule
without copying scores to CPU. There are no new kernels or dependencies.
The inspected vLLM 0.26 FlashAttention interface and SGLang attention/sgl-kernel
interfaces expose attention output/LSE, not this query-averaged, head-max score
over the original-key-only normalization domain. Ordinary attention therefore
keeps its upstream provider; scoring uses the existing repository kernel.

## Validation boundary

CPU lifecycle checks cover scratch rollback on partial failure, missing-layer
failure, deterministic global selection, incomplete batch rows, capacity
headroom and full release. The CUDA oracle explicitly forms GQA probabilities
and compares query/head/layer reductions. The existing
`scripts/debug/compare_decode_graph_eager_logits.py --method kvzip` checks full
eager versus actual graph logits, compaction evidence, padded mixed-length
batches, repeated requests and no serving recapture. An explicit
`max_model_len` in `--hyper_params` leaves room for auxiliary replay.

Quality and efficiency claims require a fixed evaluation workload and matched
measurements including reconstruction. Synthetic logits checks establish runtime
correctness, not benchmark quality or throughput.
