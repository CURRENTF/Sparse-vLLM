# TP radix prefix cache with decode CUDA graph uses rank-local mirrored caches

This decision concerns radix prefix caching. Chain lifecycle and admission are
described in [ADR 0004](0004-linear-chain-prefix-cache.md).

Sparse-VLLM supports `vanilla`, `omnikv`, and `quest` with tensor parallelism, prefix cache, and decode CUDA graph enabled together by keeping prefix caches rank-local but logically mirrored: each rank stores its own KV payload and shares stable block identities derived from the same token path and fingerprint. Decode CUDA graph keys remain shape/execution-family keys and do not include prefix-hit length or block ids; prefix hits update row, slot, page, and context metadata through the existing static decode preparation buffers. Prefix-cache control APIs execute on all ranks and return the rank-0 logical view after worker failures have been synchronized; `decode_graph_capture_sampling=True` is rejected for this combination. Current model, method, and topology eligibility is checked by `configs/prefix_cache.py`, `configs/cuda_graph.py`, and the method registry; it is not a claim of validation coverage.
