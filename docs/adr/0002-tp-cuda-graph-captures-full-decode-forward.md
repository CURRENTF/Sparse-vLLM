# TP CUDA graph captures full decode forward

Sparse-vLLM will support tensor-parallel decode CUDA graphs by having every TP rank capture and replay its own full decode forward, including existing collective operations. This keeps model-layer boundaries intact and matches the direction used by vLLM and SGLang; if the active distributed backend cannot be captured safely, the runtime should fail explicitly rather than silently falling back to static eager execution.

v1 is single-node TP only and supports `vanilla`, `streamingllm`, `snapkv`, `pyramidkv`, `omnikv`, `rkv`, and `skipkv`. DeltaKV and QuEST are excluded from v1. QuEST v1.1 support means graph-on equivalence to the same tensor-parallel QuEST static/eager path under TP-local sparse selection, not equivalence to TP=1 or global-head sparse selection. QuEST v1.1 does not include prefix-cache support; tensor-parallel decode CUDA graphs remain incompatible with prefix caching until that combination is validated separately.

Sparse methods use TP-local sparse selection: every rank selects important tokens from its own local heads or KV heads, without cross-rank sparse-index aggregation. The runtime warns when `decode_cuda_graph=True` and `tensor_parallel_size > 1` because this is not guaranteed algorithmically equivalent to TP=1 or global-head sparse selection.

TP decode CUDA graph disables `decode_cuda_graph_capture_sampling`. TP workers do not materialize rank-0 gathered logits from `ParallelLMHead`, so sampling remains outside graph capture in TP.

Regression validation for QuEST v1.1 should use the regression suite LongBench quality layer with `tensor_parallel_size=2`. Quality compares QuEST against TP vanilla in the same run and treats D grades or crashes as failures. QuEST v1.1 support requires the LongBench quality subset with the necessary `vanilla,quest` comparison, not a full default-method sweep and not only a graph-active perf smoke.
