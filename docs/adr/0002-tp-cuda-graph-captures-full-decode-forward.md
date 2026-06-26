# TP CUDA graph captures full decode forward

Sparse-vLLM will support tensor-parallel decode CUDA graphs by having every TP rank capture and replay its own full decode forward, including existing collective operations. This keeps model-layer boundaries intact and matches the direction used by vLLM and SGLang; if the active distributed backend cannot be captured safely, the runtime should fail explicitly rather than silently falling back to static eager execution.

v1 is single-node TP only and supports `vanilla`, `streamingllm`, `snapkv`, `pyramidkv`, `omnikv`, `rkv`, and `skipkv`. DeltaKV and QuEST are excluded from v1; QuEST can be added in v1.1 after separate validation.

Sparse methods use TP-local sparse selection: every rank selects important tokens from its own local heads or KV heads, without cross-rank sparse-index aggregation. The runtime warns when `decode_cuda_graph=True` and `tensor_parallel_size > 1` because this is not guaranteed algorithmically equivalent to TP=1 or global-head sparse selection.

TP decode CUDA graph disables `decode_cuda_graph_capture_sampling`. TP workers do not materialize rank-0 gathered logits from `ParallelLMHead`, so sampling remains outside graph capture in TP.

Regression validation should use the regression suite quality and perf layers with `tensor_parallel_size=2`. Quality compares v1 sparse methods against TP vanilla in the same run and treats D grades or crashes as failures. Perf must record `decode_cuda_graph_expected=true` and `decode_cuda_graph_active=true`; v1 enablement records sparse-vs-vanilla throughput but does not require every sparse method to beat vanilla.
