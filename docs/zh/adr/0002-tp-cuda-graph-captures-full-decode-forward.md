# TP CUDA Graph 捕获完整 decode forward

Sparse-vLLM 通过让每个 TP rank 捕获并 replay 各自完整的 decode forward（包括现有 collective 操作），来支持 TP decode CUDA Graph。这样可以保持模型层边界不变，并与 vLLM 和 SGLang 采用的方向一致；如果当前分布式 backend 无法被安全捕获，runtime 应明确失败，而不是静默 fallback 到静态 eager 执行。

v1 仅支持单节点 TP，支持 `vanilla`、`streamingllm`、`snapkv`、`pyramidkv`、`omnikv`、`quest`、`rkv` 和 `skipkv`。v1 不包括 DeltaKV。QuEST 支持是指：在 TP 本地稀疏选择下，启用 Graph 的结果与同一张量并行 QuEST 静态/eager 路径等价；它不表示与 TP=1 或全局注意力头稀疏选择等价。

稀疏方法采用 TP 本地稀疏选择：每个 rank 从自己的本地注意力头或 KV 头中选择重要 token，不跨 rank 聚合稀疏索引。当 `decode_graph=True` 且 `tensor_parallel_size > 1` 时，运行时会发出警告，因为算法上不能保证该结果与 TP=1 或全局注意力头稀疏选择等价。

TP decode CUDA Graph 会禁用 `decode_graph_capture_sampling`。TP worker 不会实体化由 `ParallelLMHead` 汇聚到 rank 0 的 logits，因此 sampling 仍位于 Graph 捕获之外。

回归验证应使用回归套件的质量层和性能层，并设置 `tensor_parallel_size=2`。质量测试在同一次运行中将 v1 稀疏方法与 TP vanilla 对比，把 D 级结果或崩溃视为失败。性能测试必须记录 `decode_graph_expected=true` 和 `decode_graph_active=true`；v1 启用阶段记录稀疏方法与 vanilla 的吞吐量对比，但不要求每种稀疏方法都优于 vanilla。
