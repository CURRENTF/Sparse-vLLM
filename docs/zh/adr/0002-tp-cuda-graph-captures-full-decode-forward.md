# TP CUDA Graph 捕获完整 decode forward

## 状态

已接受。当前兼容性检查位于 `configs/cuda_graph.py` 和 `method_registry.py`；
本 ADR 记录执行边界，不维护固定的方法或硬件支持清单。

## 决策

每个 TP rank 捕获并 replay 各自完整的 decode forward，包括该 forward 使用的
collective 操作，以保留模型层边界。捕获或执行失败必须明确报错，不能静默
切换到 eager 执行。

稀疏选择在 TP 本地完成：每个 rank 从本地 head 或 KV head 选择 token，不跨
rank 聚合稀疏索引。因此 Graph 正确性是与同一 TP eager/static 路径等价，
不代表与 TP=1 或全局 head 选择等价。稀疏 TP Graph 配置会对此发出警告。

Sampling 保持在 TP Graph 外，因为 worker rank 不会实体化 rank 0 汇聚的
logits。在 TP 下设置 `decode_graph_capture_sampling=True` 会在配置校验时
被拒绝，不会静默重置选项。

验证须区分实际 Graph 捕获/replay、与同一 TP 路径的数值正确性，以及条件
对齐的性能测量。配置允许某个组合，不代表该模型和拓扑已经得到验证。
