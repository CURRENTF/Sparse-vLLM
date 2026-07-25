# HF 与 Sparse-vLLM 后端参数指南

本指南已经合并到仓库范围的运行时参数审计文档中：

[`runtime-parameter-semantics.md`](runtime-parameter-semantics.md)

保留此文件作为旧链接的兼容入口。主文档现在涵盖：

- 严格的规范运行时参数以及被拒绝的旧参数名；
- HF 和 Sparse-vLLM 的 `sparse_method` 路由；
- HF 和 Sparse-vLLM 的 `deltakv_checkpoint_path` 路由；
- `hf_prefill_chunk_size` 与 `engine_prefill_chunk_size` 的区别；
- `compressor_token_group_size` 与 `deltakv_neighbor_count` 的区别；
- DeltaKV standard cache、cluster cache 和 residual-quant cache 的行为；
- LLaVA-OneVision 视觉 token 剪枝参数；
- 基准测试特定的速度与准入控制参数。
