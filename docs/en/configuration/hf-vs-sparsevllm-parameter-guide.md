# HF vs Sparse-vLLM Backend Parameter Guide

This guide has been merged into the repo-wide runtime parameter audit:

[`runtime-parameter-semantics.md`](runtime-parameter-semantics.md)

Keep this file as a compatibility entrypoint for old links. The main document
now covers:

- strict canonical runtime parameters and rejected legacy names,
- `sparse_method` routing for HF and Sparse-vLLM,
- `deltakv_checkpoint_path` routing for HF and Sparse-vLLM,
- `hf_prefill_chunk_size` versus `engine_prefill_chunk_size`,
- `compressor_token_group_size` versus `deltakv_neighbor_count`,
- DeltaKV standard, cluster, and residual-quant cache behavior,
- LLaVA-OneVision visual-token pruning parameters,
- benchmark-specific speed and admission controls.
