# MLA Triton kernels

This directory contains the minimal LightLLM-derived mathematical kernels used
by Sparse-vLLM's GLM-4.7 MLA path. It deliberately has no dependency on the
LightLLM Python package or runtime.

## Upstream source

- Repository: <https://github.com/ModelTC/lightllm>
- Commit: `65c174ee95ac6a6fd36b18b63d0b33d97e76b770`
- License: Apache-2.0; see `LICENSE.lightllm`

| Local file | Upstream file | Local changes |
| --- | --- | --- |
| `decode_stage1.py` | `lightllm/common/basemodel/triton_kernel/mla_att/decode_att/gqa_flash_decoding_stage1.py` | Removed device probing and runtime state; caller supplies strides, workspace, and static launch values; restricted shapes to GLM's 512+64 latent layout. |
| `decode_stage2.py` | `lightllm/common/basemodel/triton_kernel/mla_att/decode_att/gqa_flash_decoding_stage2.py` | Removed runtime imports; caller supplies schedule/workspace; added zero-context padded-row behavior. |
| `decode_schedule.py` | `lightllm/common/basemodel/triton_kernel/mla_att/decode_att/gqa_flash_decoding.py` | Rewritten around immutable config and caller-owned workspace; removed `infer_state`, global tuning, device inspection, and forward allocation. |
| `copy_latent.py` | `lightllm/common/basemodel/triton_kernel/kv_copy/mla_copy_kv.py` | Added `slot_mapping < 0` padding mask, bounds protection, layout checks, and opt-in-once slot validation. |
| `gather_latent.py` | `lightllm/models/deepseek2/triton_kernel/sample_kv.py` | Replaced modulo sampling with a padding-safe, ragged, request-indirected full-history gather into explicit packed outputs. |

## Contract

- Query latent: `[batch, local_heads, 512]`, BF16.
- Query RoPE: `[batch, local_heads, 64]`, BF16.
- Persistent caches: `[slots, 1, 512]` and `[slots, 1, 64]`, BF16.
- Slot/request/context metadata: INT32 CUDA tensors.
- Decode scale: `256 ** -0.5`, matching the pre-absorption GLM QK head
  dimension.
- Decode workspaces and outputs are allocated by the caller. The run path does
  not inspect device properties or allocate tensors.

`DEFAULT_GLM_MLA_DECODE_CONFIG` is a conservative correctness configuration
for the initial H100 implementation. It is not recorded as tuned until the
target GLM shape has dedicated benchmark evidence.

Synchronous slot-value validation is exposed separately so the cache manager
can validate a mapping once and reuse it across layers. Kernel calls remain
bounds-safe; passing `validate_slots=False` or `validate_metadata=False` means
the owning runtime boundary has already established the corresponding value
invariants.
