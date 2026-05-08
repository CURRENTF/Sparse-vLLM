import unittest

import torch

from sparsevllm.triton_kernel.deltakv_kernels import (
    deltakv_delta_quant_reconstruct_writeback_int4,
    deltakv_reconstruct_writeback_grouped_heads,
)
from sparsevllm.triton_kernel.quant import triton_quantize_and_pack_along_last_dim, unpack_4bit_to_16bit


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for DeltaKV Triton kernel tests.")
class DeltaKVDeltaQuantKernelTest(unittest.TestCase):
    def test_fused_int4_reconstruct_matches_unpack_then_reconstruct(self):
        torch.manual_seed(0)
        device = "cuda"
        dtype = torch.float16
        num_tokens = 5
        num_slots = 16
        num_heads = 2
        head_dim = 8
        kv_dim = num_heads * head_dim
        max_pos = 32

        angles = torch.randn(max_pos, head_dim // 2, device=device, dtype=torch.float32) * 0.1
        cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(dtype)

        k_base = torch.randn(num_slots, num_heads, head_dim, device=device, dtype=dtype)
        v_base = torch.randn_like(k_base)
        k_ref = k_base.clone()
        v_ref = v_base.clone()
        k_fused = k_base.clone()
        v_fused = v_base.clone()

        residual = torch.randn(num_tokens, 2 * kv_dim, device=device, dtype=dtype) * 0.2
        packed, scale, mn = triton_quantize_and_pack_along_last_dim(
            residual.unsqueeze(0).unsqueeze(0),
            residual.shape[-1],
            4,
        )
        packed = packed.squeeze(0).squeeze(0).contiguous()
        scale = scale.squeeze(0).squeeze(0).contiguous()
        mn = mn.squeeze(0).squeeze(0).contiguous()

        latent_slots = torch.arange(num_tokens, device=device, dtype=torch.int32)
        packed_cache = torch.empty((num_slots, packed.shape[1]), device=device, dtype=torch.int32)
        scale_cache = torch.empty((num_slots, 1), device=device, dtype=dtype)
        mn_cache = torch.empty((num_slots, 1), device=device, dtype=dtype)
        packed_cache[latent_slots.long()] = packed
        scale_cache[latent_slots.long()] = scale
        mn_cache[latent_slots.long()] = mn

        father_slots = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
            device=device,
            dtype=torch.int32,
        )
        slot_to_pos = torch.arange(num_slots, device=device, dtype=torch.int32)
        out_slots = torch.tensor([8, 9, 10, 11, 12], device=device, dtype=torch.int32)
        out_pos = torch.tensor([7, 9, 11, 13, 15], device=device, dtype=torch.int32)

        kv_delta = unpack_4bit_to_16bit(
            packed.unsqueeze(0).unsqueeze(0),
            scale.unsqueeze(0).unsqueeze(0),
            mn.unsqueeze(0).unsqueeze(0),
            residual.shape[-1],
        ).squeeze(0).squeeze(0)
        deltakv_reconstruct_writeback_grouped_heads(
            kv_delta=kv_delta,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_ref,
            v_cache=v_ref,
            heads_per_program=2,
        )
        deltakv_delta_quant_reconstruct_writeback_int4(
            packed_delta_cache=packed_cache,
            scale_cache=scale_cache,
            min_cache=mn_cache,
            latent_slots=latent_slots,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_fused,
            v_cache=v_fused,
            heads_per_program=2,
        )
        torch.cuda.synchronize()

        out = out_slots.long()
        self.assertTrue(torch.allclose(k_ref[out], k_fused[out], atol=2e-3, rtol=2e-3))
        self.assertTrue(torch.allclose(v_ref[out], v_fused[out], atol=4e-3, rtol=2e-3))


if __name__ == "__main__":
    unittest.main()
