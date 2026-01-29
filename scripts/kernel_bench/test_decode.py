import torch
import time
import numpy as np
from flash_attn import flash_attn_with_kvcache
from sparsevllm.triton_kernel.flash_decoding_stage1 import (
    flash_decode_stage1,
    flash_decode_stage1_with_score
)
from sparsevllm.triton_kernel.flash_decoding_stage2 import flash_decode_stage2
from sparsevllm.triton_kernel.gqa_decode_flashattention_nopad import gqa_decode_attention_fwd
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import flash_decode_stage1 as gqa_flash_decode_stage1
from sparsevllm.triton_kernel.gqa_flash_decoding_stage2 import flash_decode_stage2 as gqa_flash_decode_stage2

def benchmark_gqa_kernels(
    batch_sizes=[1, 8, 32],
    num_heads=28,
    num_kv_heads=4,
    head_dim=128,
    seq_lens=[16384, 65536, 131072],
    block_seq=256,
    num_iters=100,
    warmup_iters=10
):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"\nBenchmarking GQA (Grouped Query Attention) Kernels Comparison")
    print(f"Heads: {num_heads}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")
    print("-" * 140)
    
    header = f"{'Batch':>5} | {'SeqLen':>8} | {'FA2-Full':>12} | {'Base-FD':>15} | {'GQA-SinglePass':>15} | {'GQA-FlashDecode':>15} | {'Speedup (vs Base)'}"
    print(header)
    print("-" * 140)

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # --- Data Preparation ---
            q = torch.randn((batch_size, num_heads, head_dim), dtype=dtype, device=device)
            k = torch.randn((batch_size * seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
            v = torch.randn((batch_size * seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
            
            req_to_tokens = torch.arange(batch_size * seq_len, device=device, dtype=torch.int32).reshape(batch_size, seq_len)
            b_req_idx = torch.arange(batch_size, device=device, dtype=torch.int32)
            b_seqlen = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
            
            o_tensor = torch.empty((batch_size, num_heads, head_dim), dtype=dtype, device=device)
            
            num_blocks = (seq_len + block_seq - 1) // block_seq
            mid_out = torch.empty((batch_size, num_heads, num_blocks, head_dim), dtype=torch.float32, device=device)
            mid_exp = torch.empty((batch_size, num_heads, num_blocks), dtype=torch.float32, device=device)

            def time_func(func, *args, **kwargs):
                for _ in range(warmup_iters): func(*args, **kwargs)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_iters): func(*args, **kwargs)
                torch.cuda.synchronize()
                return (time.time() - start) / num_iters * 1000

            # 1. FA2
            k_fa = k.view(batch_size, seq_len, num_kv_heads, head_dim).contiguous()
            v_fa = v.view(batch_size, seq_len, num_kv_heads, head_dim).contiguous()
            q_fa = q.unsqueeze(1)
            t_fa2 = time_func(flash_attn_with_kvcache, q_fa, k_fa, v_fa, cache_seqlens=b_seqlen, causal=True)

            # 2. Base Flash Decoding
            def base_fd():
                flash_decode_stage1(q, k, v, req_to_tokens, b_req_idx, b_seqlen, seq_len, mid_out, mid_exp, block_seq)
                flash_decode_stage2(mid_out, mid_exp, b_seqlen, o_tensor, block_seq)
            t_base_fd = time_func(base_fd)

            # 3. GQA Nopad (Single Pass)
            t_gqa_nopad = time_func(gqa_decode_attention_fwd, q, k, v, o_tensor, req_to_tokens, b_req_idx, b_seqlen)

            # 4. GQA Flash Decoding (Stage 1 + Stage 2)
            def gqa_fd():
                gqa_flash_decode_stage1(q, k, v, req_to_tokens, b_req_idx, b_seqlen, seq_len, mid_out, mid_exp, block_seq)
                gqa_flash_decode_stage2(mid_out, mid_exp, b_seqlen, o_tensor, block_seq)
            t_gqa_fd = time_func(gqa_fd)

            speedup = t_base_fd / t_gqa_fd if t_gqa_fd > 0 else 0
            
            print(f"{batch_size:5d} | {seq_len:8d} | {t_fa2:12.4f} | {t_base_fd:15.4f} | {t_gqa_nopad:15.4f} | {t_gqa_fd:15.4f} | {speedup:7.2f}x")

def benchmark_flash_decode(
    batch_sizes=[1, 8, 32],
    num_heads=28,
    num_kv_heads=4,
    head_dim=128,
    seq_lens=[16384, 65536, 131072],
    num_sparse_tokens=4096,
    block_seq=256,
    num_iters=100,
    warmup_iters=10
):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Benchmarking Attention Kernels: Full vs Sparse")
    print(f"Heads: {num_heads}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")
    print(f"Sparse Tokens: {num_sparse_tokens}")
    print("-" * 140)
    
    # Header for comprehensive comparison
    header = f"{'Batch':>5} | {'Total':>7} | {'FA2-Full':>10} | {'Tri-FD-Full':>12} | {'S1-Base':>9} | {'S1-Score3D':>11} | {'S1-Score2D':>11} | {'FA2-Sparse':>10} | {'Tri-FD-Spar':>11} | {'Speedup':>7}"
    print(header)
    print("-" * 155)

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # --- Data Preparation (Full) ---
            q = torch.randn((batch_size, num_heads, head_dim), dtype=dtype, device=device)
            max_total_tokens = batch_size * seq_len
            k = torch.randn((max_total_tokens, num_kv_heads, head_dim), dtype=dtype, device=device)
            v = torch.randn((max_total_tokens, num_kv_heads, head_dim), dtype=dtype, device=device)
            
            req_to_tokens_full = torch.arange(max_total_tokens, device=device, dtype=torch.int32).reshape(batch_size, seq_len)
            b_req_idx = torch.arange(batch_size, device=device, dtype=torch.int32)
            b_seqlen_full = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
            
            num_blocks_full = (seq_len + block_seq - 1) // block_seq
            mid_out_full = torch.empty((batch_size, num_heads, num_blocks_full, head_dim), dtype=torch.float32, device=device)
            mid_exp_full = torch.empty((batch_size, num_heads, num_blocks_full), dtype=torch.float32, device=device)
            o_tensor = torch.empty((batch_size, num_heads, head_dim), dtype=dtype, device=device)
            
            attn_score_3d = torch.empty((batch_size, num_heads, seq_len), dtype=dtype, device=device)
            attn_score_2d = torch.full((batch_size, seq_len), -float('inf'), dtype=torch.float32, device=device)

            # --- Data Preparation (Sparse) ---
            sparse_indices = []
            for b in range(batch_size):
                indices = torch.randperm(seq_len, device=device)[:num_sparse_tokens].to(torch.int32)
                sparse_indices.append(indices + b * seq_len)
            sparse_indices_tensor = torch.stack(sparse_indices)
            b_seqlen_sparse = torch.full((batch_size,), num_sparse_tokens, device=device, dtype=torch.int32)
            
            num_blocks_sparse = (num_sparse_tokens + block_seq - 1) // block_seq
            mid_out_sparse = torch.empty((batch_size, num_heads, num_blocks_sparse, head_dim), dtype=torch.float32, device=device)
            mid_exp_sparse = torch.empty((batch_size, num_heads, num_blocks_sparse), dtype=torch.float32, device=device)

            def time_func(func, *args, **kwargs):
                for _ in range(warmup_iters): func(*args, **kwargs)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_iters): func(*args, **kwargs)
                torch.cuda.synchronize()
                return (time.time() - start) / num_iters * 1000

            # 1. FA2 Full
            k_fa_full = k.view(batch_size, seq_len, num_kv_heads, head_dim).contiguous()
            v_fa_full = v.view(batch_size, seq_len, num_kv_heads, head_dim).contiguous()
            q_fa = q.unsqueeze(1)
            t_fa2_full = time_func(flash_attn_with_kvcache, q_fa, k_fa_full, v_fa_full, cache_seqlens=b_seqlen_full, causal=True)

            # 2. Triton Full (S1 + S2)
            def full_tri():
                flash_decode_stage1(q, k, v, req_to_tokens_full, b_req_idx, b_seqlen_full, seq_len, mid_out_full, mid_exp_full, block_seq)
                flash_decode_stage2(mid_out_full, mid_exp_full, b_seqlen_full, o_tensor, block_seq)
            t_tri_full = time_func(full_tri)

            # 3. Triton S1 Base
            t_s1_base = time_func(flash_decode_stage1, q, k, v, req_to_tokens_full, b_req_idx, b_seqlen_full, seq_len, mid_out_full, mid_exp_full, block_seq)

            # 4. Triton S1 3D
            t_s1_3d = time_func(flash_decode_stage1_with_score, q, k, v, req_to_tokens_full, b_req_idx, b_seqlen_full, seq_len, mid_out_full, mid_exp_full, attn_score_3d, block_seq)

            # 5. Triton S1 2D
            t_s1_2d = time_func(flash_decode_stage1_with_score, q, k, v, req_to_tokens_full, b_req_idx, b_seqlen_full, seq_len, mid_out_full, mid_exp_full, attn_score_2d, block_seq)

            # 6. FA2 Sparse (IndexSelect + FA2)
            def sparse_fa2():
                k_s = torch.index_select(k, 0, sparse_indices_tensor.flatten()).view(batch_size, num_sparse_tokens, num_kv_heads, head_dim)
                v_s = torch.index_select(v, 0, sparse_indices_tensor.flatten()).view(batch_size, num_sparse_tokens, num_kv_heads, head_dim)
                return flash_attn_with_kvcache(q_fa, k_s, v_s, cache_seqlens=b_seqlen_sparse, causal=True)
            t_fa2_sparse = time_func(sparse_fa2)

            # 7. Triton Sparse (Direct)
            def sparse_tri():
                flash_decode_stage1(q, k, v, sparse_indices_tensor, b_req_idx, b_seqlen_sparse, num_sparse_tokens, mid_out_sparse, mid_exp_sparse, block_seq)
                flash_decode_stage2(mid_out_sparse, mid_exp_sparse, b_seqlen_sparse, o_tensor, block_seq)
            t_tri_sparse = time_func(sparse_tri)

            speedup = t_fa2_sparse / t_tri_sparse if t_tri_sparse > 0 else 0
            
            print(f"{batch_size:5d} | {seq_len:7d} | {t_fa2_full:10.4f} | {t_tri_full:12.4f} | {t_s1_base:9.4f} | {t_s1_3d:11.4f} | {t_s1_2d:11.4f} | {t_fa2_sparse:10.4f} | {t_tri_sparse:11.4f} | {speedup:7.2f}x")

def benchmark_memory_order(
    batch_sizes=[1, 8, 32],
    num_heads=28,
    num_kv_heads=4,
    head_dim=128,
    seq_lens=[16384, 65536, 131072],
    num_sparse_tokens=4096,
    block_seq=256,
    num_iters=100,
    warmup_iters=10
):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"\nBenchmarking Memory Order Impact (Ordered vs Shuffled)")
    print(f"Heads: {num_heads}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")
    print(f"Sparse Tokens: {num_sparse_tokens}")
    print("-" * 120)
    
    header = f"{'Batch':>5} | {'SeqLen':>8} | {'Full-Ordered':>12} | {'Full-Shuffled':>13} | {'Sparse-Ordered':>14} | {'Sparse-Shuffled':>15} | {'Slowdown (F/S)'}"
    print(header)
    print("-" * 135)

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # --- Data Preparation ---
            q = torch.randn((batch_size, num_heads, head_dim), dtype=dtype, device=device)
            k = torch.randn((batch_size * seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
            v = torch.randn((batch_size * seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
            
            b_req_idx = torch.arange(batch_size, device=device, dtype=torch.int32)
            
            # 1. Full Ordered
            req_to_tokens_full_ord = torch.arange(batch_size * seq_len, device=device, dtype=torch.int32).reshape(batch_size, seq_len)
            b_seqlen_full = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
            
            # 2. Full Shuffled
            req_to_tokens_full_shuf = torch.stack([torch.randperm(seq_len, device=device) + b * seq_len for b in range(batch_size)]).to(torch.int32)
            
            # 3. Sparse Ordered (Sorted) & Shuffled
            sparse_indices_ord = []
            sparse_indices_shuf = []
            for b in range(batch_size):
                idx = torch.randperm(seq_len, device=device)[:num_sparse_tokens].to(torch.int32)
                sparse_indices_shuf.append(idx + b * seq_len)
                sparse_indices_ord.append(torch.sort(idx)[0] + b * seq_len)
            
            req_to_tokens_sparse_ord = torch.stack(sparse_indices_ord)
            req_to_tokens_sparse_shuf = torch.stack(sparse_indices_shuf)
            b_seqlen_sparse = torch.full((batch_size,), num_sparse_tokens, device=device, dtype=torch.int32)

            num_blocks_full = (seq_len + block_seq - 1) // block_seq
            mid_out_full = torch.empty((batch_size, num_heads, num_blocks_full, head_dim), dtype=torch.float32, device=device)
            mid_exp_full = torch.empty((batch_size, num_heads, num_blocks_full), dtype=torch.float32, device=device)

            num_blocks_sparse = (num_sparse_tokens + block_seq - 1) // block_seq
            mid_out_sparse = torch.empty((batch_size, num_heads, num_blocks_sparse, head_dim), dtype=torch.float32, device=device)
            mid_exp_sparse = torch.empty((batch_size, num_heads, num_blocks_sparse), dtype=torch.float32, device=device)

            def time_func(func, *args, **kwargs):
                for _ in range(warmup_iters): func(*args, **kwargs)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_iters): func(*args, **kwargs)
                torch.cuda.synchronize()
                return (time.time() - start) / num_iters * 1000

            t_full_ord = time_func(flash_decode_stage1, q, k, v, req_to_tokens_full_ord, b_req_idx, b_seqlen_full, seq_len, mid_out_full, mid_exp_full, block_seq)
            t_full_shuf = time_func(flash_decode_stage1, q, k, v, req_to_tokens_full_shuf, b_req_idx, b_seqlen_full, seq_len, mid_out_full, mid_exp_full, block_seq)
            
            t_sparse_ord = time_func(flash_decode_stage1, q, k, v, req_to_tokens_sparse_ord, b_req_idx, b_seqlen_sparse, num_sparse_tokens, mid_out_sparse, mid_exp_sparse, block_seq)
            t_sparse_shuf = time_func(flash_decode_stage1, q, k, v, req_to_tokens_sparse_shuf, b_req_idx, b_seqlen_sparse, num_sparse_tokens, mid_out_sparse, mid_exp_sparse, block_seq)

            slowdown_full = t_full_shuf / t_full_ord if t_full_ord > 0 else 0
            slowdown_sparse = t_sparse_shuf / t_sparse_ord if t_sparse_ord > 0 else 0
            
            print(f"{batch_size:5d} | {seq_len:8d} | {t_full_ord:12.4f} | {t_full_shuf:13.4f} | {t_sparse_ord:14.4f} | {t_sparse_shuf:15.4f} | {slowdown_full:.2f}x / {slowdown_sparse:.2f}x")

if __name__ == "__main__":
    benchmark_gqa_kernels()
    benchmark_flash_decode()
    benchmark_memory_order()
