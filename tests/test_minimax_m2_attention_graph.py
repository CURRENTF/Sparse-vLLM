import pytest
import torch

from sparsevllm.triton_kernel.flash_decoding_stage2 import flash_decode_stage2
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import flash_decode_stage1
from sparsevllm.triton_kernel.store_kvcache import store_kvcache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_minimax_m2_gqa_decode_cuda_graph_replay():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(27)
    batch_size = 4
    num_q_heads = 48
    num_kv_heads = 8
    head_dim = 128
    block_seq = 256
    max_context_len = 1024
    num_blocks = max_context_len // block_seq

    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    k = torch.randn(
        160,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    v = torch.randn(
        160,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    req_to_tokens = torch.zeros(8, max_context_len, dtype=torch.int32, device=device)
    for row in range(8):
        req_to_tokens[row, :18] = torch.arange(
            row * 18,
            (row + 1) * 18,
            dtype=torch.int32,
            device=device,
        )
    req_indices = torch.arange(4, dtype=torch.int32, device=device)
    context_lens = torch.full((batch_size,), 17, dtype=torch.int32, device=device)
    slot_mapping = torch.tensor([16, 34, 52, 70], dtype=torch.int32, device=device)
    new_k = torch.randn(
        batch_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    new_v = torch.randn(
        new_k.shape,
        device=device,
        dtype=new_k.dtype,
        generator=generator,
    )
    mid_o = torch.empty(
        batch_size,
        num_q_heads,
        num_blocks,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    mid_lse = torch.empty(
        batch_size,
        num_q_heads,
        num_blocks,
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty_like(q)

    def run_decode():
        store_kvcache(new_k, new_v, k, v, slot_mapping)
        flash_decode_stage1(
            q,
            k,
            v,
            req_to_tokens,
            req_indices,
            context_lens,
            max_context_len,
            mid_o,
            mid_lse,
            block_seq,
        )
        flash_decode_stage2(mid_o, mid_lse, context_lens, output, block_seq)

    for _ in range(2):
        run_decode()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_decode()

    for context_len in (17, 18):
        req_indices.copy_(torch.arange(4, 8, dtype=torch.int32, device=device))
        q.copy_(torch.randn(q.shape, device=device, dtype=q.dtype, generator=generator))
        new_k.copy_(torch.randn(new_k.shape, device=device, dtype=new_k.dtype, generator=generator))
        new_v.copy_(torch.randn(new_v.shape, device=device, dtype=new_v.dtype, generator=generator))
        context_lens.fill_(context_len)
        slot_mapping.copy_(
            torch.tensor(
                [context_len - 1 + row * 18 for row in range(4, 8)],
                dtype=torch.int32,
                device=device,
            )
        )
        k.index_fill_(0, slot_mapping.to(torch.long), float("nan"))
        v.index_fill_(0, slot_mapping.to(torch.long), float("nan"))
        graph.replay()
        graph_output = output.clone()
        k.index_fill_(0, slot_mapping.to(torch.long), float("nan"))
        v.index_fill_(0, slot_mapping.to(torch.long), float("nan"))
        run_decode()
        torch.cuda.synchronize()
        assert torch.equal(graph_output, output)
