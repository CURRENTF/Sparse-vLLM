"""Full compression preserves carry semantics and values across prefill splits."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sparsevllm.distributed import ParallelTopology, init_parallel_context, reset_parallel_context
from sparsevllm.engine.cache_manager.methods.deepseek_v4 import CompressionChunk, CompressionPlan
from sparsevllm.engine.cache_manager.native_attention import CompressionBatchView
from sparsevllm.engine.cache_manager.storage.shared_kv import CompressionCarryStorage
from sparsevllm.models.deepseek_v4.compression import DeepseekV4Compressor
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_compressor_prefill_splits_preserve_values_and_raw_carry(tmp_path):
    # Exercise projection, pooling and carry together; isolated pooling tests
    # cannot catch a stale projection row or a misplaced chunk boundary.
    close_workspace_manager()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    init_parallel_context(topology=ParallelTopology(1, 1, 1))
    try:
        torch.manual_seed(731)
        config = SimpleNamespace(hidden_size=4096, qk_rope_head_dim=64, rms_norm_eps=1e-6)
        with torch.device("cuda"):
            model = DeepseekV4Compressor(config, ratio=4, head_dim=512, rotate=False,
                                        max_num_tokens=137, max_num_requests=1)
        for weight in (model.wkv.weight, model.wgate.weight):
            weight.copy_((torch.randn_like(weight) / 64).bfloat16())
        model.ape.normal_()
        model.norm_weight.fill_(1.)
        x = torch.randn((137, 4096), dtype=torch.bfloat16, device="cuda")
        frequencies = 10000 ** (-torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64)
        lock_workspace_manager()
        outputs, states = [], []
        for step in (len(x), 16):
            state = CompressionCarryStorage(num_rows=1, ratio=4, head_dim=512, device="cuda")
            for tensor in state.accounting_tensors():
                tensor.zero_()
            chunks = []
            for start in range(0, len(x), step):
                end = min(start + step, len(x))
                plan = CompressionPlan.prefill((CompressionChunk(0, start, end - start),), 4)
                metadata = [torch.tensor(values, dtype=torch.int32, device="cuda") for values in
                            (plan.request_rows, plan.cu_seqlens, plan.start_positions,
                             plan.boundary_requests, plan.boundary_ends)]
                view = CompressionBatchView(*state.accounting_tensors(), *metadata)
                values, positions = model(x[start:end], view, frequencies)
                torch.testing.assert_close(positions, torch.tensor(plan.boundary_ends, device="cuda", dtype=torch.int32) - 4)
                chunks.append(values.clone())
            outputs.append(torch.cat(chunks))
            states.append(state.accounting_tensors())
        for whole, chunked in zip(states[0], states[1]):
            tolerance = dict(rtol=1e-5, atol=4e-6) if whole.is_floating_point() else dict(rtol=0, atol=0)
            torch.testing.assert_close(whole, chunked, **tolerance)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-2, atol=8e-3)
    finally:
        close_workspace_manager()
        reset_parallel_context()
        dist.destroy_process_group()
