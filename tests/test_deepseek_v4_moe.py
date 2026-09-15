"""Full native FFN composition against dequantized mathematical operators."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from sparsevllm.distributed import ParallelTopology, init_parallel_context, reset_parallel_context
from sparsevllm.models.deepseek_v4.moe import DeepseekV4Moe
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager
from sparsevllm.quantization.config import QuantizationConfig
from test_mxfp4_moe import _oracle, _quantize_dequantize_activation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_hash", [True, False])
@torch.inference_mode()
def test_native_ffn_shared_addition_chunking_and_graph(use_hash, tmp_path):
    # Individual expert tests miss shared/routed composition, original token
    # IDs through multiple chunks, and scratch reuse across different routers.
    close_workspace_manager()
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    init_parallel_context(topology=ParallelTopology(1, 1, 1))
    try:
        torch.manual_seed(731)
        config = SimpleNamespace(
            hidden_size=256, moe_intermediate_size=128, n_routed_experts=4,
            num_experts_per_tok=2, vocab_size=37, num_hash_layers=1, n_shared_experts=1,
            norm_topk_prob=True, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
            swiglu_limit=10., expert_dtype="fp4",
        )
        quantization = QuantizationConfig.from_hf_config(
            {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
             "weight_block_size": [128, 128], "scale_fmt": "ue8m0"}, max_num_tokens=4,
        )
        with torch.device("cuda"):
            model = DeepseekV4Moe(config, 0 if use_hash else 1, quantization=quantization,
                                  mlp_chunk_size=4, cuda_graph=True)
        model.gate.weight.normal_(std=.1)
        if use_hash:
            model.gate.tid2eid.copy_(torch.rand((37, 4), device="cuda").argsort(-1)[:, :2])
        else:
            model.gate.bias.normal_()
        logical, shared = [], []
        for name, n, k in (("w1", 128, 256), ("w3", 128, 256), ("w2", 256, 128)):
            weight = torch.randint(0, 256, (4, n, k // 2), device="cuda", dtype=torch.uint8)
            scale = torch.randint(120, 124, (4, n, k // 32), device="cuda", dtype=torch.uint8)
            logical.append((weight, scale))
            for expert in range(4):
                model.experts.load_projection(expert, name, weight[expert], scale[expert])
            shared_weight = (torch.randn((n, k), device="cuda") * 8).to(torch.float8_e4m3fn)
            shared_scale = torch.full((n // 128, k // 128), 1 / 128, device="cuda")
            model.shared_experts.load_projection(name, shared_weight, shared_scale)
            shared.append(shared_weight.float() / 128)
        model.experts.validate_loaded_weights()
        lock_workspace_manager()
        x = torch.randn((9, 256), device="cuda", dtype=torch.bfloat16) * 3
        tokens = torch.arange(9, device="cuda", dtype=torch.int64)

        def reference():
            scores = F.softplus(F.linear(x.float(), model.gate.weight)).sqrt()
            ids = model.gate.tid2eid[tokens].long() if use_hash else (scores + model.gate.bias).topk(2, dim=-1).indices
            weights = scores.gather(1, ids)
            weights = weights / weights.sum(-1, keepdim=True) * 1.5
            result = _oracle(x, ids, weights, logical, 0, 10.)
            a = _quantize_dequantize_activation(x)
            gate = (a @ shared[0].T).bfloat16().float().clamp(max=10.)
            up = (a @ shared[1].T).bfloat16().float().clamp(-10., 10.)
            activated = (F.silu(gate) * up).bfloat16()
            shared_out = (_quantize_dequantize_activation(activated) @ shared[2].T).bfloat16()
            return (result + shared_out.float()).bfloat16()

        assert model(x[:0], tokens[:0]).shape == (0, 256)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            model(x, tokens)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = model(x, tokens)
        for _ in range(2):
            x.mul_(.75)
            tokens.add_(3)
            graph.replay()
            expected = reference()
            torch.testing.assert_close(captured, expected, rtol=1e-2, atol=3e-3)
            torch.testing.assert_close(captured, model(x, tokens), rtol=0, atol=0)
    finally:
        close_workspace_manager()
        reset_parallel_context()
        dist.destroy_process_group()
