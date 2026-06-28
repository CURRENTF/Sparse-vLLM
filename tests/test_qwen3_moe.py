import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from sparsevllm.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoePiecewiseDecodeCudaGraphState,
    Qwen3MoeSparseMoeBlock,
)
from sparsevllm.utils.loader import load_model
from sparsevllm.utils.parallel_context import ParallelContext, parallel_context_scope


def tiny_moe_config(**overrides):
    values = dict(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=3,
        hidden_act="silu",
        num_experts=3,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        head_dim=2,
        rope_theta=1000000,
        rope_scaling=None,
        mlp_chunk_size=16,
        mlp_only_layers=[],
        decoder_sparse_step=1,
        num_hidden_layers=1,
        vocab_size=16,
        tie_word_embeddings=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def reference_moe(block: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(hidden_flat, block.gate.weight)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(router_probs, block.gate.top_k, dim=-1)
    if block.gate.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(router_logits.dtype)

    output = torch.zeros_like(hidden_flat)
    for token_idx in range(hidden_flat.shape[0]):
        token = hidden_flat[token_idx : token_idx + 1]
        for top_k_pos in range(block.gate.top_k):
            expert_idx = int(selected_experts[token_idx, top_k_pos].item())
            gate, up = F.linear(token, block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, block.experts.down_proj[expert_idx])
            output[token_idx] += expert_out.squeeze(0) * routing_weights[token_idx, top_k_pos]
    return output.reshape(batch_size, sequence_length, hidden_dim)


class Qwen3MoeTest(unittest.TestCase):
    def test_sparse_moe_block_matches_reference(self):
        torch.manual_seed(0)
        cfg = tiny_moe_config()
        block = Qwen3MoeSparseMoeBlock(cfg)
        with torch.no_grad():
            block.gate.weight.copy_(torch.randn_like(block.gate.weight))
            block.experts.gate_up_proj.copy_(torch.randn_like(block.experts.gate_up_proj))
            block.experts.down_proj.copy_(torch.randn_like(block.experts.down_proj))
        hidden_states = torch.randn(2, 3, cfg.hidden_size)

        actual = block(hidden_states)
        expected = reference_moe(block, hidden_states)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_sparse_moe_block_accepts_sparsevllm_flat_tokens(self):
        torch.manual_seed(1)
        cfg = tiny_moe_config()
        block = Qwen3MoeSparseMoeBlock(cfg)
        hidden_states = torch.randn(5, cfg.hidden_size)

        output = block(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)

    def test_local_expert_apply_ignores_negative_expert_slots(self):
        torch.manual_seed(2)
        cfg = tiny_moe_config(num_experts=4, num_experts_per_tok=2)
        block = Qwen3MoeSparseMoeBlock(cfg)
        with torch.no_grad():
            block.experts.gate_up_proj.copy_(torch.randn_like(block.experts.gate_up_proj))
            block.experts.down_proj.copy_(torch.randn_like(block.experts.down_proj))
        hidden_states = torch.randn(3, cfg.hidden_size)
        selected_experts = torch.tensor([[0, -1], [1, -1], [-1, -1]])
        routing_weights = torch.tensor([[0.7, 0.3], [1.0, 0.0], [0.5, 0.5]], dtype=hidden_states.dtype)

        actual = block.experts._apply_local_experts(hidden_states, selected_experts, routing_weights)

        expected = torch.zeros_like(hidden_states)
        for token_idx, expert_idx in ((0, 0), (1, 1)):
            token = hidden_states[token_idx : token_idx + 1]
            gate, up = F.linear(token, block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, block.experts.down_proj[expert_idx])
            expected[token_idx] += expert_out.squeeze(0) * routing_weights[token_idx, 0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for torch._grouped_mm")
    def test_local_expert_grouped_mm_matches_reference_cuda(self):
        torch.manual_seed(3)
        cfg = tiny_moe_config(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=3,
        )
        block = Qwen3MoeSparseMoeBlock(cfg).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            block.experts.gate_up_proj.copy_(torch.randn_like(block.experts.gate_up_proj))
            block.experts.down_proj.copy_(torch.randn_like(block.experts.down_proj))
        hidden_states = torch.randn(7, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        selected_experts = torch.tensor(
            [
                [0, 1, -1],
                [2, 3, 1],
                [3, -1, -1],
                [1, 0, 2],
                [2, 0, 3],
                [-1, -1, -1],
                [0, 2, 1],
            ],
            device="cuda",
            dtype=torch.long,
        )
        routing_weights = torch.rand(
            selected_experts.shape,
            device="cuda",
            dtype=torch.bfloat16,
        )
        routing_weights = routing_weights / routing_weights.float().sum(dim=-1, keepdim=True).to(torch.bfloat16)

        self.assertTrue(block.experts._use_grouped_mm(hidden_states))
        actual = block.experts._apply_local_experts(hidden_states, selected_experts, routing_weights)
        expected = block.experts._apply_local_experts_reference(
            hidden_states,
            selected_experts,
            routing_weights,
        )

        torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)

    def test_sparse_moe_block_accepts_qwen3_moe_num_local_experts_config(self):
        cfg = tiny_moe_config()
        cfg.num_local_experts = cfg.num_experts
        delattr(cfg, "num_experts")

        block = Qwen3MoeSparseMoeBlock(cfg)

        self.assertEqual(block.gate.num_experts, 3)
        self.assertEqual(block.experts.num_experts, 3)

    def test_deepep_decode_graph_uses_piecewise_without_env_gate(self):
        ctx = ParallelContext(
            global_rank=0,
            global_world_size=2,
            tp_size=1,
            tp_rank=0,
            ep_size=2,
            ep_rank=0,
            local_rank=0,
        )
        owner = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        mlp=SimpleNamespace(
                            experts=SimpleNamespace(expert_parallel_backend="deepep_v2")
                        )
                    )
                ]
            )
        )
        graph_state = SimpleNamespace(key=SimpleNamespace(batch_size=1, context_capacity=1024))

        with parallel_context_scope(ctx):
            state = Qwen3MoePiecewiseDecodeCudaGraphState(owner, graph_state, graph_pool=None)
            self.assertTrue(state._uses_deepep_v2())
            self.assertFalse(hasattr(state, "_select_capture_mode"))

    def test_piecewise_capture_alignment_uses_warmup(self):
        ctx = ParallelContext(
            global_rank=0,
            global_world_size=2,
            tp_size=1,
            tp_rank=0,
            ep_size=2,
            ep_rank=0,
            local_rank=0,
        )
        owner = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        mlp=SimpleNamespace(
                            experts=SimpleNamespace(expert_parallel_backend="deepep_v2")
                        )
                    )
                ]
            )
        )
        graph_state = SimpleNamespace(key=SimpleNamespace(batch_size=1, context_capacity=1024))
        calls = []

        with parallel_context_scope(ctx):
            state = Qwen3MoePiecewiseDecodeCudaGraphState(owner, graph_state, graph_pool=None)
            state.captured = True
            state.final_graph = object()
            state.final_hidden_input = object()
            state._run_warmup = lambda input_ids, positions, sparse_controller: calls.append("warmup")
            state._barrier_ep_ranks = lambda: calls.append("barrier")
            state.replay = lambda sparse_controller: calls.append("replay")

            state.run(object(), object(), None, force_capture_alignment=True)

        self.assertEqual(calls, ["warmup", "barrier", "replay"])

    def test_decoder_layer_respects_mlp_only_and_sparse_step(self):
        ctx = ParallelContext(
            global_rank=0,
            global_world_size=1,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            local_rank=0,
        )
        with parallel_context_scope(ctx):
            dense_by_step = Qwen3MoeDecoderLayer(
                tiny_moe_config(decoder_sparse_step=2),
                layer_idx=0,
            )
            moe_by_step = Qwen3MoeDecoderLayer(
                tiny_moe_config(decoder_sparse_step=2),
                layer_idx=1,
            )
            dense_by_mlp_only = Qwen3MoeDecoderLayer(
                tiny_moe_config(decoder_sparse_step=1, mlp_only_layers=[0]),
                layer_idx=0,
            )

        self.assertIsInstance(dense_by_step.mlp, Qwen3MoeMLP)
        self.assertIsInstance(moe_by_step.mlp, Qwen3MoeSparseMoeBlock)
        self.assertIsInstance(dense_by_mlp_only.mlp, Qwen3MoeMLP)

    def test_ep_context_allocates_only_local_contiguous_experts(self):
        ctx = ParallelContext(
            global_rank=1,
            global_world_size=2,
            tp_size=1,
            tp_rank=0,
            ep_size=2,
            ep_rank=1,
            local_rank=1,
        )
        with parallel_context_scope(ctx):
            block = Qwen3MoeSparseMoeBlock(
                tiny_moe_config(num_experts=4, expert_parallel_backend="deepep_v2")
            )

        self.assertEqual(block.experts.global_expert_ids, [2, 3])
        self.assertEqual(block.experts.gate_up_proj.shape, torch.Size([2, 6, 4]))
        self.assertEqual(block.experts.down_proj.shape, torch.Size([2, 4, 3]))

    def test_loader_slices_fused_expert_tensor_for_ep_rank(self):
        ctx = ParallelContext(
            global_rank=1,
            global_world_size=2,
            tp_size=1,
            tp_rank=0,
            ep_size=2,
            ep_rank=1,
            local_rank=1,
        )
        gate_up = torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4)
        down = torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            save_file(
                {
                    "model.layers.0.mlp.experts.gate_up_proj": gate_up,
                    "model.layers.0.mlp.experts.down_proj": down,
                },
                str(model_dir / "model.safetensors"),
            )
            with parallel_context_scope(ctx):
                model = Qwen3MoeForCausalLM(
                    tiny_moe_config(num_experts=4, expert_parallel_backend="deepep_v2")
                )
                load_model(model, str(model_dir), rank=0, world_size=1)

        torch.testing.assert_close(
            model.model.layers[0].mlp.experts.gate_up_proj.detach().cpu(),
            gate_up[2:4],
        )
        torch.testing.assert_close(
            model.model.layers[0].mlp.experts.down_proj.detach().cpu(),
            down[2:4],
        )

    def test_loader_loads_named_expert_layout_for_owned_ep_rank(self):
        ctx = ParallelContext(
            global_rank=1,
            global_world_size=2,
            tp_size=1,
            tp_rank=0,
            ep_size=2,
            ep_rank=1,
            local_rank=1,
        )
        expert_tensors = {}
        gate_values = {}
        up_values = {}
        down_values = {}
        for expert_idx in range(4):
            gate = torch.full((3, 4), float(10 + expert_idx))
            up = torch.full((3, 4), float(20 + expert_idx))
            down = torch.full((4, 3), float(30 + expert_idx))
            gate_values[expert_idx] = gate
            up_values[expert_idx] = up
            down_values[expert_idx] = down
            prefix = f"model.layers.0.mlp.experts.{expert_idx}"
            expert_tensors[f"{prefix}.gate_proj.weight"] = gate
            expert_tensors[f"{prefix}.up_proj.weight"] = up
            expert_tensors[f"{prefix}.down_proj.weight"] = down

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            save_file(expert_tensors, str(model_dir / "model.safetensors"))
            with parallel_context_scope(ctx):
                model = Qwen3MoeForCausalLM(
                    tiny_moe_config(num_experts=4, expert_parallel_backend="deepep_v2")
                )
                load_model(model, str(model_dir), rank=0, world_size=1)

        gate_up = model.model.layers[0].mlp.experts.gate_up_proj.detach().cpu()
        down = model.model.layers[0].mlp.experts.down_proj.detach().cpu()
        torch.testing.assert_close(gate_up[0, :3], gate_values[2])
        torch.testing.assert_close(gate_up[0, 3:], up_values[2])
        torch.testing.assert_close(gate_up[1, :3], gate_values[3])
        torch.testing.assert_close(gate_up[1, 3:], up_values[3])
        torch.testing.assert_close(down[0], down_values[2])
        torch.testing.assert_close(down[1], down_values[3])

    def test_loader_fails_when_owned_named_expert_part_is_missing(self):
        ctx = ParallelContext(
            global_rank=1,
            global_world_size=2,
            tp_size=1,
            tp_rank=0,
            ep_size=2,
            ep_rank=1,
            local_rank=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            save_file(
                {
                    "model.layers.0.mlp.experts.2.gate_proj.weight": torch.zeros(3, 4),
                    "model.layers.0.mlp.experts.2.down_proj.weight": torch.zeros(4, 3),
                    "model.layers.0.mlp.experts.3.gate_proj.weight": torch.zeros(3, 4),
                    "model.layers.0.mlp.experts.3.up_proj.weight": torch.zeros(3, 4),
                    "model.layers.0.mlp.experts.3.down_proj.weight": torch.zeros(4, 3),
                },
                str(model_dir / "model.safetensors"),
            )
            with parallel_context_scope(ctx):
                model = Qwen3MoeForCausalLM(
                    tiny_moe_config(num_experts=4, expert_parallel_backend="deepep_v2")
                )
                with self.assertRaisesRegex(RuntimeError, "Incomplete Qwen3-MoE expert checkpoint"):
                    load_model(model, str(model_dir), rank=0, world_size=1)


if __name__ == "__main__":
    unittest.main()
