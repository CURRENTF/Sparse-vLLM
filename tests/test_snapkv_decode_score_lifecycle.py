import json
from types import SimpleNamespace
import unittest

import torch

from sparsevllm.engine.decode_cuda_graph import DecodeCudaGraphRunner
from sparsevllm.engine.llm_engine import LLMEngine
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.sparse_controller import SparseController
from sparsevllm.utils.context import reset_context, set_context


class WorkerInfoTest(unittest.TestCase):
    def test_snapkv_selection_config_is_reported(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            model="model",
            hf_config=SimpleNamespace(
                model_type="test",
                vocab_size=32_000,
            ),
            vllm_sparse_method="snapkv",
            pool_kernel_size=7,
            sparse_attn_score_dtype="float16",
        )

        benchmark_config = engine.worker_info()["benchmark_config"]

        self.assertEqual(benchmark_config["pool_kernel_size"], 7)
        self.assertEqual(
            benchmark_config["sparse_attn_score_dtype"],
            "float16",
        )
        self.assertEqual(engine.worker_info()["vocab_size"], 32_000)
        revision = engine.worker_info()["code_revision"]
        self.assertTrue(revision["git_commit"] or revision["package_version"])

    def test_prefix_offload_capacity_config_is_reported_json_safely(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            model="model",
            hf_config=SimpleNamespace(model_type="test", vocab_size=32_000),
            vllm_sparse_method="",
            enable_prefix_caching=True,
            prefix_cache_block_size=16,
            prefix_cache_max_blocks=4_096,
            prefix_cache_requested_max_blocks=8_192,
            enable_prefix_cache_offload=True,
            prefix_cache_host_size_gb=6.5,
            recurrent_state_max_bytes=1 << 30,
            prefix_cache_max_recurrent_bytes=None,
            recurrent_state_pool_bytes=768 << 20,
            recurrent_state_bytes_per_row=16_384,
            recurrent_state_row_capacity=49_152,
            prefix_recurrent_bytes_per_block=16_384,
            prefix_recurrent_capacity_bytes=256 << 20,
            prefix_kv_bytes_per_block=65_536,
            prefix_kv_block_capacity=4_096,
            kv_allocatable_bytes=512 << 20,
            num_kvcache_slots=262_144,
            max_num_seqs_in_gpu=64,
        )

        worker_info = engine.worker_info()
        benchmark_config = worker_info["benchmark_config"]

        self.assertIs(benchmark_config["enable_prefix_cache_offload"], True)
        self.assertEqual(benchmark_config["prefix_cache_host_size_gb"], 6.5)
        self.assertEqual(benchmark_config["recurrent_state_max_bytes"], 1 << 30)
        self.assertIsNone(benchmark_config["prefix_cache_max_recurrent_bytes"])
        self.assertEqual(benchmark_config["prefix_cache_requested_max_blocks"], 8_192)
        self.assertEqual(benchmark_config["prefix_cache_max_blocks"], 4_096)
        self.assertEqual(benchmark_config["recurrent_state_pool_bytes"], 768 << 20)
        self.assertEqual(benchmark_config["recurrent_state_bytes_per_row"], 16_384)
        self.assertEqual(benchmark_config["recurrent_state_row_capacity"], 49_152)
        self.assertEqual(benchmark_config["prefix_recurrent_bytes_per_block"], 16_384)
        self.assertEqual(benchmark_config["prefix_recurrent_capacity_bytes"], 256 << 20)
        self.assertEqual(benchmark_config["prefix_kv_bytes_per_block"], 65_536)
        self.assertEqual(benchmark_config["prefix_kv_block_capacity"], 4_096)
        self.assertEqual(benchmark_config["kv_allocatable_bytes"], 512 << 20)
        self.assertEqual(benchmark_config["num_kvcache_slots"], 262_144)
        self.assertEqual(worker_info["max_num_seqs_in_gpu"], 64)
        json.dumps(worker_info)


def make_controller(
    method="snapkv",
    *,
    layers=2,
    kv_len=6,
    graph=False,
    graph_capacity=None,
    keep=2,
    score_dtype="float32",
    sink=1,
    recent=1,
    pyramid_ratios=None,
    is_long_text=True,
):
    layout = SimpleNamespace(
        kv_layer_index=lambda layer: int(layer),
        is_full_attention=lambda layer: 0 <= int(layer) < layers,
    )
    config = SimpleNamespace(
        vllm_sparse_method=method,
        obs_layer_ids=[],
        full_attn_layers=[],
        hf_config=SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=8,
            num_attention_heads=2,
            torch_dtype=torch.float32,
        ),
        runtime_layout=layout,
        num_sink_tokens=sink,
        num_recent_tokens=recent,
        decode_keep_tokens=keep,
        sparse_attn_score_dtype=score_dtype,
        tensor_parallel_size=1,
        snapkv_num_full_layers=0,
        pyramid_layer_ratios=(
            pyramid_ratios
            if pyramid_ratios is not None
            else ([1.0] * layers if method == "pyramidkv" else None)
        ),
        decode_cuda_graph=graph,
        pool_kernel_size=1,
    )

    class Manager:
        device = torch.device("cpu")

        def __init__(self):
            self.compactions = []
            if graph:
                self._decode_static_max_context_len = int(
                    graph_capacity if graph_capacity is not None else kv_len
                )

        def get_layer_batch_states(self, layer):
            del layer
            return SimpleNamespace(
                context_lens=torch.tensor([kv_len], dtype=torch.int32),
                max_context_len=kv_len,
                req_indices=torch.tensor([0], dtype=torch.int32),
            )

        def decode_kv_lens_for_layer(self, layer, seqs):
            del layer
            return [kv_len for _seq in seqs]

        def free_part_slots(self, layer, seq, keep_indices):
            self.compactions.append((layer, seq.seq_id, keep_indices.clone()))

    manager = Manager()
    controller = SparseController(config, manager)
    seqs = [Sequence([1])]
    set_context(
        False,
        cache_manager=manager,
        is_long_text=is_long_text,
        seqs=seqs,
    )
    controller.prepare_forward(seqs, is_prefill=False)
    return controller, manager, seqs


class SnapKVDecodeScoreLifecycleTest(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_decode_uses_independent_fused_2d_layer_scores(self):
        controller, _manager, _seqs = make_controller()
        states = controller.layer_batch_sparse_states
        stable_ptrs = [states[layer].attn_score.data_ptr() for layer in range(2)]
        self.assertNotEqual(*stable_ptrs)
        q = torch.empty((1, 2, 4))

        score0 = controller.get_decode_selection(0, q).attn_score
        self.assertEqual(tuple(score0.shape), (1, 6))
        score0.copy_(torch.tensor([[7, 9, 8, 4, 6, 6]]))
        controller.on_layer_attention_end(0)
        torch.testing.assert_close(
            states[0].attn_score,
            torch.tensor([[7, 9, 8, 4, 6, 6]]).float(),
        )
        layer0 = states[0].attn_score.clone()

        score1 = controller.get_decode_selection(1, q).attn_score
        self.assertNotEqual(score0.data_ptr(), score1.data_ptr())
        self.assertTrue(torch.equal(score1, torch.full_like(score1, -1e20)))
        score1.copy_(torch.tensor([[4, 5, 2, 8, 2, 9]]))
        controller.on_layer_attention_end(1)
        torch.testing.assert_close(states[0].attn_score, layer0)
        torch.testing.assert_close(
            states[1].attn_score,
            torch.tensor([[4, 5, 2, 8, 2, 9]]).float(),
        )
        self.assertEqual([states[layer].attn_score.dim() for layer in range(2)], [2, 2])

    def test_pyramidkv_uses_the_same_fused_2d_lifecycle(self):
        controller, _manager, _seqs = make_controller("pyramidkv", layers=1)
        state = controller.layer_batch_sparse_states[0]
        score = controller.get_decode_selection(
            0,
            torch.empty((1, 2, 4)),
        ).attn_score
        score.copy_(torch.tensor([[6, 7, 8, 4, 5, 9]]))
        controller.on_layer_attention_end(0)
        torch.testing.assert_close(
            state.attn_score,
            torch.tensor([[6, 7, 8, 4, 5, 9]]).float(),
        )

    def test_fused_decode_score_stays_float32_for_low_precision_score_configs(self):
        configured_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        for method in ("snapkv", "pyramidkv"):
            for score_name, configured_dtype in configured_dtypes.items():
                with self.subTest(method=method, score_dtype=score_name):
                    controller, manager, seqs = make_controller(
                        method,
                        layers=1,
                        score_dtype=score_name,
                    )
                    score = controller.get_decode_selection(
                        0,
                        torch.empty((1, 2, 4)),
                    ).attn_score
                    self.assertEqual(controller.attn_score_dtype, configured_dtype)
                    self.assertEqual(score.dtype, torch.float32)
                    self.assertEqual(
                        controller._snapkv_decode_reduced_attn_score_buffers[0].dtype,
                        torch.float32,
                    )
                    score.copy_(torch.arange(6, dtype=torch.float32).reshape(1, 6))
                    controller.on_layer_attention_end(0)
                    controller.post_forward(seqs, is_prefill=False)
                    self.assertEqual(len(manager.compactions), 1)

    def test_graph_refs_are_2d_keepalive_and_score_before_trigger(self):
        controller, _manager, seqs = make_controller(
            layers=1,
            kv_len=7,
            graph=True,
            graph_capacity=16,
            keep=4,
        )
        self.assertEqual(controller._snapkv_decode_trigger_len(6), 8)
        self.assertTrue(controller._needs_attn_score(0, False, seqs))
        score = controller.get_decode_selection(
            0,
            torch.empty((1, 2, 4)),
        ).attn_score
        self.assertEqual(tuple(score.shape), (1, 16))
        score[:, :7].fill_(3)
        controller.on_layer_attention_end(0)

        runner = object.__new__(DecodeCudaGraphRunner)
        runner.sparse_controller = controller
        refs = runner._snapshot_sparse_state_refs()
        self.assertEqual(refs[0]["attn_score"].dim(), 2)
        self.assertEqual(tuple(refs[0]["attn_score"].shape), (1, 16))
        self.assertTrue(
            torch.equal(
                refs[0]["attn_score"][:, 7:],
                torch.full((1, 9), -1e20),
            )
        )
        runner._reset_graph_input_attn_scores(refs)
        self.assertTrue(
            torch.equal(
                refs[0]["attn_score"],
                torch.full((1, 16), -1e20),
            )
        )
        keepalive = controller.decode_cuda_graph_keepalive_tensors()
        self.assertEqual(sum(tensor.dim() == 3 for tensor in keepalive), 0)
        self.assertEqual(sum(tensor.dim() == 2 for tensor in keepalive), 1)
        controller.layer_batch_sparse_states[0].attn_score = None
        runner._restore_sparse_state_refs(SimpleNamespace(sparse_state_refs=refs))
        self.assertIs(controller.layer_batch_sparse_states[0].attn_score, refs[0]["attn_score"])

        controller.config.decode_cuda_graph = False
        self.assertFalse(controller._needs_attn_score(0, False, seqs))

    def test_pyramid_short_graph_uses_layer_trigger_and_graph_capacity(self):
        common = {
            "method": "pyramidkv",
            "layers": 2,
            "graph": True,
            "keep": 4_096,
            "sink": 64,
            "recent": 512,
            "pyramid_ratios": [0.6, 0.01],
            "is_long_text": False,
        }
        controller, _manager, _seqs = make_controller(
            **common,
            kv_len=700,
            graph_capacity=1_024,
        )
        low_budget = controller._get_layer_budget(1, is_prefill=False)
        self.assertEqual(low_budget, 644)
        self.assertEqual(controller._snapkv_decode_trigger_len(low_budget), 712)
        self.assertIsNone(controller.layer_batch_sparse_states[0].attn_score)
        self.assertEqual(
            tuple(controller.layer_batch_sparse_states[1].attn_score.shape),
            (1, 1_024),
        )

        below, _manager, _seqs = make_controller(
            **common,
            kv_len=700,
            graph_capacity=700,
        )
        self.assertIsNone(below.layer_batch_sparse_states[1].attn_score)

        snap_short, _manager, _seqs = make_controller(
            "snapkv",
            layers=1,
            kv_len=5,
            graph=True,
            graph_capacity=16,
            keep=4,
            sink=1,
            recent=1,
            is_long_text=False,
        )
        self.assertEqual(
            snap_short._snapkv_decode_trigger_len(
                snap_short._get_layer_budget(0, is_prefill=False)
            ),
            8,
        )
        self.assertIsNone(snap_short.layer_batch_sparse_states[0].attn_score)

        triggered, manager, seqs = make_controller(
            **common,
            kv_len=712,
            graph_capacity=1_024,
        )
        score = triggered.layer_batch_sparse_states[1].attn_score
        score.copy_(torch.arange(1_024, dtype=torch.float32).reshape(1, -1))
        triggered.on_layer_attention_end(1)
        triggered.post_forward(seqs, is_prefill=False)
        self.assertEqual(len(manager.compactions), 1)
        self.assertEqual(manager.compactions[0][0], 1)
        self.assertEqual(manager.compactions[0][2].numel(), low_budget)

    def test_post_forward_consumes_2d_and_rejects_3d_scores(self):
        controller, manager, seqs = make_controller(layers=1)
        score = controller.get_decode_selection(
            0,
            torch.empty((1, 2, 4)),
        ).attn_score
        score.copy_(torch.arange(6, dtype=torch.float32).reshape(1, 6))
        controller.on_layer_attention_end(0)
        shapes = []

        def select(scores, kv_len, budget, **_kwargs):
            shapes.append(tuple(scores.shape))
            self.assertEqual((kv_len, budget), (6, 4))
            return torch.tensor([0, 2, 3, 5])

        controller._snapkv_select_indices = select
        controller.post_forward(seqs, is_prefill=False)
        self.assertEqual(shapes, [(6,)])
        self.assertEqual(manager.compactions[0][2].tolist(), [0, 2, 3, 5])

        controller.prepare_forward(seqs, is_prefill=False)
        controller.layer_batch_sparse_states[0].attn_score = torch.zeros(
            (1, 2, 6)
        )
        with self.assertRaisesRegex(RuntimeError, r"head-reduced \[B, L\]"):
            controller.on_layer_attention_end(0)


if __name__ == "__main__":
    unittest.main()
