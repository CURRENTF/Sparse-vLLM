import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sparsevllm.config import Config
from sparsevllm.engine.cache_manager.quest import QuestCacheManager
from sparsevllm.engine.decode_cuda_graph import DecodeCudaGraphKey, DecodeCudaGraphRunner, DecodeCudaGraphState


class DecodeCudaGraphTPConfigTest(unittest.TestCase):
    def hf_config(self):
        return SimpleNamespace(
            model_type="qwen2",
            torch_dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )

    def _config(self, method: str, *, model_name: str = "TinyModel", **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / model_name
            model_dir.mkdir()
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=self.hf_config()):
                return Config(
                    model=str(model_dir),
                    vllm_sparse_method=method,
                    decode_cuda_graph=True,
                    tensor_parallel_size=2,
                    max_decoding_seqs=4,
                    **kwargs,
                )

    def test_tp_decode_cuda_graph_accepts_v1_methods(self):
        for method in ["vanilla", "streamingllm", "snapkv", "pyramidkv", "omnikv", "quest", "rkv"]:
            with self.subTest(method=method):
                cfg = self._config(method)
                self.assertTrue(cfg.decode_cuda_graph)
                self.assertEqual(cfg.tensor_parallel_size, 2)

        cfg = self._config("skipkv", model_name="DeepSeek-R1-Distill-Qwen-7B")
        self.assertEqual(cfg.vllm_sparse_method, "skipkv")

    def test_tp_decode_cuda_graph_rejects_deltakv(self):
        with self.assertRaisesRegex(ValueError, "DeltaKV is not supported"):
            self._config("deltakv", allow_missing_deltakv_path=True)

    def test_tp_decode_cuda_graph_accepts_prefix_cache_methods(self):
        for method in ["vanilla", "omnikv", "quest"]:
            with self.subTest(method=method):
                cfg = self._config(method, enable_prefix_caching=True)
                self.assertTrue(cfg.decode_cuda_graph)
                self.assertTrue(cfg.enable_prefix_caching)
                self.assertFalse(cfg.decode_cuda_graph_capture_sampling)

    def test_tp_decode_cuda_graph_rejects_capture_sampling(self):
        with self.assertRaisesRegex(ValueError, "capture_sampling is disabled"):
            self._config("snapkv", decode_cuda_graph_capture_sampling=True)

    def test_quest_derives_token_budget_from_common_keep_tokens(self):
        cfg = self._config(
            "quest",
            num_sink_tokens=3,
            decode_keep_tokens=40,
            num_recent_tokens=5,
        )
        self.assertEqual(cfg.quest_token_budget, 48)

        manager = object.__new__(QuestCacheManager)
        manager.config = cfg
        manager._build_decode_view_static = Mock(
            return_value=(
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([1]),
            )
        )
        manager.build_decode_view(
            layer_idx=cfg.quest_skip_layers,
            q=torch.empty((1, 1, 1)),
            active_slots=torch.tensor([0]),
            req_indices=torch.tensor([0]),
            context_lens=torch.tensor([1]),
            num_heads=1,
            num_kv_heads=1,
        )
        self.assertEqual(
            manager._build_decode_view_static.call_args.kwargs["token_budget"],
            48,
        )

    def test_quest_config_constructor_rejects_explicit_token_budget(self):
        with self.assertRaisesRegex(TypeError, "quest_token_budget"):
            self._config("quest", quest_token_budget=48)

    def test_quest_rejects_invalid_derived_token_budget_inputs(self):
        invalid_configs = (
            {"num_sink_tokens": -1},
            {"decode_keep_tokens": 1.5},
            {
                "num_sink_tokens": 0,
                "decode_keep_tokens": 0,
                "num_recent_tokens": 0,
            },
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    ValueError,
                    "QuEST .* must be a non-negative integer|derived token budget must be > 0",
                ):
                    self._config("quest", **kwargs)


class DecodeCudaGraphDebugRefsTest(unittest.TestCase):
    def test_replay_restores_graph_captured_model_debug_refs(self):
        class DebugModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.debug_last_tensor = torch.tensor([1.0])
                self.debug_last_nested = {"items": [torch.tensor([2.0])]}
                self.not_debug = torch.tensor([3.0])

        class Engine:
            def __init__(self):
                self.model = DebugModule()

            def run_model(self, *_args):
                raise AssertionError("not called")

        engine = Engine()
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.run_model = engine.run_model
        captured_tensor = engine.model.debug_last_tensor
        captured_nested = engine.model.debug_last_nested

        refs = runner._snapshot_model_debug_refs()
        engine.model.debug_last_tensor = torch.tensor([4.0])
        engine.model.debug_last_nested = {"items": [torch.tensor([5.0])]}
        state = DecodeCudaGraphState(
            key=DecodeCudaGraphKey(
                method="",
                batch_size=1,
                context_capacity=1024,
                is_long_text=False,
                capture_sampling=False,
            ),
            model_debug_refs=refs,
        )

        runner._restore_model_debug_refs(state)

        self.assertIs(engine.model.debug_last_tensor, captured_tensor)
        self.assertIs(engine.model.debug_last_nested, captured_nested)
        tensor_refs = list(runner._debug_tensor_refs(captured_nested))
        self.assertEqual(len(tensor_refs), 1)
        self.assertIs(tensor_refs[0], captured_nested["items"][0])
        self.assertTrue(all(name != "not_debug" for _, name, _ in refs))


if __name__ == "__main__":
    unittest.main()
