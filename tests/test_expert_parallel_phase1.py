import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.config import Config
from sparsevllm.engine.cache_manager.base import CacheManager
from sparsevllm.layers.embed_head import VocabParallelEmbedding
from sparsevllm.layers.linear import QKVParallelLinear, RowParallelLinear
from sparsevllm.utils.parallel_context import (
    ParallelContext,
    build_parallel_context,
    parallel_context_scope,
)


class DummyCacheManager(CacheManager):
    def allocate_kv_cache(self):
        raise NotImplementedError

    def get_layer_batch_states(self, layer_idx: int):
        raise NotImplementedError

    def get_layer_kv_cache(self, layer_idx: int):
        raise NotImplementedError

    def get_layer_store_view(self, layer_idx: int):
        raise NotImplementedError

    def get_layer_compute_tensors(self, layer_idx: int, selection=None):
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int):
        raise NotImplementedError

    def num_free_slots(self):
        raise NotImplementedError

    def free_seq(self, seq_id: int):
        raise NotImplementedError

    def free_part_slots(self, seq):
        raise NotImplementedError

    def _prepare_prefill(self, seqs):
        raise NotImplementedError

    def _prepare_decode(self, seqs):
        raise NotImplementedError


class ExpertParallelPhase1Test(unittest.TestCase):
    def hf_config(self):
        return SimpleNamespace(
            model_type="qwen3",
            torch_dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )

    def _config(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "TinyQwen3"
            model_dir.mkdir()
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=self.hf_config()):
                return Config(model=str(model_dir), **kwargs)

    def test_torch_backend_supports_native_data_parallel(self):
        cfg = self._config(data_parallel_size=3)

        self.assertEqual(cfg.tensor_parallel_size, 1)
        self.assertEqual(cfg.data_parallel_size, 3)
        self.assertEqual(cfg.expert_parallel_size, 1)
        self.assertEqual(cfg.parallel_world_size, 3)
        self.assertEqual(cfg.expert_parallel_backend, "torch")
        self.assertEqual(cfg.expert_placement_policy, "contiguous")

    def test_torch_backend_rejects_expert_parallel(self):
        with self.assertRaisesRegex(ValueError, "requires expert_parallel_size=1"):
            self._config(expert_parallel_size=4)

    def test_all_reduce_backend_was_removed(self):
        with self.assertRaisesRegex(ValueError, "all_reduce.*removed"):
            self._config(expert_parallel_backend="all_reduce")

    def test_overlap_flag_was_removed_from_config_api(self):
        with self.assertRaisesRegex(TypeError, "expert_parallel_overlap_data_parallel"):
            self._config(expert_parallel_overlap_data_parallel=True)

    def test_deepep_v2_config_maps_dp_to_ep_world(self):
        cfg = self._config(
            expert_parallel_size=4,
            expert_parallel_backend="deep_ep_v2",
            decode_graph=True,
        )

        self.assertEqual(cfg.expert_parallel_backend, "deepep_v2")
        self.assertEqual(cfg.data_parallel_size, 4)
        self.assertEqual(cfg.parallel_world_size, 4)
        self.assertTrue(cfg.decode_cuda_graph)
        self.assertEqual(cfg.hf_config.expert_parallel_backend, "deepep_v2")
        self.assertEqual(cfg.hf_config.data_parallel_size, 4)
        self.assertEqual(cfg.hf_config.expert_parallel_size, 4)

    def test_deepep_v2_rejects_mismatched_dp_ep(self):
        with self.assertRaisesRegex(ValueError, "must equal expert_parallel_size"):
            self._config(
                data_parallel_size=3,
                expert_parallel_size=2,
                expert_parallel_backend="deepep_v2",
            )

    def test_deepep_v2_requires_expert_parallel(self):
        with self.assertRaisesRegex(ValueError, "requires expert_parallel_size > 1"):
            self._config(
                expert_parallel_backend="deepep_v2",
            )

    def test_deepep_v2_rejects_tp_hybrid_v1(self):
        with self.assertRaisesRegex(ValueError, "requires tensor_parallel_size=1"):
            self._config(tensor_parallel_size=2, expert_parallel_size=2, expert_parallel_backend="deepep_v2")

    def test_native_dp_rejects_tp_hybrid_v1(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined with tensor_parallel_size > 1"):
            self._config(tensor_parallel_size=2, data_parallel_size=2)

    def test_distributed_master_port_validation(self):
        cfg = self._config(distributed_master_port=29501)

        self.assertEqual(cfg.distributed_master_port, 29501)
        with self.assertRaisesRegex(ValueError, "distributed_master_port must be in"):
            self._config(distributed_master_port=70000)

    def test_deepep_v2_accepts_decode_graph_alias(self):
        cfg = self._config(
            expert_parallel_size=2,
            expert_parallel_backend="deepep_v2",
            decode_graph=True,
        )

        self.assertTrue(cfg.decode_cuda_graph)

    def test_parallel_context_maps_native_dp_world_without_tp_shard(self):
        cfg = self._config(data_parallel_size=4)
        ctx = build_parallel_context(cfg, global_rank=3)

        self.assertEqual(ctx.global_rank, 3)
        self.assertEqual(ctx.global_world_size, 4)
        self.assertEqual(ctx.tp_size, 1)
        self.assertEqual(ctx.tp_rank, 0)
        self.assertEqual(ctx.ep_size, 1)
        self.assertEqual(ctx.ep_rank, 0)
        self.assertEqual(ctx.dp_size, 4)
        self.assertEqual(ctx.dp_rank, 3)

    def test_parallel_context_maps_deepep_dp_ep_rank(self):
        cfg = self._config(
            expert_parallel_size=4,
            expert_parallel_backend="deepep_v2",
        )
        ctx = build_parallel_context(cfg, global_rank=2)

        self.assertEqual(ctx.global_rank, 2)
        self.assertEqual(ctx.global_world_size, 4)
        self.assertEqual(ctx.tp_size, 1)
        self.assertEqual(ctx.tp_rank, 0)
        self.assertEqual(ctx.ep_size, 4)
        self.assertEqual(ctx.ep_rank, 2)
        self.assertEqual(ctx.dp_size, 4)
        self.assertEqual(ctx.dp_rank, 2)
        self.assertTrue(ctx.overlap_data_parallel)

    def test_dense_layers_do_not_shard_under_ep_context(self):
        ctx = ParallelContext(
            global_rank=2,
            global_world_size=4,
            tp_size=1,
            tp_rank=0,
            ep_size=4,
            ep_rank=2,
            local_rank=2,
        )
        with parallel_context_scope(ctx):
            qkv = QKVParallelLinear(
                hidden_size=8,
                head_size=4,
                total_num_heads=2,
                total_num_kv_heads=2,
                bias=False,
            )
            row = RowParallelLinear(input_size=16, output_size=8, bias=False)
            embed = VocabParallelEmbedding(num_embeddings=32, embedding_dim=8)

        self.assertEqual(qkv.tp_size, 1)
        self.assertEqual(qkv.weight.shape, torch.Size([24, 8]))
        self.assertEqual(qkv.num_heads, 2)
        self.assertEqual(qkv.num_kv_heads, 2)
        self.assertEqual(row.weight.shape, torch.Size([8, 16]))
        self.assertEqual(embed.weight.shape, torch.Size([32, 8]))

    def test_cache_manager_uses_tp_size_not_ep_world_size(self):
        cfg = self._config(
            expert_parallel_size=4,
            expert_parallel_backend="deepep_v2",
        )
        cache_manager = DummyCacheManager(cfg, rank=2, world_size=cfg.tensor_parallel_size)

        self.assertEqual(cache_manager.world_size, 1)
        self.assertEqual(cache_manager.num_kv_heads, cfg.hf_config.num_key_value_heads)

    def test_cache_manager_uses_parallel_context_local_rank_for_device(self):
        cfg = self._config(
            expert_parallel_size=4,
            expert_parallel_backend="deepep_v2",
        )
        ctx = ParallelContext(
            global_rank=2,
            global_world_size=4,
            tp_size=1,
            tp_rank=0,
            ep_size=4,
            ep_rank=2,
            local_rank=0,
        )

        with parallel_context_scope(ctx):
            cache_manager = DummyCacheManager(cfg, rank=2, world_size=cfg.tensor_parallel_size)

        self.assertEqual(cache_manager.device, cache_manager.platform.get_device(0))

    def test_qwen3_moe_decode_graph_keeps_static_eager_guard_without_piecewise(self):
        cfg = self._config()
        cfg.hf_config.model_type = "qwen3_moe"
        cache_manager = DummyCacheManager(cfg, rank=0, world_size=cfg.tensor_parallel_size)

        self.assertTrue(cache_manager.decode_cuda_graph_force_eager())


if __name__ == "__main__":
    unittest.main()
