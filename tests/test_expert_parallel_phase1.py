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

    def test_expert_parallel_config_sets_parallel_world_size(self):
        cfg = self._config(expert_parallel_size=4)

        self.assertEqual(cfg.tensor_parallel_size, 1)
        self.assertEqual(cfg.expert_parallel_size, 4)
        self.assertEqual(cfg.parallel_world_size, 4)
        self.assertEqual(cfg.expert_parallel_backend, "all_reduce")
        self.assertEqual(cfg.expert_placement_policy, "contiguous")

    def test_expert_parallel_rejects_tp_hybrid_v1(self):
        with self.assertRaisesRegex(ValueError, "does not support TP\\+EP hybrid"):
            self._config(tensor_parallel_size=2, expert_parallel_size=2)

    def test_distributed_master_port_validation(self):
        cfg = self._config(expert_parallel_size=2, distributed_master_port=29501)

        self.assertEqual(cfg.distributed_master_port, 29501)
        with self.assertRaisesRegex(ValueError, "distributed_master_port must be in"):
            self._config(expert_parallel_size=2, distributed_master_port=70000)

    def test_expert_parallel_rejects_decode_graph_aliases(self):
        with self.assertRaisesRegex(ValueError, "expert_parallel_size > 1 disables decode_cuda_graph"):
            self._config(expert_parallel_size=2, decode_graph=True)

        with self.assertRaisesRegex(ValueError, "expert_parallel_size > 1 disables decode_cuda_graph"):
            self._config(
                expert_parallel_size=2,
                vllm_sparse_method="omnikv",
                omnikv_decode_graph=True,
            )

    def test_expert_parallel_rejects_legacy_forced_decode_graph(self):
        with self.assertRaisesRegex(ValueError, "expert_parallel_size > 1 disables decode_cuda_graph"):
            self._config(
                expert_parallel_size=2,
                vllm_sparse_method="deltakv_less_memory_cudagraph",
                allow_missing_deltakv_path=True,
            )

    def test_parallel_context_maps_ep_world_without_tp_shard(self):
        cfg = self._config(expert_parallel_size=4)
        ctx = build_parallel_context(cfg, global_rank=3)

        self.assertEqual(ctx.global_rank, 3)
        self.assertEqual(ctx.global_world_size, 4)
        self.assertEqual(ctx.tp_size, 1)
        self.assertEqual(ctx.tp_rank, 0)
        self.assertEqual(ctx.ep_size, 4)
        self.assertEqual(ctx.ep_rank, 3)

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
        cfg = self._config(expert_parallel_size=4)
        cache_manager = DummyCacheManager(cfg, rank=2, world_size=cfg.tensor_parallel_size)

        self.assertEqual(cache_manager.world_size, 1)
        self.assertEqual(cache_manager.num_kv_heads, cfg.hf_config.num_key_value_heads)

    def test_cache_manager_uses_parallel_context_local_rank_for_device(self):
        cfg = self._config(expert_parallel_size=4)
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


if __name__ == "__main__":
    unittest.main()
