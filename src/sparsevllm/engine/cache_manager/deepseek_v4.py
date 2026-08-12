from __future__ import annotations

import math

import torch

from sparsevllm.utils.log import logger

from .standard import StandardCacheManager


class DeepseekV4CacheManager(StandardCacheManager):
    """Row-indexed DSV4 sliding and compressed KV storage.

    DSV4 does not use a conventional K/V pair per source token.  Every layer
    keeps a 128-token shared-KV ring, while CSA and HCA layers additionally
    retain one compressed entry per 4 or 128 source tokens.  Keeping these
    tensors here makes request ownership and CUDA-graph metadata follow the
    same cache-manager contract as the other runtimes.
    """

    def allocate_kv_cache(self) -> None:
        config = self.config
        hf_config = self.hf_config
        rows = int(self.max_buffer_rows)
        max_len = int(self.max_model_len)
        window = int(hf_config.sliding_window)
        head_dim = int(hf_config.head_dim)
        index_dim = int(hf_config.index_head_dim)
        csa_ratio = int(hf_config.compress_rates["compressed_sparse_attention"])
        hca_ratio = int(hf_config.compress_rates["heavily_compressed_attention"])
        layer_types = tuple(hf_config.layer_types)
        self.csa_layers = tuple(i for i, kind in enumerate(layer_types) if kind == "compressed_sparse_attention")
        self.hca_layers = tuple(i for i, kind in enumerate(layer_types) if kind == "heavily_compressed_attention")
        self._csa_slot = {layer: i for i, layer in enumerate(self.csa_layers)}
        self._hca_slot = {layer: i for i, layer in enumerate(self.hca_layers)}
        self.csa_ratio = csa_ratio
        self.hca_ratio = hca_ratio
        self.sliding_window = window
        self.max_csa_entries = math.ceil(max_len / csa_ratio)
        self.max_hca_entries = math.ceil(max_len / hca_ratio)
        dtype = hf_config.torch_dtype

        shapes = {
            "raw_kv": (len(layer_types), rows, window, head_dim),
            "csa_kv": (len(self.csa_layers), rows, self.max_csa_entries, head_dim),
            "csa_index": (len(self.csa_layers), rows, self.max_csa_entries, index_dim),
            "hca_kv": (len(self.hca_layers), rows, self.max_hca_entries, head_dim),
            "csa_ring_kv": (len(self.csa_layers), rows, csa_ratio, 2 * head_dim),
            "csa_ring_gate": (len(self.csa_layers), rows, csa_ratio, 2 * head_dim),
            "csa_overlap_kv": (len(self.csa_layers), rows, csa_ratio, head_dim),
            "csa_overlap_gate": (len(self.csa_layers), rows, csa_ratio, head_dim),
            "index_ring_kv": (len(self.csa_layers), rows, csa_ratio, 2 * index_dim),
            "index_ring_gate": (len(self.csa_layers), rows, csa_ratio, 2 * index_dim),
            "index_overlap_kv": (len(self.csa_layers), rows, csa_ratio, index_dim),
            "index_overlap_gate": (len(self.csa_layers), rows, csa_ratio, index_dim),
            "hca_ring_kv": (len(self.hca_layers), rows, hca_ratio, head_dim),
            "hca_ring_gate": (len(self.hca_layers), rows, hca_ratio, head_dim),
        }
        element_size = torch.empty((), dtype=dtype).element_size()
        tensor_bytes = sum(math.prod(shape) for shape in shapes.values()) * element_size
        metadata_bytes = 2 * rows * max_len * torch.empty((), dtype=torch.int32).element_size()
        cache_bytes = tensor_bytes + metadata_bytes
        free, total = self.platform.get_available_memory(self.device.index or 0)
        budget = max(0, int(total * float(config.gpu_memory_utilization)) - int(total - free))
        if cache_bytes > budget:
            raise RuntimeError(
                "DeepSeek V4 cache does not fit the configured GPU budget: "
                f"required={cache_bytes / 2**30:.2f} GiB available={budget / 2**30:.2f} GiB, "
                f"max_model_len={max_len}, max_num_seqs_in_gpu={rows}. Reduce one of these limits."
            )

        def empty(name: str) -> torch.Tensor:
            return torch.empty(shapes[name], dtype=dtype, device=self.device)

        self.raw_kv = empty("raw_kv")
        self.csa_kv = empty("csa_kv")
        self.csa_index = empty("csa_index")
        self.hca_kv = empty("hca_kv")
        self.csa_ring_kv = empty("csa_ring_kv")
        self.csa_ring_gate = empty("csa_ring_gate")
        self.csa_overlap_kv = empty("csa_overlap_kv")
        self.csa_overlap_gate = empty("csa_overlap_gate")
        self.index_ring_kv = empty("index_ring_kv")
        self.index_ring_gate = empty("index_ring_gate")
        self.index_overlap_kv = empty("index_overlap_kv")
        self.index_overlap_gate = empty("index_overlap_gate")
        self.hca_ring_kv = empty("hca_ring_kv")
        self.hca_ring_gate = empty("hca_ring_gate")
        self.reset_deepseek_v4_cache()

        # StandardCacheManager still owns admission and request-to-row metadata.
        # Its token slots are bookkeeping-only for DSV4; attention reads the
        # row-indexed tensors above.
        config.num_kvcache_slots = rows * max_len
        self.kv_cache = torch.empty(0, dtype=dtype, device=self.device)
        logger.info(
            "DeepSeek V4 cache: {:.2f} GiB, rows={}, max_len={}, CSA={}, HCA={}.",
            cache_bytes / 2**30,
            rows,
            max_len,
            len(self.csa_layers),
            len(self.hca_layers),
        )

    def reset_deepseek_v4_cache(self) -> None:
        for tensor in (
            self.raw_kv,
            self.csa_kv,
            self.csa_index,
            self.hca_kv,
            self.csa_ring_kv,
            self.csa_overlap_kv,
            self.index_ring_kv,
            self.index_overlap_kv,
            self.hca_ring_kv,
        ):
            tensor.zero_()
        for tensor in (
            self.csa_ring_gate,
            self.csa_overlap_gate,
            self.index_ring_gate,
            self.index_overlap_gate,
            self.hca_ring_gate,
        ):
            tensor.fill_(float("-inf"))

    def reset_after_warmup(self) -> None:
        super().reset_after_warmup()
        self.reset_deepseek_v4_cache()

    def free_seq(self, seq_id: int) -> None:
        row = self.seq_id_to_row.get(int(seq_id))
        super().free_seq(seq_id)
        if row is None:
            return
        self.csa_overlap_kv[:, row].zero_()
        self.csa_overlap_gate[:, row].fill_(float("-inf"))
        self.index_overlap_kv[:, row].zero_()
        self.index_overlap_gate[:, row].fill_(float("-inf"))

    def csa_slot(self, layer_idx: int) -> int:
        return self._csa_slot[int(layer_idx)]

    def hca_slot(self, layer_idx: int) -> int:
        return self._hca_slot[int(layer_idx)]

    def compressed_capacity(self, ratio: int, positions: torch.Tensor) -> int:
        static_len = self._decode_static_max_context_len
        if static_len is not None:
            return min(math.ceil(int(static_len) / int(ratio)), math.ceil(self.max_model_len / int(ratio)))
        return min(math.ceil((int(positions.max()) + 1) / int(ratio)), math.ceil(self.max_model_len / int(ratio)))

    def decode_cuda_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        return [
            self.raw_kv,
            self.csa_kv,
            self.csa_index,
            self.hca_kv,
            self.csa_ring_kv,
            self.csa_ring_gate,
            self.csa_overlap_kv,
            self.csa_overlap_gate,
            self.index_ring_kv,
            self.index_ring_gate,
            self.index_overlap_kv,
            self.index_overlap_gate,
            self.hca_ring_kv,
            self.hca_ring_gate,
        ]
