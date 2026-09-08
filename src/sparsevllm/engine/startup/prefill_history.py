from __future__ import annotations

import copy

import torch

from sparsevllm.engine.cache_manager.standard import StandardCacheManager
from sparsevllm.engine.cache_manager.storage import (
    ExplicitKVStorage,
    HeterogeneousExplicitKVStorage,
    MlaLatentStorage,
)
from sparsevllm.engine.runtime_state import RuntimeState
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.sparse_controller import SparseController
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.utils.context import reset_context
from sparsevllm.utils.log import logger

from .profiling import StartupMemoryProfiler


class PrefillHistoryCacheManager(StandardCacheManager):
    """Startup-only dense cache with synthetic history shared across layers."""

    def allocate_kv_cache(self) -> None:
        slots = int(self.config.max_model_len) - 1
        self.config.num_kvcache_slots = slots
        storage = self.attention_cache_storage
        if isinstance(storage, HeterogeneousExplicitKVStorage):
            caches = {
                shape: torch.zeros(2, slots, *shape, dtype=storage.dtype, device=self.device)
                for shape in dict.fromkeys(storage.layer_shapes)
            }
            storage.kv_cache = [caches[shape] for shape in storage.layer_shapes]
        else:
            storage.allocate(num_layers=1, num_slots=slots, device=self.device)
            for tensor in storage.accounting_tensors():
                tensor.zero_()
            # Layers execute sequentially and overwrite only the current chunk.
            # The synthetic history stays zero, so one physical layer suffices.
            if isinstance(storage, ExplicitKVStorage):
                storage.kv_cache = storage.kv_cache.expand(
                    -1, self.num_kv_layers, -1, -1, -1,
                )
            elif isinstance(storage, MlaLatentStorage):
                storage.latent_cache = storage.latent_cache.expand(
                    self.num_kv_layers, -1, -1, -1,
                )
                storage.rope_cache = storage.rope_cache.expand(
                    self.num_kv_layers, -1, -1, -1,
                )
        self.kv_cache = getattr(storage, "kv_cache", None)

    def seed_history(self, seq: Sequence) -> None:
        self._allocate(seq.seq_id, int(seq.num_prefilled_tokens))


@torch.inference_mode()
def profile_prefill_history(runner):
    """Measure one maximum-context suffix through the ordinary model forward.

    Dense synthetic history covers history-dependent attention allocations.
    Sparse compression/scoring and concurrent long requests remain covered only
    by the ordinary startup workload and the utilization headroom.
    """
    config = copy.copy(runner.config)
    context_len = int(config.max_model_len) - 1
    chunk_size = min(int(config.engine_prefill_chunk_size), int(config.max_num_batched_tokens))
    if context_len <= chunk_size:
        return None
    config.sparse_method = ""
    config.prefill_sparse_method = None
    config.enable_prefix_caching = False
    config.enable_prefix_cache_offload = False
    config.resolved_prefix_cache_mode = "disabled"
    config.startup_cache_phase = "profiling"
    manager = PrefillHistoryCacheManager(config, runner.parallel_context)
    controller = SparseController(config, manager)
    runtime = RuntimeState(config, manager, runner.recurrent_state_manager)
    seq = Sequence([0] * context_len, SamplingParams(max_tokens=1, temperature=0.0))
    seq.num_prefilled_tokens = context_len - chunk_size
    seq.current_chunk_size = chunk_size
    manager.seed_history(seq)

    model = runner.model.model
    previous = runner.cache_manager, runner.sparse_controller, runner.runtime_state
    previous_controller = model.sparse_controller
    profiler = StartupMemoryProfiler(runner.platform, runner.device)
    try:
        runner.cache_manager, runner.sparse_controller, runner.runtime_state = (
            manager, controller, runtime,
        )
        model.sparse_controller = controller
        controller.set_modules(model.layers)
        logger.info(
            "Startup profile phase=prefill_history context={} history={} chunk={} batch=1.",
            context_len, seq.num_prefilled_tokens, chunk_size,
        )
        # Keep the synthetic cache alive across both snapshots so its persistent
        # bytes are not counted as transient model memory.
        profiler.begin("prefill_history")
        runner.run([seq], is_prefill=True)
        result = profiler.finish("prefill_history")
        logger.info(
            "Startup prefill_history transient_peak={:.2f} GiB.",
            result.measurement.transient_peak_bytes / 1024**3,
        )
        return result
    finally:
        runner.cache_manager, runner.sparse_controller, runner.runtime_state = previous
        model.sparse_controller = previous_controller
        reset_context()
        runtime.free_seq(seq.seq_id)
        runtime.reset_after_warmup()
        release_bindings = getattr(runner.model, "release_cache_runtime_bindings", None)
        if callable(release_bindings):
            release_bindings(manager)
