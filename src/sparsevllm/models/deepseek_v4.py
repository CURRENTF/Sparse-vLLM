from __future__ import annotations

import re

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model

from sparsevllm.layers.embed_head import ParallelLMHead
from sparsevllm.models.deepseek_v4_native import DeepseekV4Model as NativeDeepseekV4Model
from sparsevllm.utils.context import get_context
from sparsevllm.utils.weight_target import WeightTarget


_EXPERT_SOURCE_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$"
)
_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.(gate|down|up)\.expert_weight$"
)


class DeepseekV4ForCausalLM(nn.Module):
    """DeepSeek V4 architecture adapter used by the tiny-random reference path.

    The formal FP4 DPA+EP runtime replaces the Transformers layers with native
    Sparse-vLLM operators. Keeping this reference adapter exact gives that path
    a deterministic end-to-end correctness oracle without reading checkpoint
    tensors during development.
    """

    special_weight_loaders = (".expert_weight",)

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tiny_random = bool(getattr(config, "sparsevllm_tiny_random", False))
        self.model = DeepseekV4Model(config) if self.tiny_random else NativeDeepseekV4Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self._seq_caches: dict[int, DynamicCache] = {}
        self._skipped_expert_weights: set[str] = set()
        self._skipped_expert_scales: set[str] = set()

    @staticmethod
    def scale_key_for_weight(weight_key: str) -> str | None:
        if not weight_key.endswith(".weight"):
            return None
        return weight_key[: -len(".weight")] + ".scale"

    @staticmethod
    def weight_key_for_scale(scale_key: str) -> str | None:
        if not scale_key.endswith(".scale"):
            return None
        return scale_key[: -len(".scale")] + ".weight"

    def map_weight_name(self, source_weight_name: str) -> str | None:
        expert_match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if expert_match is not None:
            layer_idx, expert_id, source_projection = expert_match.groups()
            experts = self.model.layers[int(layer_idx)].ffn.experts
            if not experts.is_local_expert(int(expert_id)):
                self._skipped_expert_weights.add(source_weight_name)
                return None
            projection = {"w1": "gate", "w2": "down", "w3": "up"}[
                source_projection
            ]
            return (
                f"model.layers.{layer_idx}.ffn.experts.{expert_id}."
                f"{projection}.expert_weight"
            )
        if source_weight_name.startswith(("mtp.", "model.mtp.", "nextn.")):
            return None
        if source_weight_name.startswith("embed."):
            mapped = "model." + source_weight_name
        elif source_weight_name.startswith("layers."):
            mapped = "model." + source_weight_name
        elif source_weight_name.startswith("norm."):
            mapped = "model." + source_weight_name
        elif source_weight_name.startswith("head."):
            mapped = "lm_" + source_weight_name
        elif source_weight_name.startswith("hc_head_"):
            mapped = "model." + source_weight_name
        else:
            mapped = source_weight_name
        replacements = (
            (".attn.wq_a.", ".attn.q_a_proj."),
            (".attn.wq_b.", ".attn.q_b_proj."),
            (".attn.wkv.", ".attn.kv_proj."),
            (".attn.wo_a.", ".attn.o_a_proj."),
            (".attn.wo_b.", ".attn.o_b_proj."),
            (".attn.q_norm.", ".attn.q_a_norm."),
            (".attn.attn_sink", ".attn.sinks"),
            (".attn.compressor.wkv.", ".attn.compressor.kv_proj."),
            (".attn.compressor.wgate.", ".attn.compressor.gate_proj."),
            (".attn.compressor.ape", ".attn.compressor.position_bias"),
            (".attn.compressor.norm.", ".attn.compressor.kv_norm."),
            (
                ".attn.indexer.compressor.wkv.",
                ".attn.compressor.indexer.kv_proj.",
            ),
            (
                ".attn.indexer.compressor.wgate.",
                ".attn.compressor.indexer.gate_proj.",
            ),
            (
                ".attn.indexer.compressor.ape",
                ".attn.compressor.indexer.position_bias",
            ),
            (
                ".attn.indexer.compressor.norm.",
                ".attn.compressor.indexer.kv_norm.",
            ),
            (
                ".attn.indexer.weights_proj.",
                ".attn.compressor.indexer.scorer.weights_proj.",
            ),
            (
                ".attn.indexer.wq_b.",
                ".attn.compressor.indexer.q_b_proj.",
            ),
            (".hc_attn_fn", ".hc_attn.fn"),
            (".hc_attn_base", ".hc_attn.base"),
            (".hc_attn_scale", ".hc_attn.scale"),
            (".hc_ffn_fn", ".hc_ffn.fn"),
            (".hc_ffn_base", ".hc_ffn.base"),
            (".hc_ffn_scale", ".hc_ffn.scale"),
            (".hc_head_fn", ".hc_head.fn"),
            (".hc_head_base", ".hc_head.base"),
            (".hc_head_scale", ".hc_head.scale"),
        )
        for source, target in replacements:
            mapped = mapped.replace(source, target)
        return mapped

    def resolve_special_weight(self, target_weight_name: str) -> WeightTarget | None:
        match = _EXPERT_TARGET_RE.match(target_weight_name)
        if match is None:
            return None
        layer_idx, expert_id, projection = match.groups()
        return WeightTarget(
            self.model.layers[int(layer_idx)].ffn.experts,
            (int(expert_id), projection),
        )

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        target = self.resolve_special_weight(target_weight_name)
        if target is None:
            return 0
        expert_id, projection = target.shard_id
        target.module.load_expert_weight(
            expert_id, projection, loaded_weight, loaded_scale
        )
        return 1

    def record_skipped_weight(
        self,
        source_weight_name: str,
        loaded_weight_shape,
        loaded_weight_dtype,
        loaded_scale_shape,
        loaded_scale_dtype,
    ) -> None:
        del loaded_weight_shape, loaded_weight_dtype, loaded_scale_shape, loaded_scale_dtype
        if _EXPERT_SOURCE_RE.match(source_weight_name) is None:
            if source_weight_name.startswith(("mtp.", "model.mtp.", "nextn.")):
                return
            raise ValueError(f"Unexpectedly skipped DeepSeek V4 tensor {source_weight_name!r}.")
        self._skipped_expert_weights.add(source_weight_name)
        self._skipped_expert_scales.add(
            source_weight_name[: -len(".weight")] + ".scale"
        )

    def _forward_sequence(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_id: int,
    ) -> torch.Tensor:
        cache = self._seq_caches.get(seq_id)
        if cache is None and self.tiny_random:
            if int(positions[0]) != 0:
                raise RuntimeError(
                    f"DeepSeek V4 sequence {seq_id} has no cache at position "
                    f"{int(positions[0])}."
                )
            cache = DynamicCache(config=self.config)
            self._seq_caches[seq_id] = cache
        if self.tiny_random:
            output = self.model(
                input_ids=input_ids.unsqueeze(0),
                position_ids=positions.unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            return output.last_hidden_state.squeeze(0)
        raise RuntimeError("Native DeepSeek V4 uses cache-manager rows, not per-sequence caches.")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        context = get_context()
        seqs = list(context.seqs or ())
        if not seqs:
            raise RuntimeError("DeepSeek V4 forward requires active engine sequences.")
        if context.is_prefill:
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None or int(cu_seqlens.numel()) != len(seqs) + 1:
                raise RuntimeError("DeepSeek V4 prefill requires one cu_seqlens entry per sequence.")
            if not self.tiny_random:
                rows = context.cache_manager.get_layer_batch_states(0).req_indices
                outputs = []
                for index in range(len(seqs)):
                    start = int(cu_seqlens[index])
                    end = int(cu_seqlens[index + 1])
                    outputs.append(
                        self.model(
                            input_ids[start:end].unsqueeze(0),
                            positions[start:end].unsqueeze(0),
                            rows[index : index + 1],
                        ).squeeze(0)
                    )
                return torch.cat(outputs, dim=0)
            outputs = []
            for index, seq in enumerate(seqs):
                start = int(cu_seqlens[index])
                end = int(cu_seqlens[index + 1])
                outputs.append(
                    self._forward_sequence(
                        input_ids[start:end],
                        positions[start:end],
                        int(seq.seq_id),
                    )
                )
            return torch.cat(outputs, dim=0)
        if self.tiny_random and int(input_ids.numel()) != len(seqs):
            raise RuntimeError(
                "DeepSeek V4 decode requires exactly one token per active sequence, "
                f"got tokens={input_ids.numel()} sequences={len(seqs)}."
            )
        if not self.tiny_random:
            rows = context.cache_manager.get_layer_batch_states(0).req_indices
            return self.model(input_ids.unsqueeze(1), positions.unsqueeze(1), rows).squeeze(1)
        return torch.cat(
            [
                self._forward_sequence(
                    input_ids[index : index + 1],
                    positions[index : index + 1],
                    int(seq.seq_id),
                )
                for index, seq in enumerate(seqs)
            ],
            dim=0,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.inference_mode()
    def warmup_moe(self, num_tokens: int = 1) -> None:
        if self.tiny_random:
            return
        for layer in self.model.layers:
            layer.ffn.experts.prepare_for_inference()

    def validate_loaded_weights(self, loaded_parameter_names: set[str]) -> None:
        if self.tiny_random:
            return
        expert_parameters = {
            name
            for name, _ in self.named_parameters()
            if name.endswith(".ffn.experts.w13_weight")
            or name.endswith(".ffn.experts.w2_weight")
        }
        missing = sorted(
            {name for name, _ in self.named_parameters()}
            - expert_parameters
            - loaded_parameter_names
        )
        if missing:
            raise ValueError(f"Missing replicated DeepSeek V4 parameters: {missing[:8]}.")
        for layer in self.model.layers:
            layer.ffn.experts.validate_loaded_weights()

    def free_sequence_cache(self, seq_id: int) -> None:
        self._seq_caches.pop(int(seq_id), None)

    def reset_after_warmup(self) -> None:
        self._seq_caches.clear()
