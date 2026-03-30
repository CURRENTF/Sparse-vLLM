from typing import Optional

import torch
from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM, Unpack

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.all_origin_residual_quant_cache import (
    AllOriginResidualQuantClusterCompressedKVCache,
)
from deltakv.modeling.qwen2.qwen2_with_compress_inference import Qwen2KVCompress as _BaseQwen2KVCompress


class Qwen2AllOriginResidualQuant(_BaseQwen2KVCompress):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        assert config.use_cluster, "AllOriginResidualQuant only supports use_cluster=True"

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[AllOriginResidualQuantClusterCompressedKVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        del attention_mask, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, cache_position, logits_to_keep, kwargs
        assert input_ids is not None
        assert input_ids.shape[0] == 1
        assert use_cache, "Inference model must use cache"

        if not isinstance(past_key_values, AllOriginResidualQuantClusterCompressedKVCache):
            past_key_values = AllOriginResidualQuantClusterCompressedKVCache(config=self.config)

        outputs = None
        for chunk_input_ids in input_ids.split(self.config.chunk_prefill_size, dim=-1):
            outputs = super(_BaseQwen2KVCompress, self).forward(
                chunk_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

        return outputs
