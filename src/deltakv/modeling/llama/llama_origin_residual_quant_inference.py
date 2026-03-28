from typing import Optional, Union

import torch
from transformers.models.llama.modeling_llama import KwargsForCausalLM, Unpack

from deltakv.configs.model_config_cls import KVLlamaConfig
from deltakv.modeling.llama.llama_with_compress_inference import LlamaKVCompress as _BaseLlamaKVCompress
from deltakv.modeling.origin_residual_quant_cache import (
    OriginResidualQuantClusterCompressedKVCache,
    OriginResidualQuantCompressedKVCache,
)


class LlamaOriginResidualQuant(_BaseLlamaKVCompress):
    def __init__(self, config: KVLlamaConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[
            Union[OriginResidualQuantCompressedKVCache, OriginResidualQuantClusterCompressedKVCache]
        ] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        del attention_mask, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, cache_position, logits_to_keep, kwargs
        assert input_ids is not None
        assert input_ids.shape[0] == 1
        assert use_cache, "Inference model must use cache"

        if not isinstance(
            past_key_values,
            (OriginResidualQuantCompressedKVCache, OriginResidualQuantClusterCompressedKVCache),
        ):
            if self.config.use_cluster:
                past_key_values = OriginResidualQuantClusterCompressedKVCache(config=self.config)
            else:
                past_key_values = OriginResidualQuantCompressedKVCache(config=self.config)

        outputs = None
        for chunk_input_ids in input_ids.split(self.config.chunk_prefill_size, dim=-1):
            outputs = super(_BaseLlamaKVCompress, self).forward(
                chunk_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

        return outputs
