from transformers.models.qwen2.modeling_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.llama.modeling_llama import LlamaConfig
from deltakv.utils.log import logger, log_once


def parse_full_attn_layers(full_attn_layers):
    if full_attn_layers is None:
        return []
    if isinstance(full_attn_layers, str):
        if not full_attn_layers.strip():
            return []
        return [int(part.strip()) for part in full_attn_layers.split(',') if part.strip()]
    return [int(part) for part in full_attn_layers]


class CustomConfigMixin:
    """
    提供简便的方法来批量更新自定义参数。
    """
    def __init__(
        self,
        kv_compressed_size=128,
        seq_chunk_size=1,
        k_neighbors=None,
        layer_chunk_size=1,
        recon_mode='delta_in_latent',
        ref_mode='avg',
        use_nonlinear_compressor=True,
        compressor_intermediate_size=2048,
        compressor_down_type='auto',
        compressor_up_type='auto',
        compressor_down_intermediate_size=-1,
        compressor_up_intermediate_size=-1,
        collect_kv_before_rope=True,
        compressor_linear_bias=True,
        split_kv=False,
        cluster_metric='l2',
        cluster_on_kv=True,
        cluster_ratio=0.1,
        # Dynamic stride schedule for clustering prototypes. When >0, stride increases
        # roughly as: stride(pos) = base_stride + stride_alpha * (pos - sink).
        # Default 0.0 keeps the legacy fixed-stride behavior.
        stride_alpha: float = 0.0,
        cluster_temp=10.0,
        cluster_soft_assignment=False,
        tail_token_size=128,
        num_recent_tokens=128,
        full_attn_layers='0,1,2,3,8,16,22',
        num_top_tokens=1024,
        num_top_tokens_in_prefill=8192,
        num_sink_tokens=8,
        omnikv_score_method='last',
        deltakv_use_omnikv_selection=True,
        deltasnapkv_total_budget=-1.0,
        deltasnapkv_ref_budget=-1.0,
        snapkv_num_full_layers=0,
        use_compression=False,
        use_cluster=True,
        chunk_prefill_size=100_000_000,
        snapkv_window_size=4,
        pool_kernel_size=1,
        chunk_prefill_accel_omnikv=False,
        pyramidkv_start_layer=2,
        pyramidkv_start_ratio=1.0,
        pyramidkv_least_layer=None,
        pyramidkv_least_ratio=0.01,
        kv_quant_bits=0,
        **kwargs
    ):
        # 初始化自定义属性
        # 这个地方好像也只能设置一下默认值了，主要目的是有语法提示。
        self.kv_compressed_size = kv_compressed_size
        self.seq_chunk_size = seq_chunk_size
        self.k_neighbors = k_neighbors
        self.layer_chunk_size = layer_chunk_size
        self.recon_mode = recon_mode
        self.ref_mode = ref_mode
        self.use_nonlinear_compressor = use_nonlinear_compressor
        self.compressor_intermediate_size = compressor_intermediate_size
        self.compressor_down_type = compressor_down_type
        self.compressor_up_type = compressor_up_type
        self.compressor_down_intermediate_size = compressor_down_intermediate_size
        self.compressor_up_intermediate_size = compressor_up_intermediate_size
        self.collect_kv_before_rope = collect_kv_before_rope
        self.compressor_linear_bias = compressor_linear_bias
        self.split_kv = split_kv
        self.cluster_metric = cluster_metric
        self.cluster_on_kv = cluster_on_kv
        self.cluster_ratio = cluster_ratio
        self.stride_alpha = stride_alpha
        self.cluster_temp = cluster_temp
        self.cluster_soft_assignment = cluster_soft_assignment
        self.tail_token_size = tail_token_size
        self.tail_token_size = num_recent_tokens
        self.full_attn_layers = parse_full_attn_layers(full_attn_layers)
        self.num_top_tokens = num_top_tokens
        self.num_top_tokens_in_prefill = num_top_tokens_in_prefill
        self.num_sink_tokens = num_sink_tokens
        self.omnikv_score_method = omnikv_score_method
        self.deltakv_use_omnikv_selection = deltakv_use_omnikv_selection
        self.deltasnapkv_total_budget = deltasnapkv_total_budget
        self.deltasnapkv_ref_budget = deltasnapkv_ref_budget
        self.snapkv_num_full_layers = snapkv_num_full_layers
        self.use_compression = use_compression
        self.use_cluster = use_cluster
        self.chunk_prefill_size = chunk_prefill_size
        self.snapkv_window_size = snapkv_window_size
        self.pool_kernel_size = pool_kernel_size
        self.chunk_prefill_accel_omnikv = chunk_prefill_accel_omnikv
        self.pyramidkv_start_layer = pyramidkv_start_layer
        self.pyramidkv_start_ratio = pyramidkv_start_ratio
        self.pyramidkv_least_layer = pyramidkv_least_layer
        self.pyramidkv_least_ratio = pyramidkv_least_ratio
        self.kv_quant_bits = kv_quant_bits
        
        # 调用 MRO 中的下一个 __init__ (Qwen2Config 或 LlamaConfig)
        super().__init__(**kwargs)

    def finalize_cluster_args(self, *, warn_on_legacy_k_neighbors: bool = False):
        if not getattr(self, "use_cluster", False):
            return

        if getattr(self, "k_neighbors", None) is None:
            legacy_k = getattr(self, "seq_chunk_size", None)
            if legacy_k is None:
                legacy_k = 1
            self.k_neighbors = int(legacy_k)
            if warn_on_legacy_k_neighbors:
                log_once(
                    f"Cluster config missing `k_neighbors`; falling back to legacy "
                    f"`seq_chunk_size={self.k_neighbors}`. Please pass `k_neighbors` explicitly.",
                    "WARNING",
                )

    def get_cluster_k_neighbors(self) -> int:
        self.finalize_cluster_args(warn_on_legacy_k_neighbors=False)
        return max(1, int(self.k_neighbors))

    def set_extra_args(self, **kwargs):
        legacy_keys = {
            "use_nonlinear_compressor",
            "compressor_intermediate_size",
            "compressor_linear_bias",
        }
        directional_keys = {
            "compressor_down_type",
            "compressor_up_type",
            "compressor_down_intermediate_size",
            "compressor_up_intermediate_size",
        }
        # 兼容旧版本：旧版本仅用上面 3 个参数控制“对称”的 compressor/decompressor。
        # 若用户只传旧参数（未显式传新参数），则重置方向相关配置为默认值，让旧参数继续完全接管。
        if legacy_keys.intersection(kwargs) and not directional_keys.intersection(kwargs):
            defaults = {
                "compressor_down_type": "auto",
                "compressor_up_type": "auto",
                "compressor_down_intermediate_size": -1,
                "compressor_up_intermediate_size": -1,
            }
            for k, v in defaults.items():
                if hasattr(self, k):
                    kwargs[k] = v

        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == 'full_attn_layers':
                    value = parse_full_attn_layers(value)
                setattr(self, key, value)
                print(f"[Config] Setting {key} = {value}")
            else:
                logger.error(f'There is NO {key} in Custom Config!')
                if key == 'num_recent_tokens':
                    logger.warning(f'为了保持兼容性，{key} 可以被映射到 tail_token_size')
                    self.tail_token_size = value

    def set_infer_args(self, **kwargs):
        self.set_extra_args(**kwargs)
        self.finalize_cluster_args(warn_on_legacy_k_neighbors="k_neighbors" not in kwargs)


class KVQwen2Config(CustomConfigMixin, Qwen2Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class KVQwen3Config(CustomConfigMixin, Qwen3Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class KVLlamaConfig(CustomConfigMixin, LlamaConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == '__main__':
    pass
