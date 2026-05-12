from transformers.models.qwen2.modeling_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.llama.modeling_llama import LlamaConfig
from deltakv.configs.runtime_params import normalize_runtime_params
from deltakv.utils.log import logger


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
        compressor_token_group_size=1,
        deltakv_neighbor_count=1,
        layer_chunk_size=1,
        recon_mode='delta_in_latent',
        ref_mode='last',
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
        deltakv_cache_impl="delta_compressed_latent_wo_full",
        chunk_prefill_size=100_000_000,
        snapkv_window_size=4,
        pool_kernel_size=1,
        chunk_prefill_accel_omnikv=False,
        pyramidkv_start_layer=2,
        pyramidkv_start_ratio=1.0,
        pyramidkv_least_layer=None,
        pyramidkv_least_ratio=0.01,
        kv_quant_bits=0,
        visual_token_prune_only=None,
        visual_token_keep_ratio=None,
        **kwargs
    ):
        # Saved compressor checkpoints may still contain the old config schema.
        # Migrate those artifact fields internally so historical weights remain
        # loadable for regression checks; runtime/API parameters are still
        # rejected by normalize_runtime_params().
        if "seq_chunk_size" in kwargs:
            legacy_value = kwargs.pop("seq_chunk_size")
            if compressor_token_group_size != 1 and compressor_token_group_size != legacy_value:
                raise ValueError(
                    "Conflicting checkpoint config fields: `seq_chunk_size` and "
                    "`compressor_token_group_size` differ."
                )
            compressor_token_group_size = legacy_value
        if "k_neighbors" in kwargs:
            legacy_value = kwargs.pop("k_neighbors")
            if deltakv_neighbor_count != 1 and deltakv_neighbor_count != legacy_value:
                raise ValueError(
                    "Conflicting checkpoint config fields: `k_neighbors` and "
                    "`deltakv_neighbor_count` differ."
                )
            deltakv_neighbor_count = legacy_value
        legacy_visual_keys = {
            "deltakv_visual_compress_only": "visual_token_prune_only",
            "deltakv_visual_keep_ratio": "visual_token_keep_ratio",
        }
        legacy_visual_found = sorted(key for key in kwargs if key in legacy_visual_keys)
        if legacy_visual_found:
            details = ", ".join(f"`{key}` -> `{legacy_visual_keys[key]}`" for key in legacy_visual_found)
            raise ValueError(
                "Legacy LLaVA visual runtime config keys are no longer accepted. "
                f"Use the split semantic names instead: {details}."
            )

        if visual_token_prune_only is None:
            visual_token_prune_only = False

        if visual_token_keep_ratio is None:
            visual_token_keep_ratio = 1.0

        # 初始化自定义属性
        # 这个地方好像也只能设置一下默认值了，主要目的是有语法提示。
        self.kv_compressed_size = kv_compressed_size
        self.compressor_token_group_size = compressor_token_group_size
        self.deltakv_neighbor_count = deltakv_neighbor_count
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
        self.num_recent_tokens = num_recent_tokens
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
        self.deltakv_cache_impl = deltakv_cache_impl
        self.chunk_prefill_size = chunk_prefill_size
        self.snapkv_window_size = snapkv_window_size
        self.pool_kernel_size = pool_kernel_size
        self.chunk_prefill_accel_omnikv = chunk_prefill_accel_omnikv
        self.pyramidkv_start_layer = pyramidkv_start_layer
        self.pyramidkv_start_ratio = pyramidkv_start_ratio
        self.pyramidkv_least_layer = pyramidkv_least_layer
        self.pyramidkv_least_ratio = pyramidkv_least_ratio
        self.kv_quant_bits = kv_quant_bits
        self.visual_token_prune_only = visual_token_prune_only
        self.visual_token_keep_ratio = visual_token_keep_ratio
        
        # 调用 MRO 中的下一个 __init__ (Qwen2Config 或 LlamaConfig)
        super().__init__(**kwargs)

    def finalize_cluster_args(self):
        if not getattr(self, "use_cluster", False):
            return

        if getattr(self, "deltakv_neighbor_count", None) is None:
            raise ValueError(
                "`deltakv_neighbor_count` is required when `use_cluster=True`. "
                "`compressor_token_group_size` no longer doubles as the cluster neighbor count."
            )

    def get_cluster_neighbor_count(self) -> int:
        self.finalize_cluster_args()
        return max(1, int(self.deltakv_neighbor_count))

    def set_native_args(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == 'full_attn_layers':
                    value = parse_full_attn_layers(value)
                setattr(self, key, value)
                if key == 'num_recent_tokens':
                    self.tail_token_size = value
                print(f"[Config] Setting {key} = {value}")
            else:
                logger.error(f'There is NO {key} in Custom Config!')
                if key == 'num_recent_tokens':
                    self.tail_token_size = value

    def set_extra_args(self, **kwargs):
        normalized_params = normalize_runtime_params(kwargs, backend="hf")
        for warning in normalized_params.warnings:
            logger.info(f"Runtime parameter normalization: {warning}")
        self.set_native_args(**normalized_params.infer_config)

    def set_infer_args(self, **kwargs):
        self.set_extra_args(**kwargs)
        self.finalize_cluster_args()


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
