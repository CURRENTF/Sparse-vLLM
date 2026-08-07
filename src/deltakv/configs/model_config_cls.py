from transformers.models.qwen2.modeling_qwen2 import Qwen2Config
from transformers.models.llama.modeling_llama import LlamaConfig
from deltakv.configs.runtime_params import normalize_runtime_params
from deltakv.utils.log import logger

try:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
except ModuleNotFoundError:
    Qwen3Config = None

try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
except ModuleNotFoundError:
    Qwen3VLTextConfig = None


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
        deltakv_neighbor_count=1,
        layer_chunk_size=1,
        recon_mode='delta_in_latent',
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
        kv_quant_group_size=0,
        full_layer_kv_quant_bits=0,
        full_layer_cluster_ratio=0.0,
        full_layer_stride_alpha=0.0,
        full_layer_kivi_group_size=32,
        full_layer_kivi_residual_length=32,
        enable_full_layer_kivi_quant=True,
        enable_sparse_ref_fp8=True,
        hf_sparse_cache_impl=None,
        kivi_quant_bits=4,
        group_size=32,
        residual_length=32,
        visual_token_prune_only=None,
        visual_token_keep_ratio=None,
        **kwargs
    ):
        removed_config_keys = sorted(
            key for key in ("compressor_token_group_size", "seq_chunk_size", "ref_mode") if key in kwargs
        )
        if removed_config_keys:
            raise ValueError(
                "Removed DeltaKV config fields are no longer accepted: "
                f"{', '.join(f'`{key}`' for key in removed_config_keys)}. "
                "Use `deltakv_neighbor_count` for cluster reference top-k; "
                "`ref_mode` and compressor token grouping belonged to removed chunk-ref training."
            )
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
        self.deltakv_neighbor_count = deltakv_neighbor_count
        self.layer_chunk_size = layer_chunk_size
        self.recon_mode = recon_mode
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
        self.kv_quant_group_size = kv_quant_group_size
        self.full_layer_kv_quant_bits = full_layer_kv_quant_bits
        self.full_layer_cluster_ratio = full_layer_cluster_ratio
        self.full_layer_stride_alpha = full_layer_stride_alpha
        self.full_layer_kivi_group_size = full_layer_kivi_group_size
        self.full_layer_kivi_residual_length = full_layer_kivi_residual_length
        self.enable_full_layer_kivi_quant = enable_full_layer_kivi_quant
        self.enable_sparse_ref_fp8 = enable_sparse_ref_fp8
        self.hf_sparse_cache_impl = hf_sparse_cache_impl
        self.kivi_quant_bits = kivi_quant_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.visual_token_prune_only = visual_token_prune_only
        self.visual_token_keep_ratio = visual_token_keep_ratio
        
        # 调用 MRO 中的下一个 __init__ (Qwen2Config 或 LlamaConfig)
        super().__init__(**kwargs)

    def finalize_cluster_args(self):
        if not getattr(self, "use_cluster", False):
            return

        if getattr(self, "deltakv_neighbor_count", None) is None:
            raise ValueError(
                "`deltakv_neighbor_count` is required when `use_cluster=True`."
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


if Qwen3Config is not None:
    class KVQwen3Config(CustomConfigMixin, Qwen3Config):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
else:
    class KVQwen3Config(CustomConfigMixin):
        def __init__(self, **kwargs):
            raise ImportError(
                "Qwen3Config is unavailable in this Transformers installation. "
                "Use a Transformers version with Qwen3 support for Qwen3 models."
            )


if Qwen3VLTextConfig is not None:
    class KVQwen3VLTextConfig(CustomConfigMixin, Qwen3VLTextConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
else:
    class KVQwen3VLTextConfig(CustomConfigMixin):
        def __init__(self, **kwargs):
            raise ImportError(
                "Qwen3VLTextConfig is unavailable in this Transformers installation. "
                "Use a Transformers version with Qwen3-VL support for Qwen3-VL compressor training."
            )


class KVLlamaConfig(CustomConfigMixin, LlamaConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == '__main__':
    pass
