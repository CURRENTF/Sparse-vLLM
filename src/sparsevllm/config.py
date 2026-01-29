import os
from dataclasses import dataclass
from transformers import AutoConfig, Qwen3Config
from typing import Union
from sparsevllm.utils.log import logger


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs_in_batch: int = 32  # 不能设置太大
    max_model_len: int = 128_000
    max_decoding_seqs: int = 64

    chunk_prefill_size: int = 8192
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: Union[Qwen3Config, AutoConfig] | None = None
    eos: int = -1
    num_kvcache_slots: int | list = -1

    # Sparse Attention Config
    vllm_sparse_method: str = ""  # "", "snapkv", "omnikv", "deltakv", "deltakv-triton", "deltakv-triton-v2", "deltakv-triton-v3", "deltakv-triton-v4", "deltakv-triton-v3-offload", "deltakv-triton-v3-cuda-offload", "pyramidkv"

    # General Sparse Config
    num_sink_tokens: int = 64
    num_recent_tokens: int = 512
    num_top_tokens: int = 4096

    # OmniKV Config
    obs_layer_ids: list[int] = None  # None means auto-calculate based on full_attn_layers (useful for omnikv)
    full_attn_layers: str | list[int] = "0" # useful for omnikv
    chunk_prefill_accel_omnikv: bool = True
    num_top_tokens_in_prefill: int | None = None

    # SnapKV Config
    snapkv_window_size: int = 32
    snapkv_num_full_layers: int = 0  # 前多少层不进行驱逐
    
    # PyramidKV Config
    pyramid_layer_ratios: list[float] | None = None  # 每层的 KV budget 比例
    pyramidkv_start_layer: int = 0
    pyramidkv_start_ratio: float = 0.6
    pyramidkv_least_layer: int | None = None
    pyramidkv_least_ratio: float = 0.01

    # DeltaKV Config
    deltakv_path: str | None = None
    deltakv_k_neighbors: int = 4
    cluster_ratio: float = 0.1
    cluster_metric: str = 'l2'  # 'l2', 'dot', 'cosine', 'fastdot' (approx; fastest)
    kv_compressed_size: int = 128
    kv_quant_bits: int = 4
    pool_kernel_size: int = 1
    # Legacy symmetric compressor controls (for backward compatibility).
    use_nonlinear_compressor: bool = True
    compressor_intermediate_size: int = 2048
    compressor_linear_bias: bool = True
    # New directional compressor controls (match latest DeltaKV training code).
    # Values: auto|linear|mlp_gelu|mlp_swiglu
    compressor_down_type: str = "auto"
    compressor_up_type: str = "auto"
    compressor_down_intermediate_size: int = -1
    compressor_up_intermediate_size: int = -1
    # DeltaKV memory split: reserve a fraction of available KV memory for the sparse full-KV pool
    # (centers + buffer + temp reconstruction slots). Larger values reduce full/latent capacity
    # but improve robustness at large batch sizes / long contexts.
    deltakv_full_pool_reserve_ratio: float = 0.1
    # DeltaKV offload: store per-layer latent cache on CPU (reduce VRAM, add PCIe overhead).
    deltakv_offload_latent: bool = False
    # Keep the first N sparse layers after each obs layer on GPU (to overlap compute with prefetch).
    # Default to 0 to avoid large GPU latent allocations; set to 1-2 to test overlap.
    deltakv_offload_keep_after_obs_layers: int = 0
    # Prefetch the next N offloaded sparse layers' latents asynchronously.
    deltakv_offload_prefetch_distance: int = 1
    # CPU gather threading (index_select/index_copy_) for latent offload path.
    deltakv_offload_cpu_threads: int = 8
    # CUDA offload gather: chunk size (map_size). Supported by provided kernel: 128/256/512/1024.
    deltakv_cuda_offload_map_size: int = 256
    # Triton kernels: group multiple KV heads per program to reduce redundant loads.
    deltakv_triton_gather_heads_per_program: int = 4
    deltakv_triton_reconstruct_heads_per_program: int = 4
    
    enable_profiler: bool = False
    throughput_log_interval_s: float = 10.0


    def __post_init__(self):
        if os.getenv("PROFILER_SVLLM"):
            self.enable_profiler = True
            
        if self.vllm_sparse_method is None:
            self.vllm_sparse_method = ""
        
        if self.num_top_tokens_in_prefill is None:
            self.num_top_tokens_in_prefill = self.num_top_tokens

        assert os.path.isdir(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.max_model_len > self.hf_config.max_position_embeddings:
            logger.warning('max_model_len > model.max_position_embeddings 输出可能不正常')
            self.hf_config.max_position_embeddings = self.max_model_len

        if self.max_num_seqs_in_batch > 32:
            logger.warning('max_num_seqs_in_batch 过大或许会占用太多显存')

        if isinstance(self.full_attn_layers, str):
            self.full_attn_layers = [int(x) for x in self.full_attn_layers.split(",")]

        # Normalize compressor type strings.
        for attr in ("compressor_down_type", "compressor_up_type"):
            v = getattr(self, attr, "auto")
            if v is None:
                v = "auto"
            v = str(v).strip().lower()
            setattr(self, attr, v if v else "auto")

        if self.vllm_sparse_method == 'deltakv' and self.deltakv_path is None:
            # 尝试在模型目录下找 deltakv_compressor.pt 或者类似文件
            # 或者让用户必须指定
            logger.warning("DeltaKV mode enabled but deltakv_path is not specified.")

        if self.obs_layer_ids is None:
            self.obs_layer_ids = []
            for l in self.full_attn_layers:
                if (l + 1) not in self.full_attn_layers:
                    self.obs_layer_ids.append(l)
        
        # 确保调度吞吐量限制不小于单次分块大小
        if self.max_num_batched_tokens < 2 * self.chunk_prefill_size:
            self.max_num_batched_tokens = 2 * self.chunk_prefill_size

        # PyramidKV 配置验证与智能生成
        if 'pyramidkv' == self.vllm_sparse_method:
            if self.pyramid_layer_ratios is None:
                num_layers = self.hf_config.num_hidden_layers
                start_l = self.pyramidkv_start_layer
                least_l = self.pyramidkv_least_layer if self.pyramidkv_least_layer is not None else num_layers - 1
                start_r = self.pyramidkv_start_ratio
                least_r = self.pyramidkv_least_ratio
                
                ratios = [1.0] * num_layers
                for i in range(start_l, num_layers):
                    if i <= least_l:
                        if least_l > start_l:
                            ratio = start_r - (start_r - least_r) * (i - start_l) / (least_l - start_l)
                        else:
                            ratio = least_r
                        ratios[i] = ratio
                    else:
                        ratios[i] = least_r
                self.pyramid_layer_ratios = ratios
                logger.info(f"PyramidKV 自动生成 layer_ratios = {[f'{r:.3f}' for r in self.pyramid_layer_ratios]}")
        
        if self.pyramid_layer_ratios is not None:
            # PyramidKV 模式自动启用 SnapKV 逻辑
            if 'pyramidkv' != self.vllm_sparse_method:
                raise ValueError('vllm_sparse_method 应为 pyramidkv')

            num_layers = self.hf_config.num_hidden_layers
            if len(self.pyramid_layer_ratios) != num_layers:
                raise ValueError(f"pyramid_layer_ratios 长度 ({len(self.pyramid_layer_ratios)}) 必须等于模型层数 ({num_layers})")

            if any(r <= 0 or r > 1.0 for r in self.pyramid_layer_ratios):
                raise ValueError("pyramid_layer_ratios 的所有值必须在 (0, 1.0] 范围内")
        
        logger.info(f"LLM Config: {self}".replace('\n', ' '))
