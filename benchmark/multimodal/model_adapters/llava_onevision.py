from __future__ import annotations

from benchmark.multimodal.visual_cache.run_visual_cache import (
    batch_to_device,
    build_llava_delta_quant_policy,
    build_llava_deltakv_policy,
    build_visual_uniform_policy,
    ensure_left_padding,
    load_llava_delta_quant_model,
    load_llava_deltakv_model,
    load_vanilla_model,
    load_visual_uniform_model,
    resolve_compressor_path,
)


def build_svllm_delta_quant_policy(args):
    return {
        "method": "svllm_deltakv_delta_quant",
        "backend": "sparsevllm",
        "selection_policy": "cluster_ref_delta_quant",
        "uses_deltakv_wrapper": False,
        "uses_learned_compressor": False,
        "uses_cluster": True,
        "uses_ref_tokens": True,
        "uses_visual_uniform_pruning": False,
        "supports_batch_generation": True,
        "kv_quant_bits": int(args.delta_quant_bits),
        "note": (
            "Uses Sparse-vLLM LLaVA-OneVision prompt embeddings plus the existing "
            "DeltaKV delta-quant cache manager. Vision/projector work runs once per "
            "processor batch; text prefill/decode reuse Sparse-vLLM scheduling."
        ),
    }


def load_svllm_delta_quant_model(args, dtype, device):
    if getattr(args, "stride_alpha", None) not in {None, 0.0}:
        raise ValueError("SVLLM DeltaKV delta-quant does not support dynamic stride; use HF backend for stride_alpha.")
    if resolve_compressor_path(args) is not None:
        raise ValueError(
            "svllm_deltakv_delta_quant does not use a learned compressor checkpoint. "
            "Set --deltakv_checkpoint_path none."
        )
    if int(args.cuda_device) != 0:
        raise ValueError(
            "Sparse-vLLM uses cuda:0 inside the visible device set. "
            "Run with CUDA_VISIBLE_DEVICES=<gpu_id> and pass --cuda_device 0."
        )

    from sparsevllm import LLM

    policy = build_svllm_delta_quant_policy(args)
    print(
        "[llava_cache_policy] "
        f"method={policy['method']} backend={policy['backend']} selection={policy['selection_policy']} "
        f"cluster={policy['uses_cluster']} compressor={policy['uses_learned_compressor']} "
        f"ref_tokens={policy['uses_ref_tokens']} kv_quant_bits={policy['kv_quant_bits']} "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    llm = LLM(
        args.model_path,
        torch_dtype=dtype,
        sparse_method="deltakv-delta-quant",
        sink_keep_tokens=args.sink_keep_tokens,
        recent_keep_tokens=args.recent_keep_tokens,
        decode_keep_tokens=args.decode_keep_tokens,
        prefill_keep_tokens=args.prefill_keep_tokens,
        deltakv_center_ratio=args.deltakv_center_ratio,
        deltakv_neighbor_count=args.deltakv_neighbor_count,
        deltakv_latent_quant_bits=args.delta_quant_bits,
        full_attention_layers=args.full_attention_layers,
        engine_prefill_chunk_size=args.svllm_chunk_prefill_size,
        max_num_batched_tokens=args.svllm_max_num_batched_tokens,
        max_num_seqs_in_batch=args.svllm_max_num_seqs_in_batch,
        max_decoding_seqs=args.svllm_max_decoding_seqs,
        gpu_memory_utilization=args.svllm_gpu_memory_utilization,
        mlp_seq_chunk_size=args.svllm_mlp_seq_chunk_size,
        prefill_schedule_policy="long_bs1full_short_batch",
    )
    llm.config.video_token_id = int(llm.config.hf_config.video_token_id)
    llm.config.image_token_id = int(llm.config.hf_config.image_token_id)
    llm.svllm_device = device
    return llm, policy

__all__ = [
    "batch_to_device",
    "build_llava_delta_quant_policy",
    "build_llava_deltakv_policy",
    "build_visual_uniform_policy",
    "ensure_left_padding",
    "load_llava_delta_quant_model",
    "load_llava_deltakv_model",
    "load_vanilla_model",
    "load_visual_uniform_model",
    "build_svllm_delta_quant_policy",
    "load_svllm_delta_quant_model",
]
