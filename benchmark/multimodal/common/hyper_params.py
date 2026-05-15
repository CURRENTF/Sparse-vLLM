from __future__ import annotations

import json
from pathlib import Path


HYPER_PARAM_ALIASES = {
    "deltakv_latent_quant_bits": "delta_quant_bits",
    "engine_prefill_chunk_size": "svllm_chunk_prefill_size",
    "max_num_batched_tokens": "svllm_max_num_batched_tokens",
    "max_num_seqs_in_batch": "svllm_max_num_seqs_in_batch",
    "max_decoding_seqs": "svllm_max_decoding_seqs",
    "gpu_memory_utilization": "svllm_gpu_memory_utilization",
    "mlp_seq_chunk_size": "svllm_mlp_seq_chunk_size",
}


def add_multimodal_hyper_param_arg(parser) -> None:
    parser.add_argument(
        "--hyper_param",
        "--hyper_params",
        dest="hyper_param",
        default=None,
        help="Path to a JSON file or inline JSON object. Values override matching runtime flags.",
    )


def apply_multimodal_hyper_params(args):
    args.hyper_param_dict = {}
    if not args.hyper_param:
        return args

    text = str(args.hyper_param).strip()
    if not text:
        raise ValueError("--hyper_param cannot be empty.")
    if not text.startswith("{"):
        path = Path(text).expanduser()
        if path.exists():
            text = path.read_text(encoding="utf-8")

    hyper_params = json.loads(text)
    if not isinstance(hyper_params, dict):
        raise ValueError("--hyper_param must be a JSON object.")

    for key, value in hyper_params.items():
        target = HYPER_PARAM_ALIASES.get(key, key)
        if not hasattr(args, target):
            raise ValueError(f"Unsupported multimodal --hyper_param key: {key}")
        setattr(args, target, value)

    args.hyper_param_dict = hyper_params
    print(f"Loaded multimodal hyper-parameters: {hyper_params}", flush=True)
    return args
