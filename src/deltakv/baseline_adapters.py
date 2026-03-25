import os
import sys

import torch
from transformers import AutoConfig

from deltakv.configs.model_config_cls import KVQwen2Config, KVLlamaConfig


def _prepend_sys_path(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)


def _resolve_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def load_omnikv_model(model_path: str, infer_config: dict, cuda_device):
    base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if base_config.model_type == "qwen2":
        from deltakv.modeling.qwen2.qwen2_with_compress_inference import Qwen2KVCompress as KVModel

        config_cls = KVQwen2Config
    elif base_config.model_type == "llama":
        from deltakv.modeling.llama.llama_with_compress_inference import LlamaKVCompress as KVModel

        config_cls = KVLlamaConfig
    else:
        raise ValueError(f"OmniKV does not support model type: {base_config.model_type}")

    config = config_cls.from_pretrained(model_path)
    config.set_infer_args(**infer_config)
    # OmniKV is the no-compression / no-cluster path over the same runtime model family.
    config.use_compression = False
    config.use_cluster = False

    return KVModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=cuda_device,
        attn_implementation="flash_attention_2",
    )


def load_kivi_model(model_path: str, infer_config: dict, cuda_device):
    repo_root = _resolve_repo_root()
    kivi_base_dir = os.path.join(repo_root, "baselines", "kivi")
    _prepend_sys_path(kivi_base_dir)

    base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    base_config.k_bits = infer_config.get("k_bits", 2)
    base_config.v_bits = infer_config.get("v_bits", 2)
    base_config.group_size = infer_config.get("group_size", 32)
    base_config.residual_length = infer_config.get("residual_length", 32)
    base_config.use_flash = True

    if base_config.model_type == "llama" or "llama" in model_path.lower():
        from models.llama_kivi import LlamaForCausalLM_KIVI as KVModel
    elif base_config.model_type == "mistral" or "mistral" in model_path.lower():
        from models.mistral_kivi import MistralForCausalLM_KIVI as KVModel
    else:
        raise ValueError(
            "KIVI baseline in this repo currently supports only Llama/Mistral checkpoints; "
            f"got model type: {base_config.model_type}"
        )

    return KVModel.from_pretrained(
        model_path,
        config=base_config,
        torch_dtype=torch.float16,
        device_map=cuda_device,
        low_cpu_mem_usage=True,
    )
