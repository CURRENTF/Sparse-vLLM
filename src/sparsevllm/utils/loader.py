import json
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from sparsevllm.utils.log import logger


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _translate_deepseek_weight_name(model: nn.Module, weight_name: str) -> str | None:
    if getattr(model, "hf_model_type", "") not in ("deepseek_v2", "deepseek_v32"):
        return None

    if weight_name == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    if weight_name == "model.norm.weight":
        return "norm.weight"
    if weight_name == "lm_head.weight":
        return "output.weight"

    if not weight_name.startswith("model.layers."):
        return None

    parts = weight_name.split(".")
    if len(parts) < 5:
        return None
    layer_idx = parts[2]
    tail = parts[3:]
    prefix = f"layers.{layer_idx}."

    if tail == ["input_layernorm", "weight"]:
        return prefix + "attn_norm.weight"
    if tail == ["post_attention_layernorm", "weight"]:
        return prefix + "ffn_norm.weight"

    if tail[:2] == ["self_attn", "q_proj"] and getattr(getattr(model, "args", None), "q_lora_rank", 0) <= 0:
        return prefix + "attn.wq.weight"
    if tail[:2] == ["self_attn", "kv_a_proj_with_mqa"]:
        return prefix + "attn.wkv_a.weight"
    if tail[:2] == ["self_attn", "kv_a_layernorm"]:
        return prefix + "attn.kv_norm.weight"
    if tail[:2] == ["self_attn", "kv_b_proj"]:
        return prefix + "attn.wkv_b.weight"
    if tail[:2] == ["self_attn", "o_proj"]:
        return prefix + "attn.wo.weight"

    if tail[:2] == ["mlp", "gate_proj"]:
        return prefix + "ffn.w1.weight"
    if tail[:2] == ["mlp", "up_proj"]:
        return prefix + "ffn.w3.weight"
    if tail[:2] == ["mlp", "down_proj"]:
        return prefix + "ffn.w2.weight"
    if tail[:2] == ["mlp", "gate"] and len(tail) == 3 and tail[2] == "weight":
        return prefix + "ffn.gate.weight"
    if tail[:2] == ["mlp", "shared_experts"] and len(tail) == 4:
        proj = tail[2]
        if proj == "gate_proj":
            return prefix + "ffn.shared_experts.w1.weight"
        if proj == "up_proj":
            return prefix + "ffn.shared_experts.w3.weight"
        if proj == "down_proj":
            return prefix + "ffn.shared_experts.w2.weight"
    if tail[:2] == ["mlp", "experts"] and len(tail) == 5:
        expert_idx = tail[2]
        proj = tail[3]
        if proj == "gate_proj":
            return prefix + f"ffn.experts.{expert_idx}.w1.weight"
        if proj == "up_proj":
            return prefix + f"ffn.experts.{expert_idx}.w3.weight"
        if proj == "down_proj":
            return prefix + f"ffn.experts.{expert_idx}.w2.weight"

    return None


def _iter_deltakv_compressor_items(state_dict: dict[str, torch.Tensor]):
    for key, weight in state_dict.items():
        parts = key.split(".")
        # Supported formats:
        # 1) HF-style per-layer: "...layers.{i}....compress_down...."
        # 2) Shared compressors: "compress_down...." (broadcast to all layers)
        if "compress_down" in parts:
            yield key, "compress_down", parts, weight
        elif "compress_up" in parts:
            yield key, "compress_up", parts, weight


def _infer_single_compressor_spec(state_dict: dict[str, torch.Tensor], comp_name: str):
    bias = False
    inferred_kind = None
    inferred_intermediate = None
    inferred_dtype = None

    for key, name, parts, weight in _iter_deltakv_compressor_items(state_dict):
        if name != comp_name:
            continue
        comp_token_idx = parts.index(comp_name)
        sub_parts = parts[comp_token_idx + 1:]
        if not sub_parts:
            continue

        inferred_dtype = inferred_dtype or weight.dtype
        if sub_parts[-1] == "bias":
            bias = True

        head = sub_parts[0]
        # nn.Linear: "...compress_down.weight" / "...compress_down.bias"
        if head in ("weight", "bias"):
            inferred_kind = inferred_kind or "linear"
            continue
        # nn.Sequential: "...compress_down.0.weight" / "...compress_down.2.weight"
        if head.isdigit():
            inferred_kind = inferred_kind or "mlp_gelu"
            if head == "0" and sub_parts[-1] == "weight":
                inferred_intermediate = inferred_intermediate or int(weight.shape[0])
            continue
        # SwiGLU: "...compress_down.w12.weight" / "...compress_down.w3.weight"
        if head == "w12":
            inferred_kind = inferred_kind or "mlp_swiglu"
            if sub_parts[-1] == "weight":
                inferred_intermediate = inferred_intermediate or int(weight.shape[0] // 2)
            continue

    return inferred_kind, inferred_intermediate, bias, inferred_dtype


def _infer_kv_compressed_size(state_dict: dict[str, torch.Tensor]) -> int | None:
    for _, name, parts, weight in _iter_deltakv_compressor_items(state_dict):
        comp_token_idx = parts.index(name)
        sub_parts = parts[comp_token_idx + 1:]
        if not sub_parts or sub_parts[-1] != "weight":
            continue

        head = sub_parts[0]
        if name == "compress_down":
            if head in ("weight", "2", "w3"):
                return int(weight.shape[0])
        else:
            if head in ("weight", "0", "w12"):
                return int(weight.shape[1])
    return None


def _resolve_deltakv_checkpoint_files(path: str) -> tuple[str, list[str], bool]:
    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path, "*.safetensors")))
        is_safetensors = len(files) > 0
        if not is_safetensors:
            files = sorted(glob(os.path.join(path, "*.bin")) + glob(os.path.join(path, "*.pt")))
        ckpt_dir = path
    elif os.path.isfile(path):
        files = [path]
        is_safetensors = path.endswith(".safetensors")
        ckpt_dir = os.path.dirname(path)
    else:
        raise FileNotFoundError(f"No compressor weights found in {path}")

    if not files:
        raise FileNotFoundError(f"No compressor weights found in {path}")
    return ckpt_dir, files, is_safetensors


def _load_deltakv_state_dict(file: str, *, is_safetensors: bool) -> dict[str, torch.Tensor]:
    if is_safetensors:
        with safe_open(file, "pt", "cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}
    return torch.load(file, map_location="cpu")


def sync_deltakv_config_from_checkpoint(config) -> bool:
    path = getattr(config, "deltakv_path", None)
    method = str(getattr(config, "vllm_sparse_method", "") or "")
    if not path or not method.startswith("deltakv"):
        return False

    ckpt_dir, files, is_safetensors = _resolve_deltakv_checkpoint_files(path)
    updates: dict[str, object] = {}
    config_json = os.path.join(ckpt_dir, "config.json")
    if os.path.isfile(config_json):
        with open(config_json, "r", encoding="utf-8") as f:
            ckpt_cfg = json.load(f)

        for key in (
            "kv_compressed_size",
            "use_nonlinear_compressor",
            "compressor_intermediate_size",
            "compressor_linear_bias",
            "compressor_down_type",
            "compressor_up_type",
            "compressor_down_intermediate_size",
            "compressor_up_intermediate_size",
        ):
            if key in ckpt_cfg:
                updates[key] = ckpt_cfg[key]

        if ckpt_cfg.get("split_kv", False):
            raise NotImplementedError(
                "Detected split_kv DeltaKV checkpoint from config.json. "
                "sparsevllm DeltaKVCacheManager currently expects unified compress_down/compress_up."
            )

    missing_shape_keys = {
        "kv_compressed_size",
        "compressor_down_type",
        "compressor_up_type",
        "compressor_down_intermediate_size",
        "compressor_up_intermediate_size",
        "compressor_linear_bias",
        "use_nonlinear_compressor",
    }
    need_weight_inference = bool(missing_shape_keys.difference(updates.keys()))
    state_dict = None
    if need_weight_inference:
        for file in files:
            state_dict = _load_deltakv_state_dict(file, is_safetensors=is_safetensors)
            if any("compress_down" in key or "compress_up" in key for key in state_dict):
                break
            state_dict = None

        if state_dict is not None:
            kv_compressed_size = _infer_kv_compressed_size(state_dict)
            down_kind, down_inter, down_bias, _ = _infer_single_compressor_spec(state_dict, "compress_down")
            up_kind, up_inter, up_bias, _ = _infer_single_compressor_spec(state_dict, "compress_up")
            if kv_compressed_size is not None and "kv_compressed_size" not in updates:
                updates["kv_compressed_size"] = kv_compressed_size
            if down_kind is not None and "compressor_down_type" not in updates:
                updates["compressor_down_type"] = down_kind
            if up_kind is not None and "compressor_up_type" not in updates:
                updates["compressor_up_type"] = up_kind
            if down_kind is not None and "compressor_down_intermediate_size" not in updates:
                updates["compressor_down_intermediate_size"] = -1 if down_kind == "linear" else int(down_inter or -1)
            if up_kind is not None and "compressor_up_intermediate_size" not in updates:
                updates["compressor_up_intermediate_size"] = -1 if up_kind == "linear" else int(up_inter or -1)
            if down_kind is not None and up_kind is not None and "use_nonlinear_compressor" not in updates:
                updates["use_nonlinear_compressor"] = (down_kind != "linear" or up_kind != "linear")
            if down_bias == up_bias and "compressor_linear_bias" not in updates:
                updates["compressor_linear_bias"] = bool(down_bias)

            for key in state_dict.keys():
                if ".k_compress_down." in key or ".v_compress_down." in key or ".k_compress_up." in key or ".v_compress_up." in key:
                    raise NotImplementedError(
                        "Detected split_kv compressor checkpoint (k_compress_*/v_compress_*). "
                        "sparsevllm DeltaKVCacheManager currently expects unified compress_down/compress_up."
                    )

    changed: dict[str, tuple[object, object]] = {}
    for key, value in updates.items():
        if value is None or not hasattr(config, key):
            continue
        current = getattr(config, key)
        if isinstance(value, str):
            value = value.strip().lower() or "auto"
        if current != value:
            setattr(config, key, value)
            changed[key] = (current, value)

    if changed:
        changes_str = ", ".join(f"{key}: {old} -> {new}" for key, (old, new) in changed.items())
        logger.info(f"Synced DeltaKV config from checkpoint {ckpt_dir}: {changes_str}")
    return bool(changed)


def _compressor_signature(mod: nn.Module):
    if isinstance(mod, nn.Linear):
        return "linear", None, (mod.bias is not None)
    if hasattr(mod, "w12") and hasattr(mod, "w3"):
        w3 = getattr(mod, "w3")
        inter = int(w3.in_features) if isinstance(w3, nn.Linear) else None
        has_bias = bool(getattr(getattr(mod, "w12", None), "bias", None) is not None) and bool(
            getattr(getattr(mod, "w3", None), "bias", None) is not None
        )
        return "mlp_swiglu", inter, has_bias
    if isinstance(mod, nn.Sequential) and len(mod) >= 3 and isinstance(mod[0], nn.Linear) and isinstance(mod[1], nn.GELU) and isinstance(mod[2], nn.Linear):
        return "mlp_gelu", int(mod[0].out_features), (mod[0].bias is not None and mod[2].bias is not None)
    return mod.__class__.__name__, None, None


def _maybe_rebuild_cache_manager_compressors(cache_manager, state_dict: dict[str, torch.Tensor]):
    config = getattr(cache_manager, "config", None)
    if config is None:
        return

    down_kind, down_inter, down_bias, down_dtype = _infer_single_compressor_spec(state_dict, "compress_down")
    up_kind, up_inter, up_bias, up_dtype = _infer_single_compressor_spec(state_dict, "compress_up")
    if down_kind is None or up_kind is None:
        return

    current_down = cache_manager.compress_down[0] if getattr(cache_manager, "compress_down", None) else None
    current_up = cache_manager.compress_up[0] if getattr(cache_manager, "compress_up", None) else None
    cur_down_kind, cur_down_inter, cur_down_bias = _compressor_signature(current_down) if current_down is not None else (None, None, None)
    cur_up_kind, cur_up_inter, cur_up_bias = _compressor_signature(current_up) if current_up is not None else (None, None, None)

    need_rebuild = (
        cur_down_kind != down_kind
        or cur_up_kind != up_kind
        or (down_kind != "linear" and cur_down_inter != down_inter)
        or (up_kind != "linear" and cur_up_inter != up_inter)
        or cur_down_bias != down_bias
        or cur_up_bias != up_bias
    )
    target_dtype = down_dtype or up_dtype
    if not need_rebuild:
        # Even if the structure matches, ensure dtype matches checkpoint weights (avoid bf16->fp32 silent casts).
        if target_dtype is not None and current_down is not None:
            try:
                cur_dtype = next(current_down.parameters()).dtype
            except StopIteration:
                cur_dtype = None
            if cur_dtype is not None and cur_dtype != target_dtype:
                logger.info(f"Casting sparsevllm compressors to {target_dtype} to match checkpoint weights.")
                cache_manager.compress_down = [m.to(dtype=target_dtype) for m in cache_manager.compress_down]
                cache_manager.compress_up = [m.to(dtype=target_dtype) for m in cache_manager.compress_up]
        return

    logger.info(
        f"Rebuilding sparsevllm compressors to match checkpoint: "
        f"down={down_kind}(inter={down_inter},bias={down_bias}) "
        f"up={up_kind}(inter={up_inter},bias={up_bias})"
    )

    # Update config so downstream logs/debug reflect the effective compressor architecture.
    if hasattr(config, "compressor_linear_bias"):
        if down_bias == up_bias:
            config.compressor_linear_bias = bool(down_bias)
        else:
            logger.warning(
                "Checkpoint has asymmetric bias setting between compress_down and compress_up; "
                "sparsevllm config only has a single `compressor_linear_bias`, keeping current value."
            )
    if hasattr(config, "use_nonlinear_compressor"):
        config.use_nonlinear_compressor = (down_kind != "linear" or up_kind != "linear")
    if hasattr(config, "compressor_down_type"):
        config.compressor_down_type = down_kind
    if hasattr(config, "compressor_up_type"):
        config.compressor_up_type = up_kind
    if hasattr(config, "compressor_down_intermediate_size"):
        config.compressor_down_intermediate_size = -1 if down_kind == "linear" else int(down_inter or -1)
    if hasattr(config, "compressor_up_intermediate_size"):
        config.compressor_up_intermediate_size = -1 if up_kind == "linear" else int(up_inter or -1)

    from sparsevllm.utils.compressor import create_compressor

    num_layers = len(cache_manager.compress_down)
    cache_manager.compress_down = [
        create_compressor(is_down=True, config=config, bias_override=down_bias).to(device="cuda", dtype=target_dtype)
        for _ in range(num_layers)
    ]
    cache_manager.compress_up = [
        create_compressor(is_down=False, config=config, bias_override=up_bias).to(device="cuda", dtype=target_dtype)
        for _ in range(num_layers)
    ]


def load_deltakv_compressors_to_cache_manager(cache_manager, path: str):
    """
    Load DeltaKV compressor weights into cache manager compressor modules.
    """
    _, files, is_safetensors = _resolve_deltakv_checkpoint_files(path)

    loaded_count = 0
    for file in files:
        state_dict = _load_deltakv_state_dict(file, is_safetensors=is_safetensors)

        # Detect unsupported split_kv checkpoints early (k_compress_down/v_compress_down).
        for key in state_dict.keys():
            if ".k_compress_down." in key or ".v_compress_down." in key or ".k_compress_up." in key or ".v_compress_up." in key:
                raise NotImplementedError(
                    "Detected split_kv compressor checkpoint (k_compress_*/v_compress_*). "
                    "sparsevllm DeltaKVCacheManager currently expects unified compress_down/compress_up."
                )

        if loaded_count == 0:
            _maybe_rebuild_cache_manager_compressors(cache_manager, state_dict)

        for key, weight in state_dict.items():
            parts = key.split('.')
            if "compress_down" not in parts and "compress_up" not in parts:
                continue

            # Two formats:
            # 1) HF-style per-layer: "...layers.{i}....compress_down...."
            # 2) Shared compressors: "compress_down...." (broadcast to all DeltaKV layers)
            is_shared = ("layers" not in parts)
            layer_idx = None
            if not is_shared:
                try:
                    layer_token_idx = parts.index("layers")
                    layer_idx = int(parts[layer_token_idx + 1])
                except (ValueError, IndexError):
                    raise ValueError(f"无法从权重键名中解析层索引: {key}")

            if "compress_down" in parts:
                comp_name = "compress_down"
            else:
                comp_name = "compress_up"

            comp_token_idx = parts.index(comp_name)
            sub_key = ".".join(parts[comp_token_idx + 1:])

            if is_shared:
                target_layer_indices = list(range(len(cache_manager.compress_down)))
            else:
                if layer_idx not in cache_manager.deltakv_layer_to_idx:
                    logger.debug(f"权重 {key} 对应的层索引 {layer_idx} 不在 deltakv_layer_to_idx 中，跳过")
                    continue
                target_layer_indices = [cache_manager.deltakv_layer_to_idx[layer_idx]]

            for l_idx in target_layer_indices:
                compressor = cache_manager.compress_down[l_idx] if comp_name == "compress_down" else cache_manager.compress_up[l_idx]
                try:
                    # 尝试获取对应的参数
                    if '.' in sub_key:
                        prefix, name = sub_key.rsplit('.', 1)
                        param = compressor.get_submodule(prefix).get_parameter(name)
                    else:
                        param = compressor.get_parameter(sub_key)

                    if param.shape != weight.shape:
                        raise ValueError(f"权重 {key} 形状不匹配: 预期 {param.shape}, 实际 {weight.shape}")

                    param.data.copy_(weight)
                    loaded_count += 1
                except Exception as e:
                    # 尝试直接访问属性作为备选方案
                    try:
                        target = getattr(compressor, sub_key)
                        if isinstance(target, nn.Parameter):
                            if target.shape != weight.shape:
                                raise ValueError(f"权重 {key} 形状不匹配: 预期 {target.shape}, 实际 {weight.shape}")
                            target.data.copy_(weight)
                            loaded_count += 1
                            continue
                    except Exception:
                        pass

                    raise RuntimeError(f"未能将权重 {key} 加载到压缩器模块 (layer {layer_idx}): {e}")

    assert loaded_count > 0, f"No DeltaKV compressor weights were loaded into cache manager from {path}"
    print(f"Successfully loaded {loaded_count} DeltaKV compressor weights into cache manager from {path}")


def load_model(model: nn.Module, path: str, *, rank: int | None = None, world_size: int | None = None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    assert len(files) > 0, f"No safetensors found in {path}"

    # DeepSeek official converters often emit one file per rank:
    #   model{rank}-mp{world_size}.safetensors
    # In that case we should only load the local rank shard.
    if rank is not None and world_size is not None:
        shard = os.path.join(path, f"model{rank}-mp{world_size}.safetensors")
        if os.path.isfile(shard):
            files = [shard]
        else:
            mp_files = sorted(glob(os.path.join(path, f"model*-mp{world_size}.safetensors")))
            if mp_files:
                raise FileNotFoundError(
                    "Detected per-rank weight shards but missing expected shard for this rank. "
                    f"expected={shard} available={mp_files}"
                )
    
    loaded_count = 0
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for source_weight_name in f.keys():
                param_name = _translate_deepseek_weight_name(model, source_weight_name) or source_weight_name
                for k in packed_modules_mapping:
                    if k in param_name:
                        v, shard_id = packed_modules_mapping[k]
                        packed_param_name = param_name.replace(k, v)
                        param = model.get_parameter(packed_param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(source_weight_name), shard_id)
                        loaded_count += 1
                        break
                else:
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(source_weight_name))
                    loaded_count += 1
    
    assert loaded_count > 0, f"No weights were loaded from {path}"
    print(f"Successfully loaded {loaded_count} weights from {path}")
