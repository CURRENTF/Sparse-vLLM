import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from sparsevllm.utils.log import logger


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


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
    files = glob(os.path.join(path, "*.safetensors"))
    is_safetensors = len(files) > 0
    if not is_safetensors:
        files = glob(os.path.join(path, "*.bin")) + glob(os.path.join(path, "*.pt"))

    if not files:
        if os.path.isfile(path):
            files = [path]
            is_safetensors = path.endswith(".safetensors")
        else:
            raise FileNotFoundError(f"No compressor weights found in {path}")

    loaded_count = 0
    for file in files:
        if is_safetensors:
            with safe_open(file, "pt", "cpu") as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}
        else:
            state_dict = torch.load(file, map_location="cpu")

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


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = glob(os.path.join(path, "*.safetensors"))
    assert len(files) > 0, f"No safetensors found in {path}"
    
    loaded_count = 0
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        loaded_count += 1
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded_count += 1
    
    assert loaded_count > 0, f"No weights were loaded from {path}"
    print(f"Successfully loaded {loaded_count} weights from {path}")
