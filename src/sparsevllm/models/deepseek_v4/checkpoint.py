"""Native checkpoint names over the shared streaming weight loader."""

import re

from sparsevllm.utils.weight_target import WeightTarget


_EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")


class DeepseekV4Checkpoint:
    checkpoint_scale_suffix = ".scale"
    ignored_weight_prefixes = ("mtp.",)
    special_weight_loaders = (".native_weight",)
    packed_modules_mapping = {
        ".shared_experts.w1.": (".shared_experts.gate_up.", 0),
        ".shared_experts.w3.": (".shared_experts.gate_up.", 1),
        ".shared_experts.w2.": (".shared_experts.down.", None),
    }

    def map_weight_name(self, name):
        if match := _EXPERT.fullmatch(name):
            layer, expert, projection = match.groups()
            experts = self.model.layers[int(layer)].ffn.experts
            spec = experts.spec
            if not spec.local_expert_start <= int(expert) < spec.local_expert_start + spec.num_local_experts:
                return None
            return f"model.layers.{layer}.ffn.experts.{expert}.{projection}.native_weight"
        if name.startswith("layers."):
            name = "model." + name
            name = re.sub(r"\.hc_(attn|ffn)_(fn|scale|base)$", r".hc_\1.\2", name)
            name = name.replace(".kv_norm.weight", ".kv_norm_weight")
            name = name.replace(".compressor.norm.weight", ".compressor.norm_weight")
            if name.endswith(".wo_a.weight"):
                name = name[:-len("weight")] + "native_weight"
            return name
        if name.startswith("hc_head_"):
            return "model.hc_head." + name[len("hc_head_"):]
        if name == "head.weight":
            return "lm_head.weight"
        return "model." + name

    def resolve_special_weight(self, name):
        if not name.endswith(".native_weight"):
            return None
        path = name[:-len(".native_weight")]
        if ".ffn.experts." in path:
            prefix, expert, projection = path.rsplit(".", 2)
            return WeightTarget(self.get_submodule(prefix), (int(expert), projection))
        return WeightTarget(self.get_submodule(path))

    def load_special_weight(self, name, weight, scale):
        target = self.resolve_special_weight(name)
        if target is None:
            return 0
        if scale is None:
            raise ValueError(f"Missing native quantization scale for {name}.")
        if target.shard_id is None:
            target.module.load_quantized_weight(weight, scale)
        else:
            target.module.load_projection(*target.shard_id, weight, scale)
        return 1

    def validate_loaded_weights(self, loaded_names):
        required = set(dict(self.named_parameters()))
        required.update(name for name, _ in self.named_buffers() if name.endswith(".tid2eid"))
        if missing := required - loaded_names:
            raise ValueError(f"Missing native model weights: {sorted(missing)[:8]}.")
        for layer in self.model.layers:
            layer.ffn.experts.validate_loaded_weights()
            if not layer.attn.wo_a._weight_loaded:
                raise ValueError("Missing native grouped attention output weight.")
