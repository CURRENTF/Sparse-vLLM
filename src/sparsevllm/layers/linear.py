import torch
from torch import nn
import torch.nn.functional as F

from sparsevllm.distributed import get_parallel_context
from sparsevllm.quantization import QuantizationRegistry


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quantization=None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.parallel_context = get_parallel_context()
        self.tp_rank = self.parallel_context.tp_rank
        self.tp_size = self.parallel_context.tp_size
        self.quantization_config = quantization
        self.quantized = bool(getattr(quantization, "enabled", False))
        self._quantized_weight_loaded = not self.quantized
        self._quantized_loaded_ranges: list[tuple[int, int]] = []
        if self.quantized:
            if tuple(getattr(quantization, "weight_block_size", (128, 128))) != (128, 128):
                raise ValueError(
                    "Linear FP8 quantization requires weight_block_size=(128, 128), "
                    f"got {getattr(quantization, 'weight_block_size', None)}."
                )
            self.quant_backend = QuantizationRegistry.create_linear_backend(quantization)
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.register_buffer(
                "weight_scale_inv",
                torch.empty(self._scale_shape_for_weight_shape((output_size, input_size)), dtype=torch.float32),
            )
        else:
            self.quant_backend = None
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.register_buffer("weight_scale_inv", None)
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _scale_shape_for_weight_shape(shape: tuple[int, int]) -> tuple[int, int]:
        out_features, in_features = int(shape[0]), int(shape[1])
        return ((out_features + 127) // 128, (in_features + 127) // 128)

    @staticmethod
    def _require_scale_shardable(name: str, size: int, tp_size: int) -> int:
        size = int(size)
        tp_size = int(tp_size)
        if size % tp_size != 0:
            raise ValueError(f"{name} scale shard dimension is not divisible by TP size: {size} % {tp_size}.")
        return size // tp_size

    def _ensure_quantized_loader(self):
        if not self.quantized:
            raise RuntimeError(
                f"{type(self).__name__} received weight_scale_inv but was not constructed with FP8 quantization."
            )

    def rank_local_weight_slice(
        self,
        source_shape: tuple[int, ...],
        *,
        loaded_shard_id=None,
        is_scale: bool = False,
    ) -> tuple[slice, ...] | None:
        del loaded_shard_id, is_scale
        if self.tp_size == 1 or self.tp_dim is None or len(source_shape) <= self.tp_dim:
            return None
        shard_size = divide(int(source_shape[self.tp_dim]), self.tp_size)
        start = self.tp_rank * shard_size
        slices = [slice(None)] * len(source_shape)
        slices[self.tp_dim] = slice(start, start + shard_size)
        return tuple(slices)

    def _copy_quantized_weight_and_scale(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor,
        *,
        weight_target: torch.Tensor | None = None,
        scale_target: torch.Tensor | None = None,
    ) -> None:
        self._ensure_quantized_loader()
        weight_target = self.weight.data if weight_target is None else weight_target
        scale_target = self.weight_scale_inv if scale_target is None else scale_target
        if loaded_weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"FP8 Linear weight must be torch.float8_e4m3fn, got {loaded_weight.dtype}.")
        if tuple(loaded_weight.shape) != tuple(weight_target.shape):
            raise ValueError(
                f"FP8 weight shape mismatch for {type(self).__name__}: "
                f"expected={tuple(weight_target.shape)}, got={tuple(loaded_weight.shape)}."
            )
        if tuple(loaded_scale.shape) != tuple(scale_target.shape):
            raise ValueError(
                f"weight_scale_inv shape mismatch for {type(self).__name__}: "
                f"expected={tuple(scale_target.shape)}, got={tuple(loaded_scale.shape)}."
            )
        weight_target.copy_(loaded_weight)
        scale_target.copy_(loaded_scale.to(dtype=scale_target.dtype))
        self._mark_quantized_weight_range_loaded(weight_target)

    def _mark_quantized_weight_range_loaded(self, weight_target: torch.Tensor) -> None:
        if not self.quantized:
            return
        base = self.weight.data
        if int(weight_target.data_ptr()) == int(base.data_ptr()) and tuple(weight_target.shape) == tuple(base.shape):
            self._quantized_loaded_ranges = [(0, int(base.shape[0]))]
            self._quantized_weight_loaded = True
            return
        if weight_target.dim() != 2 or base.dim() != 2:
            raise RuntimeError("FP8 Linear load tracking expects rank-2 weights.")
        if int(weight_target.stride(0)) != int(base.stride(0)) or int(weight_target.stride(1)) != int(base.stride(1)):
            raise RuntimeError("FP8 Linear load tracking only supports contiguous row ranges.")
        row_stride_bytes = int(base.stride(0) * base.element_size())
        offset_bytes = int(weight_target.data_ptr()) - int(base.data_ptr())
        if row_stride_bytes <= 0 or offset_bytes < 0 or offset_bytes % row_stride_bytes != 0:
            raise RuntimeError("FP8 Linear load tracking could not resolve loaded row range.")
        start = offset_bytes // row_stride_bytes
        end = int(start + weight_target.shape[0])
        if start < 0 or end > int(base.shape[0]):
            raise RuntimeError(
                f"FP8 Linear loaded row range is out of bounds: start={start} end={end} rows={base.shape[0]}."
            )
        self._quantized_loaded_ranges.append((int(start), int(end)))
        merged: list[tuple[int, int]] = []
        for cur_start, cur_end in sorted(self._quantized_loaded_ranges):
            if not merged or cur_start > merged[-1][1]:
                merged.append((cur_start, cur_end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], cur_end))
        self._quantized_loaded_ranges = merged
        self._quantized_weight_loaded = len(merged) == 1 and merged[0] == (0, int(base.shape[0]))

    def load_quantized_weight(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor,
        loaded_shard_id=None,
    ) -> None:
        if loaded_shard_id is not None:
            raise ValueError(f"{type(self).__name__} does not accept loaded_shard_id={loaded_shard_id!r}.")
        self._copy_quantized_weight_and_scale(loaded_weight, loaded_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantization=None,
    ):
        super().__init__(input_size, output_size, bias, quantization=quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            return self.quant_backend(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantization=None,
    ):
        tp_size = get_parallel_context().tp_size
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quantization=quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        if loaded_weight.size(self.tp_dim) != shard_size:
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def load_quantized_weight(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor,
        loaded_shard_id=None,
    ) -> None:
        if loaded_shard_id is not None:
            raise ValueError(f"{type(self).__name__} does not accept loaded_shard_id={loaded_shard_id!r}.")
        self._ensure_quantized_loader()
        param_data = self.weight.data
        shard_size = param_data.size(self.tp_dim)
        if loaded_weight.size(self.tp_dim) == shard_size:
            weight_shard = loaded_weight
        else:
            start_idx = self.tp_rank * shard_size
            weight_shard = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)

        if tuple(loaded_scale.shape) == tuple(self.weight_scale_inv.shape):
            scale_shard = loaded_scale
        else:
            scale_shard_size = self._require_scale_shardable(
                f"{type(self).__name__}",
                loaded_scale.size(self.tp_dim),
                self.tp_size,
            )
            scale_start = self.tp_rank * scale_shard_size
            scale_shard = loaded_scale.narrow(
                self.tp_dim, scale_start, scale_shard_size
            )
        self._copy_quantized_weight_and_scale(weight_shard, scale_shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            return self.quant_backend(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quantization=None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, quantization=quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        if loaded_weight.size(self.tp_dim) != shard_size:
            loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def load_quantized_weight(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        self._ensure_quantized_loader()
        loaded_shard_id = int(loaded_shard_id)
        local_output_sizes = [divide(size, self.tp_size) for size in self.output_sizes]
        local_offset = sum(local_output_sizes[:loaded_shard_id])
        local_size = local_output_sizes[loaded_shard_id]
        if local_offset % 128 != 0 or local_size % 128 != 0:
            raise ValueError(
                "MergedColumnParallelLinear FP8 scale sharding requires each local merged "
                f"output shard to be 128-aligned, got offset={local_offset}, size={local_size}."
            )

        weight_target = self.weight.data.narrow(self.tp_dim, local_offset, local_size)
        weight_shard = (
            loaded_weight
            if loaded_weight.size(self.tp_dim) == local_size
            else loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        )
        scale_offset = local_offset // 128
        scale_size = local_size // 128
        scale_target = self.weight_scale_inv.narrow(0, scale_offset, scale_size)
        if tuple(loaded_scale.shape) == tuple(scale_target.shape):
            scale_shard = loaded_scale
        else:
            scale_shard_size = self._require_scale_shardable(
                "MergedColumnParallelLinear",
                loaded_scale.size(0),
                self.tp_size,
            )
            scale_shard = loaded_scale.narrow(
                0, self.tp_rank * scale_shard_size, scale_shard_size
            )
        self._copy_quantized_weight_and_scale(
            weight_shard,
            scale_shard,
            weight_target=weight_target,
            scale_target=scale_target,
        )


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quantization=None,
    ):
        tp_size = get_parallel_context().tp_size
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quantization=quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        if loaded_weight.size(self.tp_dim) != shard_size:
            loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def load_quantized_weight(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor,
        loaded_shard_id: str,
    ) -> None:
        self._ensure_quantized_loader()
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        if shard_offset % 128 != 0 or shard_size % 128 != 0:
            raise ValueError(
                "QKVParallelLinear FP8 scale sharding requires each local q/k/v output shard "
                f"to be 128-aligned, got offset={shard_offset}, size={shard_size}."
            )
        weight_target = self.weight.data.narrow(self.tp_dim, shard_offset, shard_size)
        weight_shard = (
            loaded_weight
            if loaded_weight.size(self.tp_dim) == shard_size
            else loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        )
        scale_target = self.weight_scale_inv.narrow(0, shard_offset // 128, shard_size // 128)
        if tuple(loaded_scale.shape) == tuple(scale_target.shape):
            scale_shard = loaded_scale
        else:
            scale_shard_size = self._require_scale_shardable(
                "QKVParallelLinear",
                loaded_scale.size(0),
                self.tp_size,
            )
            scale_shard = loaded_scale.narrow(
                0, self.tp_rank * scale_shard_size, scale_shard_size
            )
        self._copy_quantized_weight_and_scale(
            weight_shard,
            scale_shard,
            weight_target=weight_target,
            scale_target=scale_target,
        )


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantization=None,
    ):
        tp_size = get_parallel_context().tp_size
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, quantization=quantization)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        if loaded_weight.size(self.tp_dim) != shard_size:
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def load_quantized_weight(
        self,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor,
        loaded_shard_id=None,
    ) -> None:
        if loaded_shard_id is not None:
            raise ValueError(f"{type(self).__name__} does not accept loaded_shard_id={loaded_shard_id!r}.")
        self._ensure_quantized_loader()
        param_data = self.weight.data
        shard_size = param_data.size(self.tp_dim)
        if loaded_weight.size(self.tp_dim) == shard_size:
            weight_shard = loaded_weight
        else:
            start_idx = self.tp_rank * shard_size
            weight_shard = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)

        if tuple(loaded_scale.shape) == tuple(self.weight_scale_inv.shape):
            scale_shard = loaded_scale
        else:
            scale_shard_size = self._require_scale_shardable(
                "RowParallelLinear",
                loaded_scale.size(self.tp_dim),
                self.tp_size,
            )
            scale_start = self.tp_rank * scale_shard_size
            scale_shard = loaded_scale.narrow(
                self.tp_dim, scale_start, scale_shard_size
            )
        self._copy_quantized_weight_and_scale(weight_shard, scale_shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        if self.quantized:
            y = self.quant_backend(x, self.weight, self.weight_scale_inv, bias)
        else:
            y = F.linear(x, self.weight, bias)
        return self.parallel_context.tp_all_reduce(y)
