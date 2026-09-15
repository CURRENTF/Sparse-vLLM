"""Public FlashInfer variable-length top-k with physical slot translation."""

from functools import lru_cache
import inspect

from sparsevllm.kernels.external.flashinfer.support import flashinfer_kernel_support
from sparsevllm.kernels.external.support import ExternalKernelContractError


@lru_cache(maxsize=1)
def index_topk_op():
    flashinfer_kernel_support("compressed index top-k")
    try:
        from flashinfer.topk import top_k_page_table_transform
        expected = ("input", "src_page_table", "lengths", "k", "row_to_batch", "deterministic",
                    "tie_break", "dsa_graph_safe", "row_starts", "page_table_row_starts",
                    "page_size", "out", "out_raw_indices")
        if tuple(inspect.signature(top_k_page_table_transform).parameters) != expected:
            raise TypeError("unsupported top_k_page_table_transform schema")
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise ExternalKernelContractError("flashinfer-python", "compressed index top-k", str(error)) from error
    return top_k_page_table_transform
