import argparse
from types import SimpleNamespace

import pytest

from benchmark.vllm_microbench import _parse_positive_ints, _validate_args


def test_vllm_microbench_parses_unique_batch_sizes():
    assert _parse_positive_ints("1,2,4") == [1, 2, 4]


@pytest.mark.parametrize("value", ["", "0,1", "1,1"])
def test_vllm_microbench_rejects_invalid_batch_sizes(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_positive_ints(value)


def test_vllm_microbench_rejects_short_model_context():
    args = SimpleNamespace(
        input_len=1024,
        output_len=128,
        num_warmups=2,
        num_iters=5,
        tensor_parallel_size=2,
        max_model_len=1151,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.7,
    )

    with pytest.raises(ValueError, match=r"input_len \+ output_len"):
        _validate_args(args)
