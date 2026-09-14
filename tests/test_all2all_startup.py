"""Optional transport failures must be startup errors, never runtime fallback."""

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sparsevllm.configs.moe_communication import normalize_moe_communication
from sparsevllm.kernels.external import deepep
from sparsevllm.operators.all2all import AllToAllOpSpec


def test_unselected_transport_does_not_load_optional_dependency(monkeypatch):
    check = Mock(side_effect=AssertionError("unselected optional dependency imported"))
    monkeypatch.setattr("sparsevllm.operators.all2all.check_all2all_dependency", check)
    for backend, dp in (("auto", 1), ("auto", 2), ("agrs", 2)):
        config = SimpleNamespace(
            moe_communication_backend=backend, data_parallel_size=dp
        )
        normalize_moe_communication(config)
    check.assert_not_called()


def test_selected_transport_dependency_failure_is_not_replaced(monkeypatch):
    check = Mock(side_effect=RuntimeError("extension ABI mismatch"))
    monkeypatch.setattr("sparsevllm.operators.all2all.check_all2all_dependency", check)
    config = SimpleNamespace(
        moe_communication_backend="all2all",
        data_parallel_size=2,
        expert_parallel_size=2,
    )
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        normalize_moe_communication(config)
    assert config.resolved_moe_communication_backend == "all2all"


def test_missing_package_and_incompatible_major_fail_before_import(monkeypatch):
    load = Mock()
    monkeypatch.setattr(deepep.importlib, "import_module", load)
    monkeypatch.setattr(
        deepep, "version", Mock(side_effect=PackageNotFoundError("deep_ep"))
    )
    with pytest.raises(RuntimeError, match="not installed"):
        deepep.load_deepep_v1()
    for incompatible in ("2.0.0", "2.0.0a1"):
        monkeypatch.setattr(deepep, "version", lambda _, v=incompatible: v)
        with pytest.raises(RuntimeError, match="found 2"):
            deepep.load_deepep_v1()
    load.assert_not_called()


def test_broken_extension_and_missing_graph_api_fail_explicitly(monkeypatch):
    monkeypatch.setattr(deepep, "version", lambda _: "1.2.1")
    monkeypatch.setattr(
        deepep.importlib, "import_module", Mock(side_effect=OSError("undefined symbol"))
    )
    with pytest.raises(RuntimeError, match="undefined symbol"):
        deepep.load_deepep_v1()
    monkeypatch.setattr(
        deepep.importlib, "import_module", lambda _: SimpleNamespace(Buffer=object)
    )
    with pytest.raises(RuntimeError, match="required public API"):
        deepep.load_deepep_v1()


def test_expert_partition_contract_rejects_fractional_ownership():
    with pytest.raises(ValueError, match="evenly partitioned"):
        AllToAllOpSpec(4, 2048, 7, 2, 8, torch.bfloat16, True)


def test_dispatch_failure_propagates_without_replacing_transport():
    from sparsevllm.distributed.moe_all2all import AllToAllMoeCommunication

    parallel = SimpleNamespace(uses_dp_attention=True, moe_tp_size=1, expert=object())
    spec = AllToAllOpSpec(2, 2048, 8, 2, 8, torch.bfloat16, True)
    comm = AllToAllMoeCommunication(parallel, spec)
    op = Mock()
    op.dispatch.side_effect = RuntimeError("communication launch failed")
    comm.op = op
    x = torch.empty(1, 2048, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="communication launch failed"):
        comm.run(
            x, route=lambda _: (None, None), experts=None, chunk_size=8, capacity=1
        )
    assert comm.op is op
    op.dispatch.assert_called_once()
    op.combine.assert_not_called()


@pytest.mark.parametrize("nvlink", [False, RuntimeError("topology query failed")])
def test_incompatible_topology_fails_before_creating_ipc_buffer(monkeypatch, nvlink):
    from sparsevllm.platforms.interface import PlatformEnum

    buffer = Mock()
    buffer.is_sm90_compiled.return_value = True
    monkeypatch.setattr(
        deepep, "load_deepep_v1", lambda: (SimpleNamespace(Buffer=buffer), "1.2.1")
    )
    platform = Mock()
    platform.get_device_caps.return_value = SimpleNamespace(
        platform=PlatformEnum.CUDA, compute_capability=(9, 0)
    )
    if isinstance(nvlink, Exception):
        platform.supports_nvlink_group.side_effect = nvlink
    else:
        platform.supports_nvlink_group.return_value = nvlink
    monkeypatch.setattr(deepep, "platforms", SimpleNamespace(current_platform=platform))
    monkeypatch.setattr(deepep.dist, "get_backend", lambda _: "nccl")

    def gather(output, value, group):
        if isinstance(value, tuple):
            output[:] = [(value[0], i, value[2]) for i in range(2)]
        else:
            output[:] = [value, value]

    monkeypatch.setattr(deepep.dist, "all_gather_object", gather)
    spec = AllToAllOpSpec(2, 2048, 8, 2, 8, torch.bfloat16, True)
    group = SimpleNamespace(size=2, rank=0, process_group=object())
    with pytest.raises(RuntimeError, match="NVLink validation failed"):
        deepep.DeepEPV1Normal(spec, group=group, device_index=0)
    buffer.assert_not_called()
