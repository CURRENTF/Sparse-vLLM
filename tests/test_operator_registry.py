from dataclasses import dataclass
from unittest.mock import patch

import pytest

import sparsevllm.operators.registry as operator_registry
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
    log_operator_implementations,
    record_operator_binding,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class _Spec:
    enabled: bool = True


def _caps() -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="test",
        compute_capability=(9, 0),
    )


def test_resolver_uses_deterministic_supported_priority():
    registry = OpRegistry("_test")

    @registry.register
    class Portable:
        name = "portable"
        priority = 10

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register
    class Specialized:
        name = "specialized"
        priority = 20

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes() if spec.enabled else SupportResult.no("disabled")

    with patch.dict(operator_registry._OPERATOR_BINDINGS, {}, clear=True):
        resolved = OpResolver(registry).resolve(_Spec(), _caps())
        assert resolved.provider in operator_registry._OPERATOR_BINDINGS["_test"]

    assert resolved.provider.name == "specialized"


def test_resolver_reports_every_rejection():
    registry = OpRegistry("_test")

    @registry.register
    class First:
        name = "first"
        priority = 10

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.no("wrong dtype")

    @registry.register
    class Second:
        name = "second"
        priority = 20

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.no("wrong architecture")

    with pytest.raises(RuntimeError, match="wrong dtype.*wrong architecture"):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_registry_rejects_duplicate_provider_names():
    registry = OpRegistry("_test")

    @registry.register
    class First:
        name = "same"
        priority = 1

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    with pytest.raises(ValueError, match="already registered"):

        @registry.register
        class Second:
            name = "same"
            priority = 2

            @classmethod
            def supports(cls, spec, caps):
                return SupportResult.yes()


def test_equal_priority_is_resolved_by_provider_name():
    registry = OpRegistry("_test")

    @registry.register
    class Zulu:
        name = "zulu"
        priority = 10

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    @registry.register
    class Alpha:
        name = "alpha"
        priority = 10

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    resolved = OpResolver(registry).resolve(_Spec(), _caps())

    assert resolved.provider.name == "alpha"


def test_resolver_falls_back_and_preserves_rejection_diagnostics():
    registry = OpRegistry("_test")

    @registry.register
    class Specialized:
        name = "specialized"
        priority = 100

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.no("missing optional library")

    @registry.register
    class Portable:
        name = "portable"
        priority = 10

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    resolved = OpResolver(registry).resolve(_Spec(), _caps())

    assert resolved.provider.name == "portable"
    assert resolved.rejected == (("specialized", "missing optional library"),)


def test_operator_organization_logs_live_bound_implementations():
    class Provider:
        def __init__(self, implementation_name):
            self.implementation_name = implementation_name

    attention = Provider("triton")
    linear = Provider("flashinfer_sm90")
    linear_fallback = Provider("triton")

    with (
        patch.dict(operator_registry._OPERATOR_BINDINGS, {}, clear=True),
        patch("sparsevllm.operators.registry.logger.info") as log_info,
    ):
        record_operator_binding("Attention", attention)
        record_operator_binding("block-scaled FP8 Linear", linear)
        record_operator_binding("block-scaled FP8 Linear", linear_fallback)

        log_operator_implementations(3)

    log_info.assert_called_once_with(
        "Operator implementations (rank {}):\n{}",
        3,
        "  Attention: triton\n"
        "  block-scaled FP8 Linear: flashinfer_sm90, triton",
    )


def test_resolver_forwards_provider_constructor_arguments():
    registry = OpRegistry("_test")

    @registry.register
    class Configured:
        name = "configured"
        priority = 1

        def __init__(self, marker):
            self.marker = marker

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    resolved = OpResolver(registry).resolve(_Spec(), _caps(), marker="ready")

    assert resolved.provider.marker == "ready"


def test_empty_registry_fails_with_explicit_reason():
    with pytest.raises(RuntimeError, match="no providers registered"):
        OpResolver(OpRegistry("_empty")).resolve(_Spec(), _caps())
