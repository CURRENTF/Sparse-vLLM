from dataclasses import dataclass

import pytest

from sparsevllm.operators.registry import OpRegistry, OpResolver, SupportResult
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

    resolved = OpResolver(registry).resolve(_Spec(), _caps())

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
