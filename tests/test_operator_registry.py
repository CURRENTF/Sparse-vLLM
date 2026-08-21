import json
import weakref
from dataclasses import dataclass
from unittest.mock import patch

import pytest

import sparsevllm.operators.registry as operator_registry
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
    log_operator_implementations,
    operator_binding_report,
    operator_binding_reports,
    operator_runtime_stats,
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

    with (
        patch.dict(operator_registry._OPERATOR_BINDINGS, {}, clear=True),
        patch.object(
            operator_registry,
            "_BINDING_REPORTS",
            weakref.WeakKeyDictionary(),
        ),
    ):
        resolved = OpResolver(registry).resolve(_Spec(), _caps())
        assert resolved.provider in operator_registry._OPERATOR_BINDINGS["_test"]
        assert operator_binding_report(resolved.provider) is resolved.report
        reports = operator_binding_reports()

    assert resolved.provider.name == "specialized"
    assert resolved.report.as_dict() == {
        "operator_type": "_test",
        "spec_type": "_Spec",
        "spec": {"enabled": True},
        "device_caps": {
            "platform": "CUDA",
            "device_type": "cuda",
            "device_index": 0,
            "device_name": "test",
            "compute_capability": [9, 0],
            "runtime_version": None,
            "supports_graph_capture": False,
            "supports_torch_compile": False,
            "supports_triton": False,
            "supports_pin_memory": False,
            "supports_bfloat16": False,
            "supports_native_fp8": False,
        },
        "selected_provider": "specialized",
        "selected_priority": 20,
        "selected_reason": "supported",
        "candidates": [
            {
                "provider": "portable",
                "priority": 10,
                "supported": True,
                "reason": "supported",
            },
            {
                "provider": "specialized",
                "priority": 20,
                "supported": True,
                "reason": "supported",
            },
        ],
        "provider_metadata": {},
    }
    assert reports == [{**resolved.report.as_dict(), "bound_provider_count": 1}]
    json.dumps(reports)


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

        log_operator_implementations()

    log_info.assert_called_once_with(
        "Operator implementations:\n{}",
        "  Attention: triton\n"
        "  block-scaled FP8 Linear: flashinfer_sm90, triton",
    )


def test_operator_runtime_stats_aggregate_live_provider_kernel_paths():
    class Provider:
        name = "composite"

        def __init__(self, eager: int, captured: int, fallback: int):
            self.eager = eager
            self.captured = captured
            self.fallback = fallback

        def runtime_kernel_stats(self):
            return {
                "kernel_paths": {
                    "tilelang_score": {
                        "eager_dispatches": self.eager,
                        "cuda_graph_capture_dispatches": self.captured,
                    }
                },
                "fallback_reasons": {"noncontiguous:output": self.fallback},
            }

    first = Provider(2, 1, 0)
    second = Provider(3, 4, 1)
    with patch.dict(operator_registry._OPERATOR_BINDINGS, {}, clear=True):
        record_operator_binding("MLA attention", first)
        record_operator_binding("MLA attention", second)

        stats = operator_runtime_stats()

    assert stats == {
        "MLA attention": [
            {
                "implementation": "composite",
                "bound_provider_count": 2,
                "instrumented_provider_count": 2,
                "kernel_paths": {
                    "tilelang_score": {
                        "cuda_graph_capture_dispatches": 5,
                        "eager_dispatches": 5,
                    }
                },
                "fallback_reasons": {"noncontiguous:output": 1},
            }
        ]
    }


def test_resolver_forwards_provider_constructor_arguments():
    registry = OpRegistry("_test")

    @registry.register
    class Configured:
        name = "configured"
        priority = 1

        def __init__(self, marker):
            self.marker = marker

        def binding_metadata(self):
            return {
                "empty_mapping": {},
                "empty_sequence": [],
                "layout_id": "test_layout",
                "profile": {"status": "generic"},
            }

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

    resolved = OpResolver(registry).resolve(_Spec(), _caps(), marker="ready")

    assert resolved.provider.marker == "ready"
    assert resolved.report.as_dict()["provider_metadata"] == {
        "empty_mapping": {},
        "empty_sequence": [],
        "layout_id": "test_layout",
        "profile": {"status": "generic"},
    }


def test_resolver_rejects_invalid_binding_metadata() -> None:
    registry = OpRegistry("_test")

    @registry.register
    class Invalid:
        name = "invalid"
        priority = 1

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

        def binding_metadata(self):
            return "not structured"

    with pytest.raises(TypeError, match="binding_metadata.*dict"):
        OpResolver(registry).resolve(_Spec(), _caps())


def test_resolver_uses_provider_bind_hook_for_prepared_plans() -> None:
    registry = OpRegistry("_test")

    @registry.register
    class Prepared:
        name = "prepared"
        priority = 1

        def __init__(self, spec, marker):
            self.spec = spec
            self.marker = marker

        @classmethod
        def supports(cls, spec, caps):
            return SupportResult.yes()

        @classmethod
        def bind(cls, spec, caps, *, marker):
            assert caps.device_name == "test"
            return cls(spec, marker)

    resolved = OpResolver(registry).resolve(_Spec(), _caps(), marker="ready")

    assert resolved.provider.spec == _Spec()
    assert resolved.provider.marker == "ready"


def test_empty_registry_fails_with_explicit_reason():
    with pytest.raises(RuntimeError, match="no providers registered"):
        OpResolver(OpRegistry("_empty")).resolve(_Spec(), _caps())
