from __future__ import annotations

import re
import weakref
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

from sparsevllm.platforms.interface import DeviceCaps
from sparsevllm.utils.log import logger


SpecT = TypeVar("SpecT")
ProviderT = TypeVar("ProviderT", bound="OperatorProvider")


_OPERATOR_BINDINGS: dict[str, weakref.WeakSet[object]] = {}
_BINDING_REPORTS: weakref.WeakKeyDictionary[object, BindingReport] = (
    weakref.WeakKeyDictionary()
)


def record_operator_binding(
    operator_type: str,
    provider: object,
    *,
    report: BindingReport | None = None,
) -> None:
    _OPERATOR_BINDINGS.setdefault(operator_type, weakref.WeakSet()).add(provider)
    if report is not None:
        _BINDING_REPORTS[provider] = report


def _implementation_name(provider: object) -> str:
    return (
        getattr(provider, "implementation_name", None)
        or getattr(provider, "name", None)
        or provider.provider_name
    )


@dataclass(frozen=True, slots=True)
class _FrozenMapping:
    items: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _FrozenSequence:
    items: tuple[object, ...]


def _freeze_snapshot(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _FrozenMapping(
            tuple(
                (field.name, _freeze_snapshot(getattr(value, field.name)))
                for field in fields(value)
            )
        )
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return _FrozenMapping(
            tuple(
                (str(key), _freeze_snapshot(item))
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: str(pair[0]),
                )
            )
        )
    if isinstance(value, (list, tuple)):
        return _FrozenSequence(tuple(_freeze_snapshot(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _thaw_snapshot(value: object) -> object:
    if isinstance(value, _FrozenMapping):
        return {key: _thaw_snapshot(item) for key, item in value.items}
    if isinstance(value, _FrozenSequence):
        return [_thaw_snapshot(item) for item in value.items]
    return value


@dataclass(frozen=True, slots=True)
class ProviderDecision:
    provider: str
    priority: int
    supported: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "priority": self.priority,
            "supported": self.supported,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BindingReport:
    operator_type: str
    spec_type: str
    spec: object
    device_caps: object
    selected_provider: str
    selected_priority: int
    selected_reason: str
    candidates: tuple[ProviderDecision, ...]
    provider_metadata: _FrozenMapping = _FrozenMapping(())

    @property
    def rejected(self) -> tuple[ProviderDecision, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.supported)

    def as_dict(self) -> dict[str, object]:
        return {
            "operator_type": self.operator_type,
            "spec_type": self.spec_type,
            "spec": _thaw_snapshot(self.spec),
            "device_caps": _thaw_snapshot(self.device_caps),
            "selected_provider": self.selected_provider,
            "selected_priority": self.selected_priority,
            "selected_reason": self.selected_reason,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "provider_metadata": _thaw_snapshot(self.provider_metadata),
        }


def operator_binding_reports() -> list[dict[str, object]]:
    """Return stable, JSON-ready decisions for every live resolver binding."""

    counts: dict[BindingReport, int] = {}
    for report in _BINDING_REPORTS.values():
        counts[report] = counts.get(report, 0) + 1
    rows = []
    for report, count in sorted(
        counts.items(),
        key=lambda item: (
            item[0].operator_type,
            item[0].selected_provider,
            repr(item[0].spec),
        ),
    ):
        row = report.as_dict()
        row["bound_provider_count"] = count
        rows.append(row)
    return rows


def operator_binding_report(provider: object) -> BindingReport | None:
    """Return the immutable resolver decision associated with one provider."""

    return _BINDING_REPORTS.get(provider)


def operator_runtime_stats() -> dict[str, list[dict[str, object]]]:
    """Return aggregate runtime kernel paths for every live bound provider."""

    result: dict[str, list[dict[str, object]]] = {}
    for operator_type, providers in sorted(_OPERATOR_BINDINGS.items()):
        grouped: dict[str, list[object]] = {}
        for provider in providers:
            grouped.setdefault(_implementation_name(provider), []).append(provider)
        entries: list[dict[str, object]] = []
        for implementation, bound_providers in sorted(grouped.items()):
            kernel_paths: dict[str, dict[str, int]] = {}
            fallback_reasons: dict[str, int] = {}
            instrumented = 0
            for provider in bound_providers:
                stats_fn = getattr(provider, "runtime_kernel_stats", None)
                if not callable(stats_fn):
                    continue
                instrumented += 1
                stats = stats_fn()
                for path, counts in stats.get("kernel_paths", {}).items():
                    aggregate = kernel_paths.setdefault(str(path), {})
                    for key, count in counts.items():
                        aggregate[str(key)] = int(aggregate.get(str(key), 0)) + int(count)
                for reason, count in stats.get("fallback_reasons", {}).items():
                    fallback_reasons[str(reason)] = (
                        int(fallback_reasons.get(str(reason), 0)) + int(count)
                    )
            entries.append(
                {
                    "implementation": implementation,
                    "bound_provider_count": len(bound_providers),
                    "instrumented_provider_count": instrumented,
                    "kernel_paths": {
                        path: dict(sorted(counts.items()))
                        for path, counts in sorted(kernel_paths.items())
                    },
                    "fallback_reasons": dict(sorted(fallback_reasons.items())),
                }
            )
        if entries:
            result[operator_type] = entries
    return result


def log_operator_implementations() -> None:
    entries = sorted(
        (
            operator_type,
            ", ".join(
                sorted({_implementation_name(provider) for provider in providers})
            ),
        )
        for operator_type, providers in _OPERATOR_BINDINGS.items()
        if providers
    )
    if not entries:
        return
    rows = "\n".join(
        f"  {operator_type}: {implementation}"
        for operator_type, implementation in entries
    )
    logger.info("Operator implementations:\n{}", rows)
    runtime_rows = []
    for operator_type, implementations in operator_runtime_stats().items():
        for implementation in implementations:
            for path, counts in implementation["kernel_paths"].items():
                runtime_rows.append(
                    f"  {operator_type}/{implementation['implementation']}/{path}: "
                    + ", ".join(
                        f"{key}={value}" for key, value in counts.items()
                    )
                )
    if runtime_rows:
        logger.info("Operator runtime kernels:\n{}", "\n".join(runtime_rows))
    binding_rows = []
    for report in operator_binding_reports():
        rejected = [
            f"{candidate['provider']}: {candidate['reason']}"
            for candidate in report["candidates"]
            if not candidate["supported"]
        ]
        binding_rows.append(
            f"  {report['operator_type']}: selected={report['selected_provider']} "
            f"count={report['bound_provider_count']} "
            f"rejected={rejected or 'none'}"
        )
    if binding_rows:
        logger.info("Operator binding decisions:\n{}", "\n".join(binding_rows))


def runtime_version_at_least(
    version: str | None,
    minimum: tuple[int, int],
) -> bool:
    if version is None:
        return False
    match = re.match(r"^\s*(\d+)\.(\d+)", str(version))
    if match is None:
        return False
    current = tuple(map(int, match.groups()))
    return current >= minimum


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reason: str

    @classmethod
    def yes(cls, reason: str = "supported") -> "SupportResult":
        return cls(True, reason)

    @classmethod
    def no(cls, reason: str) -> "SupportResult":
        return cls(False, reason)


class OperatorProvider(Protocol[SpecT]):
    name: str
    priority: int

    @classmethod
    def supports(cls, spec: SpecT, caps: DeviceCaps) -> SupportResult: ...


class OpRegistry(Generic[SpecT, ProviderT]):
    def __init__(self, family: str) -> None:
        self.family = str(family)
        self._providers: dict[str, type[ProviderT]] = {}

    def register(self, provider: type[ProviderT]) -> type[ProviderT]:
        name = str(provider.name)
        if name in self._providers:
            raise ValueError(
                f"Provider {name!r} is already registered for {self.family!r}."
            )
        self._providers[name] = provider
        return provider

    @property
    def providers(self) -> tuple[type[ProviderT], ...]:
        return tuple(self._providers.values())


@dataclass(frozen=True)
class ResolvedProvider(Generic[ProviderT]):
    provider: ProviderT
    rejected: tuple[tuple[str, str], ...]
    report: BindingReport


class OpResolver(Generic[SpecT, ProviderT]):
    def __init__(self, registry: OpRegistry[SpecT, ProviderT]) -> None:
        self.registry = registry

    def resolve(
        self,
        spec: SpecT,
        caps: DeviceCaps,
        **provider_kwargs,
    ) -> ResolvedProvider[ProviderT]:
        supported: list[tuple[type[ProviderT], SupportResult]] = []
        rejected: list[tuple[str, str]] = []
        decisions: list[ProviderDecision] = []
        for provider in self.registry.providers:
            result = provider.supports(spec, caps)
            decisions.append(
                ProviderDecision(
                    provider=str(provider.name),
                    priority=int(provider.priority),
                    supported=result.supported,
                    reason=result.reason,
                )
            )
            if result.supported:
                supported.append((provider, result))
            else:
                rejected.append((provider.name, result.reason))
        if not supported:
            details = "; ".join(f"{name}: {reason}" for name, reason in rejected)
            raise RuntimeError(
                f"No {self.registry.family} provider supports spec={spec!r} on "
                f"device={caps.device_name!r}: {details or 'no providers registered'}."
            )
        supported.sort(key=lambda item: (-int(item[0].priority), item[0].name))
        selected_type, selected_support = supported[0]
        bind_fn = getattr(selected_type, "bind", None)
        if callable(bind_fn):
            selected = bind_fn(spec, caps, **provider_kwargs)
        else:
            selected = selected_type(**provider_kwargs)
        metadata_fn = getattr(selected, "binding_metadata", None)
        metadata = metadata_fn() if callable(metadata_fn) else {}
        if not isinstance(metadata, dict):
            raise TypeError(
                f"Provider {selected_type.name!r} binding_metadata() must return "
                f"a dict, got {type(metadata).__name__}."
            )
        report = BindingReport(
            operator_type=self.registry.family,
            spec_type=type(spec).__name__,
            spec=_freeze_snapshot(spec),
            device_caps=_freeze_snapshot(caps),
            selected_provider=str(selected_type.name),
            selected_priority=int(selected_type.priority),
            selected_reason=selected_support.reason,
            candidates=tuple(
                sorted(decisions, key=lambda decision: decision.provider)
            ),
            provider_metadata=_freeze_snapshot(metadata),
        )
        record_operator_binding(self.registry.family, selected, report=report)
        return ResolvedProvider(selected, tuple(rejected), report)
