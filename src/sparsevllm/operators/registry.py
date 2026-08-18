from __future__ import annotations

import re
import weakref
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from sparsevllm.platforms.interface import DeviceCaps
from sparsevllm.utils.log import logger


SpecT = TypeVar("SpecT")
ProviderT = TypeVar("ProviderT", bound="OperatorProvider")


_OPERATOR_BINDINGS: dict[str, weakref.WeakSet[object]] = {}


def record_operator_binding(operator_type: str, provider: object) -> None:
    _OPERATOR_BINDINGS.setdefault(operator_type, weakref.WeakSet()).add(provider)


def _implementation_name(provider: object) -> str:
    return (
        getattr(provider, "implementation_name", None)
        or getattr(provider, "name", None)
        or provider.provider_name
    )


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


class OpResolver(Generic[SpecT, ProviderT]):
    def __init__(self, registry: OpRegistry[SpecT, ProviderT]) -> None:
        self.registry = registry

    def resolve(
        self,
        spec: SpecT,
        caps: DeviceCaps,
        **provider_kwargs,
    ) -> ResolvedProvider[ProviderT]:
        supported: list[type[ProviderT]] = []
        rejected: list[tuple[str, str]] = []
        for provider in self.registry.providers:
            result = provider.supports(spec, caps)
            if result.supported:
                supported.append(provider)
            else:
                rejected.append((provider.name, result.reason))
        if not supported:
            details = "; ".join(f"{name}: {reason}" for name, reason in rejected)
            raise RuntimeError(
                f"No {self.registry.family} provider supports spec={spec!r} on "
                f"device={caps.device_name!r}: {details or 'no providers registered'}."
            )
        supported.sort(key=lambda provider: (-int(provider.priority), provider.name))
        selected = supported[0](**provider_kwargs)
        record_operator_binding(self.registry.family, selected)
        return ResolvedProvider(selected, tuple(rejected))
