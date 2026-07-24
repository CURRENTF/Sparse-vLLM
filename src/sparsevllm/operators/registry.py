from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from sparsevllm.platforms.interface import DeviceCaps


SpecT = TypeVar("SpecT")
ProviderT = TypeVar("ProviderT", bound="OperatorProvider")


def runtime_version_at_least(
    version: str | None,
    minimum: tuple[int, int],
) -> bool:
    if version is None:
        return False
    parts = str(version).split(".")
    if len(parts) < 2:
        return False
    try:
        current = (int(parts[0]), int(parts[1]))
    except ValueError:
        return False
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
        return ResolvedProvider(selected, tuple(rejected))
