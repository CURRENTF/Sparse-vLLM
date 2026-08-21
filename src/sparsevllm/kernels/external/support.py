from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KernelFamilyState(str, Enum):
    ABSENT = "absent"
    BROKEN = "broken"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class KernelFamilyHealth:
    family: str
    state: KernelFamilyState
    version: str | None
    reason: str

    @property
    def ready(self) -> bool:
        return self.state is KernelFamilyState.READY


class ExternalKernelError(RuntimeError):
    """Base error for an invalid external-kernel environment or contract."""


class ExternalKernelFamilyError(ExternalKernelError):
    def __init__(self, health: KernelFamilyHealth, *, feature: str) -> None:
        self.health = health
        self.feature = str(feature)
        super().__init__(
            f"{health.family} family is {health.state.value} for {self.feature}: "
            f"{health.reason}"
        )


class ExternalKernelContractError(ExternalKernelError):
    def __init__(self, family: str, feature: str, reason: str) -> None:
        self.family = str(family)
        self.feature = str(feature)
        self.reason = str(reason)
        super().__init__(f"{self.family} {self.feature} contract is invalid: {self.reason}")


__all__ = [
    "ExternalKernelContractError",
    "ExternalKernelError",
    "ExternalKernelFamilyError",
    "KernelFamilyHealth",
    "KernelFamilyState",
]
