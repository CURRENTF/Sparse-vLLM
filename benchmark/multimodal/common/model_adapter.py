from __future__ import annotations

from typing import Protocol, Any


class MultimodalModelAdapter(Protocol):
    name: str

    def load_model(self, args: Any, dtype: Any, device: Any) -> tuple[Any, dict | None]:
        ...
