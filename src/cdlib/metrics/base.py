"""torchmetrics-style metric contract.

Vendored from the team's merged `change-detect` monorepo — the frozen
metric interface every `cdlib.metrics.*` class implements. Not this
repo's own contribution; included so this repo's metrics run standalone.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """Frozen metric interface (CLAUDE.md)."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None: ...

    @abstractmethod
    def compute(self) -> dict[str, float]: ...
