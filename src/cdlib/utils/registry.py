"""Generic component registry.

The whole team's parallelism depends on this pattern (see CLAUDE.md
"The registry pattern"): every pluggable component is one new file plus
one registry line. Nobody edits a shared builder to add a component.
"""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    """A name -> constructor mapping with duplicate-key protection.

    Usage:
        ENCODER_REGISTRY = Registry("encoder")

        @ENCODER_REGISTRY.register("resnet18")
        class ResNet18Encoder(nn.Module):
            ...

        model = ENCODER_REGISTRY.build("resnet18", pretrained=True)
    """

    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def _decorator(cls_or_fn: Callable[..., T]) -> Callable[..., T]:
            if key in self._entries:
                raise KeyError(
                    f"{key!r} is already registered in the {self.name!r} registry "
                    f"(existing: {self._entries[key]!r}). Registry keys must be unique."
                )
            self._entries[key] = cls_or_fn
            return cls_or_fn

        return _decorator

    def get(self, key: str) -> Callable[..., T]:
        if key not in self._entries:
            raise KeyError(
                f"{key!r} not found in the {self.name!r} registry. "
                f"Available keys: {sorted(self._entries)}"
            )
        return self._entries[key]

    def build(self, key: str, **kwargs) -> T:
        return self.get(key)(**kwargs)

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, keys={self.keys()})"
