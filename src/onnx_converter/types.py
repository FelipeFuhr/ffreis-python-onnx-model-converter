"""Shared type aliases and protocols for converter modules."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

FrameworkKind = Literal["pytorch", "tensorflow", "sklearn"]


class ModelArtifact(Protocol):
    """Marker protocol for in-memory framework model objects."""


class TensorSpecLike(Protocol):
    """Marker protocol for TensorFlow tensor signatures."""


class SklearnInitialTypeLike(Protocol):
    """Marker protocol for skl2onnx initial type declarations."""


OptionScalar = str | int | float | bool | None | Path
OptionValue = (
    OptionScalar
    | tuple["OptionValue", ...]
    | list["OptionValue"]
    | dict[str, "OptionValue"]
    | dict[int, "OptionValue"]
)
OptionMap = Mapping[str, OptionValue]
MutableOptionMap = dict[str, OptionValue]
