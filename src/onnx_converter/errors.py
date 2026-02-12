"""Custom error types for ONNX conversion workflows."""

from __future__ import annotations


class ConversionError(RuntimeError):
    """Base user-facing conversion failure."""

    exit_code: int = 1


class DependencyError(ConversionError):
    """Raised when optional runtime dependencies are unavailable."""

    exit_code = 2


class UnsafeLoadError(ConversionError):
    """Raised when a conversion path requires unsafe deserialization."""

    exit_code = 3


class UnsupportedModelError(ConversionError):
    """Raised when artifact/model type cannot be handled."""

    exit_code = 4


class ParityError(ConversionError):
    """Raised when pre/post conversion parity checks fail."""

    exit_code = 5


class PostprocessError(ConversionError):
    """Raised when ONNX post-processing fails."""

    exit_code = 6


class PluginError(ConversionError):
    """Raised when converter plugin discovery/loading/execution fails."""

    exit_code = 7
