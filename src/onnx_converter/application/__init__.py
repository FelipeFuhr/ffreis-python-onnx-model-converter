"""Application-layer use-cases and option objects."""

from __future__ import annotations

from onnx_converter.application.options import (
    ConversionOptions,
    ParityOptions,
    PostprocessOptions,
)
from onnx_converter.application.results import ConversionResult


def build_conversion_options(*args: object, **kwargs: object) -> ConversionOptions:
    """Build typed conversion options via lazy use-case import."""
    from onnx_converter.application.use_cases import build_conversion_options as _impl

    return _impl(*args, **kwargs)


def convert_torch_file(*args: object, **kwargs: object) -> ConversionResult:
    """Convert PyTorch model artifact via lazy use-case import."""
    from onnx_converter.application.use_cases import convert_torch_file as _impl

    return _impl(*args, **kwargs)


def convert_tensorflow_file(*args: object, **kwargs: object) -> ConversionResult:
    """Convert TensorFlow model artifact via lazy use-case import."""
    from onnx_converter.application.use_cases import convert_tensorflow_file as _impl

    return _impl(*args, **kwargs)


def convert_sklearn_file(*args: object, **kwargs: object) -> ConversionResult:
    """Convert sklearn model artifact via lazy use-case import."""
    from onnx_converter.application.use_cases import convert_sklearn_file as _impl

    return _impl(*args, **kwargs)


def convert_custom_file(*args: object, **kwargs: object) -> ConversionResult:
    """Convert custom model artifact via lazy use-case import."""
    from onnx_converter.application.use_cases import convert_custom_file as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "ConversionOptions",
    "ParityOptions",
    "PostprocessOptions",
    "ConversionResult",
    "build_conversion_options",
    "convert_torch_file",
    "convert_tensorflow_file",
    "convert_sklearn_file",
    "convert_custom_file",
]
