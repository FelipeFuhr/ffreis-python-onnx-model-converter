"""Converter namespace with lazy imports for optional dependency isolation."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"


def convert_pytorch_to_onnx(
    model: Any,
    output_path: str,
    input_shape: tuple[int, ...],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 14,
    **kwargs: Any,
) -> str:
    from .pytorch_converter import convert_pytorch_to_onnx as _impl

    return _impl(
        model=model,
        output_path=output_path,
        input_shape=input_shape,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        **kwargs,
    )


def convert_tensorflow_to_onnx(
    model: Any,
    output_path: str,
    input_signature: Any = None,
    opset_version: int = 14,
    **kwargs: Any,
) -> str:
    from .tensorflow_converter import convert_tensorflow_to_onnx as _impl

    return _impl(
        model=model,
        output_path=output_path,
        input_signature=input_signature,
        opset_version=opset_version,
        **kwargs,
    )


def convert_sklearn_to_onnx(
    model: Any,
    output_path: str,
    initial_types: Any = None,
    target_opset: int | None = None,
    **kwargs: Any,
) -> str:
    from .sklearn_converter import convert_sklearn_to_onnx as _impl

    return _impl(
        model=model,
        output_path=output_path,
        initial_types=initial_types,
        target_opset=target_opset,
        **kwargs,
    )


__all__ = [
    "convert_pytorch_to_onnx",
    "convert_tensorflow_to_onnx",
    "convert_sklearn_to_onnx",
]
