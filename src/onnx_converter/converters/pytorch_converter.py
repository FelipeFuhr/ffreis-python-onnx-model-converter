"""PyTorch-to-ONNX conversion utilities."""

from __future__ import annotations

import os
from typing import Any
from typing import Optional

import torch
import torch.onnx


def convert_pytorch_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple[int, ...],
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
    opset_version: int = 14,
    **kwargs: Any,
) -> str:
    """Convert a PyTorch model to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance to convert.
    output_path : str
        Path where the ONNX model will be written.
    input_shape : tuple[int, ...]
        Shape of the dummy input tensor used during export.
    input_names : list, optional
        Input tensor names. Defaults to ``["input"]``.
    output_names : list, optional
        Output tensor names. Defaults to ``["output"]``.
    dynamic_axes : dict, optional
        Dynamic axis mapping passed to ``torch.onnx.export``.
    opset_version : int, default=14
        ONNX opset version used by the exporter.
    **kwargs
        Additional keyword arguments forwarded to ``torch.onnx.export``.

    Returns
    -------
    str
        Path to the saved ONNX model.
    """
    model.eval()

    dummy_input = torch.randn(*input_shape)

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        **kwargs
    )

    print(f"✓ PyTorch model successfully converted to ONNX: {output_path}")
    return output_path
