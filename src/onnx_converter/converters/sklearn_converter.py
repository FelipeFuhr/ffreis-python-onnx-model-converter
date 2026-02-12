"""Scikit-learn-to-ONNX conversion utilities."""

from __future__ import annotations

import os
from typing import Any
from typing import Optional

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def convert_sklearn_to_onnx(
    model: Any,
    output_path: str,
    initial_types: Optional[list[tuple[str, Any]]] = None,
    target_opset: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """Convert a scikit-learn model or pipeline to ONNX format.

    Parameters
    ----------
    model
        Scikit-learn model or pipeline instance to convert.
    output_path : str
        Path where the ONNX model will be written.
    initial_types : list[tuple[str, Any]], optional
        Input type declarations expected by ``skl2onnx``.
        When omitted, input types are inferred from ``model.n_features_in_``.
    target_opset : int, optional
        ONNX opset version used by ``skl2onnx``.
    **kwargs
        Additional keyword arguments forwarded to ``convert_sklearn``.

    Returns
    -------
    str
        Path to the saved ONNX model.

    Raises
    ------
    ValueError
        If input types cannot be inferred and ``initial_types`` is not provided.
    """
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    if initial_types is None:
        if hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
            initial_types = [("input", FloatTensorType([None, n_features]))]
        else:
            raise ValueError(
                "Could not infer input types. Please provide 'initial_types' parameter.\n"
                "Example: [('input', FloatTensorType([None, n_features]))]"
            )

    onx = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=target_opset,
        **kwargs
    )

    with open(output_path, "wb") as handle:
        handle.write(onx.SerializeToString())

    print(f"✓ Scikit-learn model successfully converted to ONNX: {output_path}")
    return output_path
