"""TensorFlow/Keras-to-ONNX conversion utilities."""

from __future__ import annotations

import os
from typing import Any
from typing import Optional

import tensorflow as tf
import tf2onnx


def convert_tensorflow_to_onnx(
    model: str | tf.keras.Model,
    output_path: str,
    input_signature: Optional[list[tf.TensorSpec]] = None,
    opset_version: int = 14,
    **kwargs: Any,
) -> str:
    """Convert a TensorFlow or Keras model to ONNX format.

    Parameters
    ----------
    model
        TensorFlow/Keras model instance or SavedModel path.
    output_path : str
        Path where the ONNX model will be written.
    input_signature : list[tf.TensorSpec], optional
        Input tensor signature passed to ``tf2onnx``.
    opset_version : int, default=14
        ONNX opset version used by ``tf2onnx``.
    **kwargs
        Additional keyword arguments forwarded to ``tf2onnx.convert``.

    Returns
    -------
    str
        Path to the saved ONNX model.

    Raises
    ------
    ValueError
        If ``model`` is neither a SavedModel path nor a ``tf.keras.Model``.
    """
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    if isinstance(model, str):
        tf2onnx.convert.from_saved_model(
            model,
            input_signature=input_signature,
            opset=opset_version,
            output_path=output_path,
            **kwargs
        )
    elif isinstance(model, tf.keras.Model):
        if input_signature is None:
            if hasattr(model, "input_shape"):
                input_shape = model.input_shape
                if isinstance(input_shape, list):
                    input_signature = [
                        tf.TensorSpec(shape, tf.float32, name=f"input_{i}")
                        for i, shape in enumerate(input_shape)
                    ]
                else:
                    input_signature = [tf.TensorSpec(input_shape, tf.float32, name="input")]

        tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset_version,
            output_path=output_path,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    print(f"✓ TensorFlow/Keras model successfully converted to ONNX: {output_path}")
    return output_path
