#!/usr/bin/env python3
"""Example script for converting a tiny TensorFlow/Keras model to ONNX."""

from __future__ import annotations

from pathlib import Path

from numpy import abs as np_abs
from numpy import allclose as np_allclose
from numpy import array as np_array
from numpy import float32 as np_float32
from numpy import max as np_max
from onnx import checker as onnx_checker
from onnx import load as onnx_load
from onnxruntime import InferenceSession as ort_InferenceSession
from tensorflow import TensorSpec as tf_TensorSpec
from tensorflow import float32 as tf_float32
from tensorflow import keras as tf_keras
from tensorflow import random as tf_random

from onnx_converter import convert_tensorflow_to_onnx


def main() -> None:
    """Convert and verify a tiny Keras model."""
    print("=" * 60)
    print("TensorFlow/Keras to ONNX Conversion Example")
    print("=" * 60)

    tf_random.set_seed(7)
    model = tf_keras.Sequential(
        [
            tf_keras.layers.Input(shape=(4,), name="input"),
            tf_keras.layers.Dense(8, activation="relu"),
            tf_keras.layers.Dense(3, activation="softmax"),
        ]
    )

    output_path = Path("outputs/tiny_keras.onnx")
    output_path.parent.mkdir(exist_ok=True)

    convert_tensorflow_to_onnx(
        model=model,
        output_path=str(output_path),
        input_signature=[tf_TensorSpec((None, 4), tf_float32, name="input")],
        opset_version=14,
    )

    if not output_path.exists():
        raise SystemExit("FAIL: ONNX file was not created.")

    onnx_model = onnx_load(str(output_path))
    onnx_checker.check_model(onnx_model)
    print("ONNX graph validated.")

    batch = np_array(
        [[0.1, -0.2, 0.3, 0.4], [0.5, -0.6, 0.7, -0.8]],
        dtype=np_float32,
    )

    tf_out = model(batch, training=False).numpy()
    session = ort_InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: batch})[0]

    if tf_out.shape != onnx_out.shape:
        raise SystemExit(
            f"FAIL: shape mismatch (tf={tf_out.shape}, onnx={onnx_out.shape})."
        )

    max_abs_diff = float(np_max(np_abs(tf_out - onnx_out)))
    print(f"Max abs diff: {max_abs_diff:.8f}")
    if not np_allclose(tf_out, onnx_out, atol=1e-5, rtol=1e-4):
        raise SystemExit(
            "FAIL: output mismatch "
            f"(max_abs_diff={max_abs_diff:.8f}, atol=1e-5, rtol=1e-4)."
        )

    print(f"PASS: {output_path}")


if __name__ == "__main__":
    main()
