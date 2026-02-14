from __future__ import annotations

import pytest


def test_tensorflow_convert(tmp_path) -> None:
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    pytest.importorskip("onnxruntime")

    from onnx_converter.converters.tensorflow_converter import (
        convert_tensorflow_to_onnx,
    )

    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2)]
    )
    output_path = tmp_path / "model.onnx"

    out = convert_tensorflow_to_onnx(
        model=model,
        output_path=str(output_path),
        opset_version=14,
    )

    assert output_path.exists()
    assert str(output_path) == out
