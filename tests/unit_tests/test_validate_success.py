"""Cover the validate happy path: real ONNX model passes checker + ORT."""

from __future__ import annotations

from pathlib import Path

import pytest
from onnx import TensorProto, helper
from onnx import save as onnx_save

from onnx_converter.validate import validate_onnx_if_requested

pytest.importorskip("onnxruntime")


def _write_minimal_onnx(path: Path) -> None:
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
    )
    onnx_save(helper.make_model(graph, producer_name="t"), str(path))


def test_validate_real_onnx_model_passes(tmp_path: Path) -> None:
    """Confirms the happy path runs end-to-end without raising."""
    model_path = tmp_path / "model.onnx"
    _write_minimal_onnx(model_path)
    # Should not raise.
    validate_onnx_if_requested(model_path, validate=True)
