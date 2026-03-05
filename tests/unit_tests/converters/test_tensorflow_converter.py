"""Unit tests for TensorFlow conversion orchestration paths."""

from __future__ import annotations

from pathlib import Path
from sys import modules as sys_modules
from types import SimpleNamespace as types_SimpleNamespace

from pytest import MonkeyPatch as pytest_MonkeyPatch

from onnx_converter import api as api_module

from .conftest import mock_converter_dependencies


def _install_dummy_tf(monkeypatch: pytest_MonkeyPatch) -> None:
    dummy_tf = types_SimpleNamespace()

    class _Models:
        @staticmethod
        def load_model(path: str) -> str:
            return f"loaded:{path}"

    class _Keras:
        models = _Models

    dummy_tf.keras = _Keras

    monkeypatch.setitem(sys_modules, "tensorflow", dummy_tf)
    monkeypatch.setitem(sys_modules, "tf2onnx", types_SimpleNamespace())


def test_convert_tf_path_uses_savedmodel_dir(
    tmp_path: Path, monkeypatch: pytest_MonkeyPatch
) -> None:
    """Pass SavedModel directory path directly into TensorFlow converter."""
    model_path = tmp_path / "saved_model"
    model_path.mkdir()
    output_path = tmp_path / "out.onnx"

    _install_dummy_tf(monkeypatch)

    # Mock the converter to avoid importing real tf2onnx
    def fake_convert(**kwargs: object) -> str:
        # For SavedModel directories, the model parameter should be the path string
        assert kwargs["model"] == str(model_path)
        return str(output_path)

    monkeypatch.setattr("onnx_converter.convert_tensorflow_to_onnx", fake_convert)
    mock_converter_dependencies(monkeypatch, framework="tensorflow")

    out = api_module.convert_tf_path_to_onnx(
        model_path=model_path,
        output_path=output_path,
        opset_version=14,
    )

    assert out == output_path


def test_convert_tf_path_loads_file(
    tmp_path: Path, monkeypatch: pytest_MonkeyPatch
) -> None:
    """Load file-based TensorFlow model before conversion."""
    model_path = tmp_path / "model.h5"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    _install_dummy_tf(monkeypatch)

    # Mock the converter to avoid importing real tf2onnx
    def fake_convert(**kwargs: object) -> str:
        # For regular files, the loader should load the model
        # (checking for "loaded:" prefix)
        assert kwargs["model"] == f"loaded:{model_path}"
        return str(output_path)

    monkeypatch.setattr("onnx_converter.convert_tensorflow_to_onnx", fake_convert)
    mock_converter_dependencies(monkeypatch, framework="tensorflow")

    out = api_module.convert_tf_path_to_onnx(
        model_path=model_path,
        output_path=output_path,
        opset_version=14,
    )

    assert out == output_path
