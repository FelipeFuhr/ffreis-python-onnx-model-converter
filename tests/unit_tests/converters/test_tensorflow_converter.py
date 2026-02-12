from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from onnx_converter import api as api_module


def _install_dummy_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_tf = types.SimpleNamespace()

    class _Models:
        @staticmethod
        def load_model(path: str) -> str:
            return f"loaded:{path}"

    class _Keras:
        models = _Models

    dummy_tf.keras = _Keras

    monkeypatch.setitem(sys.modules, "tensorflow", dummy_tf)
    monkeypatch.setitem(sys.modules, "tf2onnx", types.SimpleNamespace())


def test_convert_tf_path_uses_savedmodel_dir(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "saved_model"
    model_path.mkdir()
    output_path = tmp_path / "out.onnx"

    _install_dummy_tf(monkeypatch)

    def fake_convert(*, model: Any, output_path: Any, opset_version: int) -> Any:
        assert model == str(model_path)
        return output_path

    monkeypatch.setattr(api_module, "_get_tensorflow_converter", lambda: fake_convert)

    out = api_module.convert_tf_path_to_onnx(
        model_path=model_path,
        output_path=output_path,
        opset_version=14,
    )

    assert out == output_path


def test_convert_tf_path_loads_file(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "model.h5"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    _install_dummy_tf(monkeypatch)

    def fake_convert(*, model: Any, output_path: Any, opset_version: int) -> Any:
        assert model == f"loaded:{model_path}"
        return output_path

    monkeypatch.setattr(api_module, "_get_tensorflow_converter", lambda: fake_convert)

    out = api_module.convert_tf_path_to_onnx(
        model_path=model_path,
        output_path=output_path,
        opset_version=14,
    )

    assert out == output_path
