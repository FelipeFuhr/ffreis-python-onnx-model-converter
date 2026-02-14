from __future__ import annotations

from pathlib import Path

from onnx_converter.adapters.converters import (
    SklearnModelConverter,
    TensorflowModelConverter,
    TorchModelConverter,
)


def test_torch_adapter_roundtrip_contract(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "out.onnx"
    called = {}

    def fake_convert(**kwargs):
        called.update(kwargs)
        return str(out)

    import onnx_converter

    monkeypatch.setattr(onnx_converter, "convert_pytorch_to_onnx", fake_convert)

    result = TorchModelConverter().convert(
        model=object(),
        output_path=out,
        options={
            "input_shape": (1, 4),
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": {"input": {0: "batch"}},
            "opset_version": 14,
        },
    )

    assert result == out
    assert called["output_path"] == str(out)
    assert called["input_shape"] == (1, 4)


def test_tensorflow_adapter_roundtrip_contract(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "out.onnx"
    called = {}

    def fake_convert(**kwargs):
        called.update(kwargs)
        return str(out)

    import onnx_converter

    monkeypatch.setattr(onnx_converter, "convert_tensorflow_to_onnx", fake_convert)

    result = TensorflowModelConverter().convert(
        model=object(),
        output_path=out,
        options={"opset_version": 14, "input_signature": None},
    )

    assert result == out
    assert called["output_path"] == str(out)
    assert called["opset_version"] == 14


def test_sklearn_adapter_roundtrip_contract(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "out.onnx"
    called = {}

    def fake_convert(**kwargs):
        called.update(kwargs)
        return str(out)

    import onnx_converter

    monkeypatch.setattr(onnx_converter, "convert_sklearn_to_onnx", fake_convert)

    result = SklearnModelConverter().convert(
        model=object(),
        output_path=out,
        options={
            "n_features": 4,
            "target_opset": 14,
            "initial_types": [("input", object())],
        },
    )

    assert result == out
    assert called["output_path"] == str(out)
    assert called["target_opset"] == 14
