from __future__ import annotations

from pathlib import Path

import pytest

from onnx_converter.errors import PluginError
from onnx_converter.plugins import builtins
from onnx_converter.plugins.builtins import SklearnFilePlugin


class DummyLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, bool]] = []

    def load(self, model_path: Path, allow_unsafe: bool = False) -> object:
        self.calls.append((model_path, allow_unsafe))
        return object()


class DummyConverter:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.calls: list[dict[str, object]] = []

    def convert(
        self,
        model: object,
        output_path: Path,
        options: dict[str, object],
    ) -> Path:
        del model
        self.calls.append({"output_path": output_path, "options": options})
        return self.out


class DummyParity:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def check(self, model: object, onnx_path: Path, parity: object) -> None:
        self.calls.append((model, onnx_path, parity))


class DummyPost:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def run(
        self,
        output_path: Path,
        source_path: Path,
        framework: str,
        config_metadata: dict[str, str],
        options: object,
    ) -> None:
        self.calls.append(
            {
                "output_path": output_path,
                "source_path": source_path,
                "framework": framework,
                "config_metadata": config_metadata,
                "options": options,
            }
        )


def test_requires_n_features() -> None:
    plugin = SklearnFilePlugin()
    with pytest.raises(PluginError):
        plugin.convert(Path("model.joblib"), Path("out.onnx"), options={})


def test_rejects_bad_metadata_type() -> None:
    plugin = SklearnFilePlugin()
    with pytest.raises(PluginError):
        plugin.convert(
            Path("model.joblib"),
            Path("out.onnx"),
            options={"n_features": 4, "metadata": "nope"},
        )


def test_rejects_non_path_parity_input() -> None:
    plugin = SklearnFilePlugin()
    with pytest.raises(PluginError):
        plugin.convert(
            Path("model.joblib"),
            Path("out.onnx"),
            options={"n_features": 4, "parity_input_path": "bad"},
        )


def test_rejects_bad_opset_type() -> None:
    plugin = SklearnFilePlugin()
    with pytest.raises(PluginError):
        plugin.convert(
            Path("model.joblib"),
            Path("out.onnx"),
            options={"n_features": 4, "opset_version": "14"},
        )


def test_calls_adapters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loader = DummyLoader()
    converter = DummyConverter(out=tmp_path / "out.onnx")
    parity = DummyParity()
    post = DummyPost()

    monkeypatch.setattr(builtins, "SklearnModelLoader", lambda: loader)
    monkeypatch.setattr(builtins, "SklearnModelConverter", lambda: converter)
    monkeypatch.setattr(builtins, "SklearnParityChecker", lambda: parity)
    monkeypatch.setattr(builtins, "OnnxPostProcessorImpl", lambda: post)

    plugin = SklearnFilePlugin()
    out = plugin.convert(
        model_path=tmp_path / "model.skops",
        output_path=tmp_path / "out.onnx",
        options={
            "n_features": 8,
            "allow_unsafe": True,
            "metadata": {"owner": "test"},
            "parity_input_path": tmp_path / "batch.npy",
            "parity_atol": 1e-5,
            "parity_rtol": 1e-4,
            "opset_version": 14,
            "optimize": True,
        },
    )

    assert out == tmp_path / "out.onnx"
    assert loader.calls == [(tmp_path / "model.skops", True)]
    assert converter.calls
    assert parity.calls
    assert post.calls
