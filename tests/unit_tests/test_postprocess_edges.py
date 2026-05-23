# ruff: noqa: D101, D102, E501
# flake8: noqa: E501
"""Additional postprocess tests: quantize body, metadata key normalization."""

from __future__ import annotations

from pathlib import Path
from sys import modules as sys_modules
from types import SimpleNamespace

from onnx import TensorProto, helper
from onnx import load as onnx_load
from onnx import save as onnx_save
from pytest import MonkeyPatch as pytest_MonkeyPatch

from onnx_converter.postprocess import add_onnx_metadata, quantize_onnx_dynamic


def _write_minimal_onnx(path: Path) -> None:
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
    )
    onnx_save(helper.make_model(graph, producer_name="t"), str(path))


class TestQuantizeOnnxDynamicBody:
    """Exercise the quantize_dynamic happy path using a fake onnxruntime."""

    def test_runs_with_fake_quantize_module(
        self, monkeypatch: pytest_MonkeyPatch, tmp_path: Path
    ) -> None:
        original = tmp_path / "model.onnx"
        _write_minimal_onnx(original)
        original.write_bytes(b"original-content")  # mark for swap verification

        def fake_quantize_dynamic(
            *, model_input: str, model_output: str, weight_type: object
        ) -> None:
            # Real onnxruntime would write a quantized ONNX file; we write a
            # marker so the test can verify the swap that happens at line 78.
            Path(model_output).write_bytes(b"quantized-content")

        class _QuantType:
            QInt8 = "qint8"

        fake_module = SimpleNamespace(
            QuantType=_QuantType,
            quantize_dynamic=fake_quantize_dynamic,
        )
        monkeypatch.setitem(sys_modules, "onnxruntime.quantization", fake_module)

        quantize_onnx_dynamic(original)

        # After quantize, the original file should hold the quantized content.
        assert original.read_bytes() == b"quantized-content"
        # The intermediate .quantized.onnx file should have been moved.
        assert not original.with_suffix(".quantized.onnx").exists()


class TestMetadataMerge:
    def test_keys_case_sensitive_no_collision(self, tmp_path: Path) -> None:
        # Verify case difference produces two distinct entries (no collapse).
        path = tmp_path / "m.onnx"
        _write_minimal_onnx(path)
        add_onnx_metadata(path, {"Version": "1", "version": "2"})
        reloaded = onnx_load(str(path))
        values = {entry.key: entry.value for entry in reloaded.metadata_props}
        assert values == {"Version": "1", "version": "2"}

    def test_keys_sorted_after_merge(self, tmp_path: Path) -> None:
        path = tmp_path / "m.onnx"
        _write_minimal_onnx(path)
        add_onnx_metadata(path, {"z": "1", "a": "2", "m": "3"})
        reloaded = onnx_load(str(path))
        keys = [entry.key for entry in reloaded.metadata_props]
        assert keys == sorted(keys)

    def test_non_string_keys_coerced(self, tmp_path: Path) -> None:
        path = tmp_path / "m.onnx"
        _write_minimal_onnx(path)
        # add_onnx_metadata internally calls str() on keys.
        add_onnx_metadata(path, {1: "one", 2: "two"})  # type: ignore[dict-item]
        reloaded = onnx_load(str(path))
        values = {entry.key: entry.value for entry in reloaded.metadata_props}
        assert values == {"1": "one", "2": "two"}

    def test_empty_metadata_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "m.onnx"
        _write_minimal_onnx(path)
        add_onnx_metadata(path, {"keep": "me"})
        # Adding empty does not remove or change existing entries.
        add_onnx_metadata(path, {})
        reloaded = onnx_load(str(path))
        values = {entry.key: entry.value for entry in reloaded.metadata_props}
        assert values == {"keep": "me"}
