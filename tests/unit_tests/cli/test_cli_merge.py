"""Unit tests for the ``merge`` CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto
from onnx import helper as onnx_helper
from pytest import MonkeyPatch as pytest_MonkeyPatch
from typer.testing import CliRunner

from onnx_converter.cli import cli as cli_module
from onnx_converter.core.merge import MergeError

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toy_graphs(tmp_path: Path) -> tuple[Path, Path]:
    """Write minimal ONNX graphs: Mul-pre + Add-model."""
    n = 4

    def _model(
        in_name: str, op: str, initializer_name: str, out_name: str, graph_name: str
    ) -> onnx.ModelProto:
        x = onnx_helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [None, n])
        y = onnx_helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [None, n])
        init = onnx_helper.make_tensor(
            initializer_name, TensorProto.FLOAT, [1, n], [2.0] * n
        )
        node = onnx_helper.make_node(op, [in_name, initializer_name], [out_name])
        graph = onnx_helper.make_graph([node], graph_name, [x], [y], [init])
        opset = [onnx_helper.make_opsetid("", 14)]
        m = onnx_helper.make_model(graph, opset_imports=opset)
        m.ir_version = 8
        return m

    pre_path = tmp_path / "pre.onnx"
    model_path = tmp_path / "model.onnx"
    onnx.save(_model("input", "Mul", "scale", "output", "pre"), str(pre_path))
    onnx.save(
        _model("model_input", "Add", "bias", "model_output", "model"),
        str(model_path),
    )
    return pre_path, model_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_cmd_calls_merge_onnx_models(
    tmp_path: Path, monkeypatch: pytest_MonkeyPatch
) -> None:
    """CLI merge command invokes merge_onnx_models with expected arguments."""
    pre_path, model_path = _make_toy_graphs(tmp_path)
    output_path = tmp_path / "merged.onnx"

    called: dict[str, Any] = {}

    def fake_merge(
        preprocessing_path: Path,
        model_path: Path,  # noqa: A002 — shadows outer local intentionally
        output_path: Path,  # noqa: A002
        *,
        prefix: str = "pre_",
    ) -> None:
        called["preprocessing_path"] = preprocessing_path
        called["model_path"] = model_path
        called["output_path"] = output_path
        called["prefix"] = prefix

    import onnx_converter.core.merge as merge_module

    monkeypatch.setattr(merge_module, "merge_onnx_models", fake_merge)

    result = runner.invoke(
        cli_module.app,
        [
            "merge",
            "--preprocessing",
            str(pre_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Merged:" in result.output
    assert called["preprocessing_path"] == pre_path
    assert called["model_path"] == model_path
    assert called["output_path"] == output_path
    assert called["prefix"] == "pre_"


@pytest.mark.unit
def test_merge_cmd_custom_prefix(
    tmp_path: Path, monkeypatch: pytest_MonkeyPatch
) -> None:
    """--prefix flag is forwarded to merge_onnx_models."""
    pre_path, model_path = _make_toy_graphs(tmp_path)
    output_path = tmp_path / "merged.onnx"
    received: dict[str, Any] = {}

    def fake_merge(*, prefix: str = "pre_", **_: object) -> None:
        received["prefix"] = prefix

    import onnx_converter.core.merge as merge_module

    monkeypatch.setattr(merge_module, "merge_onnx_models", fake_merge)

    result = runner.invoke(
        cli_module.app,
        [
            "merge",
            "--preprocessing",
            str(pre_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--prefix",
            "zscore_",
        ],
    )

    assert result.exit_code == 0, result.output
    assert received["prefix"] == "zscore_"


@pytest.mark.unit
def test_merge_cmd_verify_requires_verify_shape(
    tmp_path: Path, monkeypatch: pytest_MonkeyPatch
) -> None:
    """--verify without --verify-shape produces a parameter error."""
    pre_path, model_path = _make_toy_graphs(tmp_path)
    output_path = tmp_path / "merged.onnx"

    monkeypatch.setattr(
        cli_module,
        "_is_importable",
        lambda _name: True,
    )

    import onnx_converter.core.merge as merge_module

    monkeypatch.setattr(merge_module, "merge_onnx_models", lambda **_: None)

    result = runner.invoke(
        cli_module.app,
        [
            "merge",
            "--preprocessing",
            str(pre_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--verify",
        ],
    )

    assert result.exit_code != 0


@pytest.mark.unit
def test_merge_cmd_surfaces_merge_error(
    tmp_path: Path, monkeypatch: pytest_MonkeyPatch
) -> None:
    """A MergeError from the core layer produces a non-zero exit and error output."""
    pre_path, model_path = _make_toy_graphs(tmp_path)
    output_path = tmp_path / "merged.onnx"

    def fake_merge(**_: object) -> None:
        raise MergeError("incompatible io shapes")

    import onnx_converter.core.merge as merge_module

    monkeypatch.setattr(merge_module, "merge_onnx_models", fake_merge)

    result = runner.invoke(
        cli_module.app,
        [
            "merge",
            "--preprocessing",
            str(pre_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code != 0


@pytest.mark.unit
def test_merge_cmd_surfaces_generic_error(
    tmp_path: Path, monkeypatch: pytest_MonkeyPatch
) -> None:
    """An unexpected exception from the core layer produces a non-zero exit."""
    pre_path, model_path = _make_toy_graphs(tmp_path)
    output_path = tmp_path / "merged.onnx"

    def fake_merge(**_: object) -> None:
        raise ValueError("unexpected core failure")

    import onnx_converter.core.merge as merge_module

    monkeypatch.setattr(merge_module, "merge_onnx_models", fake_merge)

    result = runner.invoke(
        cli_module.app,
        [
            "merge",
            "--preprocessing",
            str(pre_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code != 0


@pytest.mark.unit
def test_merge_cmd_help_output(tmp_path: Path) -> None:
    """Verify that ``merge --help`` lists expected flags."""
    # scan-fix(ci:terminal-width): force wide terminal so Rich does not truncate
    # long option names (--preprocessing) in narrow act container environments
    result = runner.invoke(cli_module.app, ["merge", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--preprocessing" in result.output
    assert "--model" in result.output
    assert "--output" in result.output
    assert "--verify" in result.output
