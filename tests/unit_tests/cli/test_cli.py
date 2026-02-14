from __future__ import annotations

from typer.testing import CliRunner

from onnx_converter.cli import cli as cli_module

runner = CliRunner()


def test_help_shows_commands() -> None:
    result = runner.invoke(cli_module.app, ["--help"])
    assert result.exit_code == 0
    assert "pytorch" in result.output
    assert "tensorflow" in result.output
    assert "sklearn" in result.output


def test_pytorch_missing_deps(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    monkeypatch.setattr(
        cli_module,
        "_is_importable",
        lambda name: False if name == "torch" else True,
    )

    result = runner.invoke(
        cli_module.app,
        [
            "pytorch",
            str(model_path),
            str(output_path),
            "--input-shape",
            "1",
            "--input-shape",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert "Missing optional dependencies" in result.output


def test_pytorch_invokes_api(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    monkeypatch.setattr(cli_module, "_is_importable", lambda name: True)

    called = {}

    def fake_convert(
        *,
        model_path,
        output_path,
        input_shape,
        opset_version,
        allow_unsafe,
        **kwargs,
    ):
        called["model_path"] = model_path
        called["output_path"] = output_path
        called["input_shape"] = input_shape
        called["opset_version"] = opset_version
        called["allow_unsafe"] = allow_unsafe
        called["kwargs"] = kwargs
        return output_path

    import onnx_converter.api as api_module

    monkeypatch.setattr(api_module, "convert_torch_file_to_onnx", fake_convert)

    result = runner.invoke(
        cli_module.app,
        [
            "pytorch",
            str(model_path),
            str(output_path),
            "--input-shape",
            "1",
            "--input-shape",
            "3",
            "--input-shape",
            "224",
            "--input-shape",
            "224",
        ],
    )

    assert result.exit_code == 0
    assert "Saved:" in result.output
    assert called["model_path"] == model_path
    assert called["output_path"] == output_path
    assert called["input_shape"] == (1, 3, 224, 224)
    assert called["opset_version"] == 14
    assert called["allow_unsafe"] is False
    assert called["kwargs"] == {}
