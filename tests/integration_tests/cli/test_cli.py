from __future__ import annotations

from typer.testing import CliRunner

from onnx_converter.cli import cli as cli_module

runner = CliRunner()


def test_cli_help_smoke() -> None:
    result = runner.invoke(cli_module.app, ["--help"])
    assert result.exit_code == 0
    assert "Convert ML models" in result.output
