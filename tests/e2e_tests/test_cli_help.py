"""End-to-end smoke test for CLI help output."""

from __future__ import annotations

from subprocess import run as subprocess_run

from onnx_converter import __version__ as onnx_converter___version__


def test_package_import_smoke() -> None:
    """Ensure package can be imported in the test process."""
    assert onnx_converter___version__


def test_cli_help_smoke() -> None:
    """Ensure the installed CLI entrypoint responds to --help."""
    result = subprocess_run(
        ["convert-to-onnx", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Convert ML models" in result.stdout


def test_cli_sklearn_missing_model_path_fails_cleanly() -> None:
    """Ensure CLI returns a user-facing validation error for missing model path."""
    result = subprocess_run(
        [
            "convert-to-onnx",
            "sklearn",
            "/tmp/definitely-missing-model.joblib",
            "/tmp/out.onnx",
            "--n-features",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "does not exist" in result.stderr.lower()
