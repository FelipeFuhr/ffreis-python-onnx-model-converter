"""End-to-end tests for the public converter CLI."""

from __future__ import annotations

from pathlib import Path
from subprocess import run as subprocess_run

from onnx import checker as onnx_checker
from onnx import load as onnx_load
from pytest import importorskip as pytest_importorskip


def test_cli_sklearn_roundtrip(tmp_path: Path) -> None:
    """Run a full sklearn -> ONNX conversion through the public CLI."""
    pytest_importorskip("joblib")
    pytest_importorskip("sklearn")
    pytest_importorskip("skl2onnx")
    pytest_importorskip("onnxruntime")
    from joblib import dump as joblib_dump
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    features, labels = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200, random_state=42).fit(features, labels)

    model_path = tmp_path / "model.joblib"
    onnx_path = tmp_path / "model.onnx"
    joblib_dump(model, model_path)

    cmd = [
        "convert-to-onnx",
        "sklearn",
        str(model_path),
        str(onnx_path),
        "--n-features",
        str(features.shape[1]),
        "--allow-unsafe",
    ]
    result = subprocess_run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert onnx_path.exists()
    onnx_checker.check_model(onnx_load(str(onnx_path)))
