#!/usr/bin/env python3
"""Train a custom sklearn pipeline, convert via CLI, and validate parity."""

from __future__ import annotations

from os import environ as os_environ
from pathlib import Path
from subprocess import run as subprocess_run
from sys import path as sys_path

from joblib import dump as joblib_dump
from numpy import abs as np_abs
from numpy import allclose as np_allclose
from numpy import array as np_array
from numpy import array_equal as np_array_equal
from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from numpy import max as np_max
from onnx import checker as onnx_checker
from onnx import load as onnx_load
from onnxruntime import InferenceSession as ort_InferenceSession
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Run custom-converter CLI flow with explicit pass/fail checks."""
    if str(PROJECT_ROOT) not in sys_path:
        sys_path.insert(0, str(PROJECT_ROOT))
    from examples.custom_sklearn_transformer import MultiplyByConstant

    X, y = load_iris(return_X_y=True)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    pipeline = Pipeline(
        [
            ("scale", MultiplyByConstant(factor=1.5)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(X, y)

    model_path = output_dir / "custom_sklearn.joblib"
    onnx_path = output_dir / "custom_sklearn.onnx"
    joblib_dump(pipeline, model_path)

    command = [
        "convert-to-onnx",
        "sklearn",
        str(model_path),
        str(onnx_path),
        "--n-features",
        str(X.shape[1]),
        "--custom-converter-module",
        "examples.custom_sklearn_transformer",
        "--allow-unsafe",
    ]
    env = os_environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}:{existing_pythonpath}"
        if existing_pythonpath
        else str(PROJECT_ROOT)
    )
    result = subprocess_run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise SystemExit(f"FAIL: CLI conversion failed.\n{result.stderr}")

    if not onnx_path.exists():
        raise SystemExit("FAIL: custom_sklearn.onnx was not created.")
    onnx_checker.check_model(onnx_load(str(onnx_path)))

    batch = X[:16].astype(np_float32)
    sk_pred = np_asarray(pipeline.predict(batch))
    sk_proba = np_asarray(pipeline.predict_proba(batch), dtype=np_float32)

    session = ort_InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": batch})
    onnx_pred = np_asarray(outputs[0])

    if isinstance(outputs[1], list) and outputs[1] and isinstance(outputs[1][0], dict):
        classes = np_asarray(pipeline.classes_)
        onnx_proba = np_array(
            [[row[int(cls)] for cls in classes] for row in outputs[1]], dtype=np_float32
        )
    else:
        onnx_proba = np_asarray(outputs[1], dtype=np_float32)

    if onnx_pred.shape != sk_pred.shape or not np_array_equal(onnx_pred, sk_pred):
        raise SystemExit("FAIL: predicted labels mismatch between sklearn and ONNX.")

    max_abs_diff = float(np_max(np_abs(sk_proba - onnx_proba)))
    print(f"Max |proba diff|: {max_abs_diff:.8f}")
    if not np_allclose(sk_proba, onnx_proba, atol=1e-5, rtol=1e-4):
        raise SystemExit(
            "FAIL: probability mismatch "
            f"(max_abs_diff={max_abs_diff:.8f}, atol=1e-5, rtol=1e-4)."
        )

    print(f"PASS: {onnx_path}")


if __name__ == "__main__":
    main()
