#!/usr/bin/env python3
"""Examples for converting scikit-learn models/pipelines to ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from numpy import abs as np_abs
from numpy import allclose as np_allclose
from numpy import array as np_array
from numpy import array_equal as np_array_equal
from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from numpy import max as np_max
from numpy import ndarray as np_ndarray
from onnx import checker as onnx_checker
from onnx import load as onnx_load
from onnxruntime import InferenceSession as ort_InferenceSession
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from onnx_converter import convert_sklearn_to_onnx


def _to_prob_matrix(raw: Any, classes: np_ndarray) -> np_ndarray:
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return np_array(
            [[row[int(cls)] for cls in classes] for row in raw], dtype=np_float32
        )
    return np_asarray(raw, dtype=np_float32)


def _assert_classifier_parity(model: Any, onnx_path: Path, batch: np_ndarray) -> None:
    sklearn_pred = np_asarray(model.predict(batch))
    sklearn_proba = np_asarray(model.predict_proba(batch), dtype=np_float32)

    session = ort_InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": batch.astype(np_float32)})
    onnx_pred = np_asarray(outputs[0])
    onnx_proba = _to_prob_matrix(outputs[1], np_asarray(model.classes_))

    if onnx_pred.shape != sklearn_pred.shape or not np_array_equal(
        onnx_pred, sklearn_pred
    ):
        raise SystemExit("FAIL: predicted labels mismatch between sklearn and ONNX.")
    if sklearn_proba.shape != onnx_proba.shape:
        raise SystemExit("FAIL: probability tensor shape mismatch.")

    max_abs_diff = float(np_max(np_abs(sklearn_proba - onnx_proba)))
    print(f"Max |proba diff|: {max_abs_diff:.8f}")
    if not np_allclose(sklearn_proba, onnx_proba, atol=1e-5, rtol=1e-4):
        raise SystemExit(
            "FAIL: probability mismatch "
            f"(max_abs_diff={max_abs_diff:.8f}, atol=1e-5, rtol=1e-4)."
        )


def example_simple_classifier() -> None:
    """Convert RandomForest and enforce parity checks."""
    print("\n" + "=" * 60)
    print("Example 1: Random Forest Classifier")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        min_samples_leaf=1,
        max_features="sqrt",
    )
    model.fit(X, y)

    output_path = Path("outputs/rf_classifier.onnx")
    output_path.parent.mkdir(exist_ok=True)
    convert_sklearn_to_onnx(
        model=model,
        output_path=str(output_path),
        initial_types=[("input", FloatTensorType([None, 4]))],
    )

    if not output_path.exists():
        raise SystemExit("FAIL: rf_classifier.onnx was not created.")
    onnx_checker.check_model(onnx_load(str(output_path)))

    _assert_classifier_parity(model, output_path, X[:16].astype(np_float32))
    print(f"PASS: {output_path}")


def example_pipeline() -> None:
    """Convert preprocessing+classifier pipeline and enforce parity checks."""
    print("\n" + "=" * 60)
    print("Example 2: Pipeline (Scaler + Random Forest)")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    output_path = Path("outputs/pipeline.onnx")
    output_path.parent.mkdir(exist_ok=True)
    cache_dir = output_path.parent / "pipeline_cache"
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=10,
                    random_state=42,
                    min_samples_leaf=1,
                    max_features="sqrt",
                ),
            ),
        ],
        memory=str(cache_dir),
    )
    pipeline.fit(X, y)

    convert_sklearn_to_onnx(
        model=pipeline,
        output_path=str(output_path),
        initial_types=[("input", FloatTensorType([None, 4]))],
    )

    if not output_path.exists():
        raise SystemExit("FAIL: pipeline.onnx was not created.")
    onnx_checker.check_model(onnx_load(str(output_path)))

    _assert_classifier_parity(pipeline, output_path, X[:16].astype(np_float32))
    print(f"PASS: {output_path}")


def main() -> None:
    """Run all sklearn examples with strict pass/fail criteria."""
    example_simple_classifier()
    example_pipeline()
    print("PASS: sklearn examples complete.")


if __name__ == "__main__":
    main()
