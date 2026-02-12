#!/usr/bin/env python3
"""Train a pipeline with a custom transformer for CLI conversion."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from examples.custom_sklearn_transformer import MultiplyByConstant


def main() -> None:
    """Train and persist a custom scikit-learn pipeline.

    The generated artifact can be converted with the CLI by importing the
    custom converter module in this examples package.
    """
    data = load_iris()
    X, y = data.data, data.target

    pipeline = Pipeline(
        [
            ("scale", MultiplyByConstant(factor=1.5)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    pipeline.fit(X, y)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "custom_sklearn.joblib"
    joblib.dump(pipeline, model_path)

    print(f"Saved sklearn model to: {model_path}")
    print("\nConvert with CLI:")
    print(
        "convert-to-onnx sklearn outputs/custom_sklearn.joblib outputs/custom_sklearn.onnx "
        "--n-features 4 --custom-converter-module examples/custom_sklearn_transformer.py"
    )


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
