"""Parity checks between framework outputs and ONNX Runtime outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from numpy import abs as np_abs
from numpy import allclose as np_allclose
from numpy import array as np_array
from numpy import array_equal as np_array_equal
from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from numpy import int64 as np_int64
from numpy import load as np_load
from numpy import loadtxt as np_loadtxt
from numpy import max as np_max
from numpy.typing import NDArray as npt_NDArray

from onnx_converter.errors import ParityError

FloatArray = npt_NDArray[np_float32]
LabelArray = npt_NDArray[np_int64]


class _PredictorProtocol(Protocol):
    """Protocol for models exposing ``predict``."""

    def predict(self, features: FloatArray) -> LabelArray:
        """Predict labels for features."""


class _ProbaPredictorProtocol(Protocol):
    """Protocol for models exposing ``predict_proba`` and ``classes_``."""

    classes_: npt_NDArray[np_int64]

    def predict_proba(self, features: FloatArray) -> FloatArray:
        """Predict class probabilities for features."""


def load_parity_input(input_path: Path) -> FloatArray:
    """Load parity input data from .npy/.npz/.csv/.txt."""
    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        data = np_load(str(input_path))
    elif suffix == ".npz":
        archive = np_load(str(input_path))
        if not archive.files:
            raise ParityError("Parity .npz file is empty.")
        data = archive[archive.files[0]]
    elif suffix in {".csv", ".txt"}:
        data = np_loadtxt(str(input_path), delimiter=",", dtype=np_float32)
    else:
        raise ParityError(
            "Unsupported parity input format. Use .npy, .npz, .csv, or .txt."
        )

    arr = np_asarray(data, dtype=np_float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim < 2:
        raise ParityError("Parity input must have at least 2 dimensions.")
    return arr


def _run_onnx_first_output(onnx_path: Path, batch: FloatArray) -> FloatArray:
    try:
        from onnxruntime import InferenceSession as ort_InferenceSession
    except Exception as exc:
        raise ParityError("Parity check requires onnxruntime to be installed.") from exc

    session = ort_InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: batch.astype(np_float32)})[0]
    return np_asarray(result, dtype=np_float32)


def check_tensor_parity(
    expected: FloatArray,
    onnx_path: Path,
    parity_input: FloatArray,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    """Check allclose parity for tensor outputs."""
    actual = _run_onnx_first_output(onnx_path, parity_input)
    expected = np_asarray(expected, dtype=np_float32)
    if expected.shape != actual.shape:
        raise ParityError(
            f"{label} parity failed: shape mismatch "
            f"(expected {expected.shape}, got {actual.shape})."
        )
    if not np_allclose(expected, actual, atol=atol, rtol=rtol):
        max_abs = float(np_max(np_abs(expected - actual)))
        raise ParityError(
            f"{label} parity failed: outputs differ "
            f"(max_abs_diff={max_abs:.8g}, atol={atol}, rtol={rtol})."
        )


def _probabilities_to_matrix(
    raw_probs: FloatArray | list[Mapping[int, float]], classes: npt_NDArray[np_int64]
) -> FloatArray:
    """Normalize various ONNX classifier probability encodings."""
    if isinstance(raw_probs, list) and raw_probs and isinstance(raw_probs[0], dict):
        return np_array(
            [[row[int(cls)] for cls in classes] for row in raw_probs], dtype=np_float32
        )
    return np_asarray(raw_probs, dtype=np_float32)


def check_sklearn_parity(
    model: _PredictorProtocol,
    onnx_path: Path,
    parity_input: FloatArray,
    atol: float,
    rtol: float,
) -> None:
    """Check label/probability parity for sklearn classifiers."""
    try:
        from onnxruntime import InferenceSession as ort_InferenceSession
    except Exception as exc:
        raise ParityError(
            "Sklearn parity check requires onnxruntime to be installed."
        ) from exc

    predictor = model
    sklearn_pred = np_asarray(predictor.predict(parity_input))
    proba_predictor = cast(_ProbaPredictorProtocol, model)
    sklearn_proba = (
        np_asarray(proba_predictor.predict_proba(parity_input), dtype=np_float32)
        if hasattr(model, "predict_proba")
        else None
    )

    session = ort_InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {input_name: parity_input.astype(np_float32)})

    onnx_pred = np_asarray(outputs[0])
    if onnx_pred.shape != sklearn_pred.shape or not np_array_equal(
        onnx_pred, sklearn_pred
    ):
        raise ParityError("Sklearn parity failed: predicted labels differ.")

    if sklearn_proba is not None and len(outputs) > 1:
        classes = np_asarray(proba_predictor.classes_)
        onnx_proba = _probabilities_to_matrix(outputs[1], classes)
        if sklearn_proba.shape != onnx_proba.shape:
            raise ParityError(
                "Sklearn parity failed: probability output shape differs."
            )
        if not np_allclose(sklearn_proba, onnx_proba, atol=atol, rtol=rtol):
            max_abs = float(np_max(np_abs(sklearn_proba - onnx_proba)))
            raise ParityError(
                "Sklearn parity failed: probabilities differ "
                f"(max_abs_diff={max_abs:.8g}, atol={atol}, rtol={rtol})."
            )
