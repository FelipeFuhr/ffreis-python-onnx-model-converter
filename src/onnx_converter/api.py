"""File-based conversion helpers used by the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from typing import Iterable

from pydantic import ValidationError

from onnx_converter.errors import ConversionError
from onnx_converter.schemas import SklearnFileConversionConfig
from onnx_converter.schemas import TensorflowFileConversionConfig
from onnx_converter.schemas import TorchFileConversionConfig


def _get_pytorch_converter() -> Callable[..., str]:
    from onnx_converter.converters.pytorch_converter import convert_pytorch_to_onnx

    return convert_pytorch_to_onnx


def _get_tensorflow_converter() -> Callable[..., str]:
    from onnx_converter.converters.tensorflow_converter import convert_tensorflow_to_onnx

    return convert_tensorflow_to_onnx


def _get_sklearn_converter() -> Callable[..., str]:
    from onnx_converter.converters.sklearn_converter import convert_sklearn_to_onnx

    return convert_sklearn_to_onnx


def convert_torch_file_to_onnx(
    model_path: Path,
    output_path: Path,
    input_shape: Iterable[int],
    opset_version: int = 14,
    allow_unsafe: bool = False,
) -> Path:
    """Convert a serialized PyTorch model file into ONNX.

    Parameters
    ----------
    model_path : Path
        Path to a TorchScript file or a pickled model/checkpoint.
    output_path : Path
        Destination path for the generated ONNX file.
    input_shape : Iterable[int]
        Input tensor shape used to build a dummy input for export.
    opset_version : int, default=14
        ONNX opset version used by the exporter.
    allow_unsafe : bool, default=False
        Whether to allow pickle-based loading via ``torch.load`` when
        TorchScript loading fails.

    Returns
    -------
    Path
        Path to the generated ONNX model.

    Raises
    ------
    ConversionError
        If dependencies are missing, model loading is unsafe, or conversion
        fails due to unsupported input artifacts.
    """
    try:
        config = TorchFileConversionConfig(
            model_path=model_path,
            output_path=output_path,
            input_shape=tuple(input_shape),
            opset_version=opset_version,
            allow_unsafe=allow_unsafe,
        )
    except ValidationError as exc:
        raise ConversionError(f"Invalid PyTorch conversion parameters: {exc}") from exc

    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised in CLI-level checks
        raise ConversionError("PyTorch is required for this conversion.") from exc

    model_path = config.model_path
    output_path = config.output_path
    opset_version = config.opset_version
    allow_unsafe = config.allow_unsafe
    input_shape = config.input_shape

    try:
        model = torch.jit.load(str(model_path))
    except Exception:
        if not allow_unsafe:
            raise ConversionError(
                "TorchScript loading failed. Re-run with --allow-unsafe to use torch.load."
            )
        model = torch.load(str(model_path), map_location="cpu")

    if isinstance(model, dict) and (
        "model_state_dict" in model or "state_dict" in model
    ):
        raise ConversionError(
            "Model appears to be a checkpoint. Load the architecture and export from code."
        )

    convert_pytorch_to_onnx = _get_pytorch_converter()

    out = convert_pytorch_to_onnx(
        model=model,
        output_path=str(output_path),
        input_shape=input_shape,
        opset_version=opset_version,
    )
    return Path(out)


def convert_tf_path_to_onnx(
    model_path: Path,
    output_path: Path,
    opset_version: int = 14,
) -> Path:
    """Convert a TensorFlow model path to ONNX.

    Parameters
    ----------
    model_path : Path
        Path to a SavedModel directory or ``.h5`` file.
    output_path : Path
        Destination path for the generated ONNX file.
    opset_version : int, default=14
        ONNX opset version used by the exporter.

    Returns
    -------
    Path
        Path to the generated ONNX model.

    Raises
    ------
    ConversionError
        If TensorFlow is unavailable or conversion fails.
    """
    try:
        config = TensorflowFileConversionConfig(
            model_path=model_path,
            output_path=output_path,
            opset_version=opset_version,
        )
    except ValidationError as exc:
        raise ConversionError(
            f"Invalid TensorFlow conversion parameters: {exc}"
        ) from exc

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - exercised in CLI-level checks
        raise ConversionError("TensorFlow is required for this conversion.") from exc

    model_path = config.model_path
    output_path = config.output_path
    opset_version = config.opset_version

    if model_path.is_dir():
        model = str(model_path)
    else:
        model = tf.keras.models.load_model(str(model_path))

    convert_tensorflow_to_onnx = _get_tensorflow_converter()

    out = convert_tensorflow_to_onnx(
        model=model,
        output_path=str(output_path),
        opset_version=opset_version,
    )
    return Path(out)


def convert_sklearn_file_to_onnx(
    model_path: Path,
    output_path: Path,
    n_features: int,
    allow_unsafe: bool = False,
) -> Path:
    """Convert a serialized scikit-learn model file to ONNX.

    Parameters
    ----------
    model_path : Path
        Path to a serialized scikit-learn model file (``.joblib``, ``.skops``,
        ``.pkl``, or ``.pickle``).
    output_path : Path
        Destination path for the generated ONNX file.
    n_features : int
        Number of expected float input features.
    allow_unsafe : bool, default=False
        Whether to allow loading pickle-based artifacts.

    Returns
    -------
    Path
        Path to the generated ONNX model.

    Raises
    ------
    ConversionError
        If the model extension is unsupported, unsafe loading is blocked,
        or conversion fails.
    """
    try:
        config = SklearnFileConversionConfig(
            model_path=model_path,
            output_path=output_path,
            n_features=n_features,
            allow_unsafe=allow_unsafe,
        )
    except ValidationError as exc:
        raise ConversionError(f"Invalid sklearn conversion parameters: {exc}") from exc

    model_path = config.model_path
    output_path = config.output_path
    n_features = config.n_features
    allow_unsafe = config.allow_unsafe
    suffix = model_path.suffix.lower()

    if suffix in {".pkl", ".pickle"} and not allow_unsafe:
        raise ConversionError(
            "Pickle loading is unsafe. Use .joblib/.skops or pass --allow-unsafe."
        )

    if suffix == ".skops":
        from skops.io import load as skops_load

        model = skops_load(str(model_path))
    elif suffix in {".joblib", ".jl"}:
        import joblib

        model = joblib.load(str(model_path))
    elif suffix in {".pkl", ".pickle"}:
        import pickle

        with model_path.open("rb") as handle:
            model = pickle.load(handle)
    else:
        raise ConversionError(
            "Unsupported model file extension. Use .joblib, .skops, or .pkl/.pickle."
        )

    from skl2onnx.common.data_types import FloatTensorType

    convert_sklearn_to_onnx = _get_sklearn_converter()

    initial_types = [("input", FloatTensorType([None, n_features]))]

    out = convert_sklearn_to_onnx(
        model=model,
        output_path=str(output_path),
        initial_types=initial_types,
    )
    return Path(out)
