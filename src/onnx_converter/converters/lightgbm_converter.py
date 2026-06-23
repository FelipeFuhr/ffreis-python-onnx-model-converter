"""LightGBM-to-ONNX conversion utilities using onnxmltools."""

from __future__ import annotations

from os import makedirs as os_makedirs
from os import path as os_path
from pathlib import Path

from pydantic import ValidationError

from onnx_converter.errors import ConversionError
from onnx_converter.schemas import SklearnConversionConfig
from onnx_converter.types import ModelArtifact, OptionValue, SklearnInitialTypeLike


def convert_lightgbm_to_onnx(
    model: ModelArtifact,
    output_path: str,
    n_features: int,
    initial_types: list[tuple[str, SklearnInitialTypeLike]] | None = None,
    target_opset: int | None = None,
    **kwargs: OptionValue,
) -> str:
    """Convert a LightGBM classifier or regressor to ONNX format.

    Uses ``onnxmltools.convert_lightgbm`` to produce the ONNX graph, then writes
    it to ``output_path``.

    Parameters
    ----------
    model : ModelArtifact
        Trained LightGBM ``LGBMClassifier`` or ``LGBMRegressor`` instance.
    output_path : str
        Destination path for the ONNX model file.
    n_features : int
        Number of input features; used to build the ``initial_types`` when
        ``initial_types`` is not provided.
    initial_types : list of (str, type), optional
        ``onnxmltools``-style input type declarations.  When omitted, a single
        ``FloatTensorType([None, n_features])`` entry is created automatically.
    target_opset : int, optional
        ONNX opset version override.  Defaults to ``15`` (stable LightGBM
        coverage; raise only when the consumer requires a higher opset).
    **kwargs
        Additional keyword arguments forwarded to ``onnxmltools.convert_lightgbm``.

    Returns
    -------
    str
        Path to the saved ONNX model.

    Raises
    ------
    ConversionError
        If ``onnxmltools`` or ``skl2onnx`` are not installed, or if the
        conversion fails.
    """
    if n_features <= 0:
        raise ConversionError(
            "n_features must be a positive integer for LightGBM conversion."
        )

    # scan-fix(runtime): onnxmltools requires its own FloatTensorType, not skl2onnx's;
    # skl2onnx.FloatTensorType raises RuntimeError in onnxmltools shape calculators.
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:
        raise ConversionError(
            "onnxmltools is required for LightGBM conversion. "
            "Install with: pip install onnxmltools"
        ) from exc

    if initial_types is None:
        initial_types = [("input", FloatTensorType([None, n_features]))]

    effective_opset = target_opset if target_opset is not None else 15

    try:
        config = SklearnConversionConfig(
            output_path=Path(output_path),
            initial_types=initial_types,
            target_opset=effective_opset,
        )
    except ValidationError as exc:
        raise ConversionError(f"Invalid LightGBM export options: {exc}") from exc

    output_path_str = str(config.output_path)
    os_makedirs(
        os_path.dirname(output_path_str) if os_path.dirname(output_path_str) else ".",
        exist_ok=True,
    )

    try:
        onx = onnxmltools.convert_lightgbm(
            model,
            initial_types=initial_types,
            target_opset=effective_opset,
            **kwargs,
        )
    except Exception as exc:
        raise ConversionError(f"LightGBM ONNX conversion failed: {exc}") from exc

    with open(output_path_str, "wb") as handle:
        handle.write(onx.SerializeToString())

    return output_path_str
