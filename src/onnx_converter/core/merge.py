"""ONNX model merging: combine a preprocessing graph with a model graph.

The primary use-case is producing a single self-contained inference artifact
from a preprocessing graph (e.g. z-score normalisation, feature extraction)
and a downstream model graph (e.g. a neural-network policy or classifier).

``onnx.compose.merge_models`` is available in ``onnx>=1.13`` and is already
satisfied by this package's ``onnx>=1.17.0`` dependency floor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnx.compose
from numpy.typing import NDArray

from onnx_converter.errors import ConversionError


class MergeError(ConversionError):
    """Raised when ONNX graph merging or parity verification fails."""

    exit_code = 8


def merge_onnx_models(
    preprocessing_path: Path,
    model_path: Path,
    output_path: Path,
    *,
    prefix: str = "pre_",
) -> None:
    """Merge a preprocessing ONNX graph with a model ONNX graph.

    The preprocessing graph's outputs become the model graph's inputs.
    A name-collision prefix is added to all nodes/tensors in the
    preprocessing graph so that shared names (e.g. ``output``) do not clash
    with the model graph.

    Parameters
    ----------
    preprocessing_path : Path
        Path to the ONNX file that contains the preprocessing graph.
    model_path : Path
        Path to the ONNX file that contains the model graph.
    output_path : Path
        Destination path for the merged single-graph ONNX artifact.
    prefix : str, default ``"pre_"``
        String prepended to every node/tensor name in the preprocessing graph
        to avoid name collisions with the model graph.

    Raises
    ------
    MergeError
        If either source file cannot be loaded, if
        ``onnx.compose.merge_models`` fails, or if the merged graph cannot
        be validated.

    Notes
    -----
    - Both source graphs must share a compatible opset version.  If they
      differ, the higher opset is preserved in the merged output.
    - The merge is purely graph-structural: no ORT session is started.
      Use :func:`verify_merge_parity` to confirm numerical equivalence.
    """
    try:
        pre_model = onnx.load(str(preprocessing_path))
    except Exception as exc:
        raise MergeError(
            f"Failed to load preprocessing model from {preprocessing_path}: {exc}"
        ) from exc

    try:
        main_model = onnx.load(str(model_path))
    except Exception as exc:
        raise MergeError(f"Failed to load model from {model_path}: {exc}") from exc

    try:
        merged = onnx.compose.merge_models(
            pre_model,
            main_model,
            io_map=[
                (out.name, inp.name)
                for out, inp in zip(
                    pre_model.graph.output,
                    main_model.graph.input,
                    strict=False,
                )
            ],
            prefix1=prefix,
        )
    except Exception as exc:
        raise MergeError(f"onnx.compose.merge_models failed: {exc}") from exc

    try:
        onnx.checker.check_model(merged)
    except Exception as exc:
        raise MergeError(f"Merged model failed ONNX checker validation: {exc}") from exc

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(merged, str(output_path))
    except Exception as exc:
        raise MergeError(
            f"Failed to save merged model to {output_path}: {exc}"
        ) from exc


def verify_merge_parity(
    preprocessing_path: Path,
    model_path: Path,
    merged_path: Path,
    sample_input: NDArray[np.float32],
    *,
    atol: float = 1e-4,
) -> None:
    """Verify that chained sessions produce the same output as the merged graph.

    Runs three ORT inference sessions:

    1. ``preprocessing_path`` → intermediate output
    2. ``model_path`` with the intermediate output as input
    3. ``merged_path`` with the raw input

    Asserts that the outputs of sessions (2) and (3) are within ``atol``.

    Parameters
    ----------
    preprocessing_path : Path
        Path to the preprocessing ONNX graph.
    model_path : Path
        Path to the model ONNX graph.
    merged_path : Path
        Path to the merged ONNX graph produced by :func:`merge_onnx_models`.
    sample_input : numpy.ndarray
        A representative input batch (shape and dtype must match the
        preprocessing graph's first input).
    atol : float, default ``1e-4``
        Absolute tolerance for the element-wise output comparison.

    Raises
    ------
    MergeError
        If ``onnxruntime`` is not installed, if any session raises, or if
        the parity check fails.
    """
    try:
        from onnxruntime import InferenceSession as _OrtSession
    except ImportError as exc:
        raise MergeError(
            "verify_merge_parity requires onnxruntime. "
            "Install with: uv pip install -e '.[runtime]'"
        ) from exc

    providers = ["CPUExecutionProvider"]

    try:
        pre_sess = _OrtSession(str(preprocessing_path), providers=providers)
        pre_in_name = pre_sess.get_inputs()[0].name
        pre_out_name = pre_sess.get_outputs()[0].name
        intermediate = pre_sess.run(
            [pre_out_name], {pre_in_name: sample_input.astype(np.float32)}
        )[0]

        model_sess = _OrtSession(str(model_path), providers=providers)
        model_in_name = model_sess.get_inputs()[0].name
        model_out_name = model_sess.get_outputs()[0].name
        chained_output = model_sess.run(
            [model_out_name], {model_in_name: intermediate.astype(np.float32)}
        )[0]

        merged_sess = _OrtSession(str(merged_path), providers=providers)
        merged_in_name = merged_sess.get_inputs()[0].name
        merged_out_name = merged_sess.get_outputs()[0].name
        merged_output = merged_sess.run(
            [merged_out_name], {merged_in_name: sample_input.astype(np.float32)}
        )[0]
    except MergeError:
        raise
    except Exception as exc:
        raise MergeError(f"ORT inference failed during parity check: {exc}") from exc

    chained = np.asarray(chained_output, dtype=np.float32)
    merged = np.asarray(merged_output, dtype=np.float32)

    if chained.shape != merged.shape:
        raise MergeError(
            f"Parity shape mismatch: chained={chained.shape}, merged={merged.shape}"
        )

    if not np.allclose(chained, merged, atol=atol, rtol=0.0):
        max_diff = float(np.max(np.abs(chained - merged)))
        raise MergeError(
            f"Merge parity failed: max absolute diff={max_diff:.6g} > atol={atol}"
        )
