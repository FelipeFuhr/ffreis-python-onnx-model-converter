"""Unit tests for onnx_converter.core.merge.

Tests use two toy ONNX graphs whose composition is analytically verifiable:

- **Preprocessing graph** (``pre_model``): z-score-like normalisation
  implemented as ``Y = X * scale``, where ``scale = [2.0, 2.0, 2.0, 2.0]``.
- **Model graph** (``model_model``): a linear shift implemented as
  ``Z = Y + bias``, where ``bias = [1.0, 1.0, 1.0, 1.0]``.

The expected merged output for a batch of ones is therefore:
``(1.0 * 2.0) + 1.0 = 3.0`` for each feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx import helper as onnx_helper

from onnx_converter.core.merge import MergeError
from onnx_converter.core.merge import merge_onnx_models
from onnx_converter.core.merge import verify_merge_parity

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_scale_model(
    in_name: str,
    out_name: str,
    scale_values: list[float] | None = None,
    n_features: int = 4,
) -> onnx.ModelProto:
    """Return a single-node Mul graph: ``output = input * scale``."""
    if scale_values is None:
        scale_values = [2.0] * n_features
    x = onnx_helper.make_tensor_value_info(
        in_name, TensorProto.FLOAT, [None, n_features]
    )
    y = onnx_helper.make_tensor_value_info(
        out_name, TensorProto.FLOAT, [None, n_features]
    )
    scale = onnx_helper.make_tensor(
        "scale", TensorProto.FLOAT, [1, n_features], scale_values
    )
    mul_node = onnx_helper.make_node("Mul", [in_name, "scale"], [out_name])
    graph = onnx_helper.make_graph([mul_node], "pre", [x], [y], [scale])
    opset = [onnx_helper.make_opsetid("", 14)]
    model = onnx_helper.make_model(graph, opset_imports=opset)
    model.ir_version = 8
    return model


def _make_add_model(
    in_name: str,
    out_name: str,
    bias_values: list[float] | None = None,
    n_features: int = 4,
) -> onnx.ModelProto:
    """Return a single-node Add graph: ``output = input + bias``."""
    if bias_values is None:
        bias_values = [1.0] * n_features
    y = onnx_helper.make_tensor_value_info(
        in_name, TensorProto.FLOAT, [None, n_features]
    )
    z = onnx_helper.make_tensor_value_info(
        out_name, TensorProto.FLOAT, [None, n_features]
    )
    bias = onnx_helper.make_tensor(
        "bias", TensorProto.FLOAT, [1, n_features], bias_values
    )
    add_node = onnx_helper.make_node("Add", [in_name, "bias"], [out_name])
    graph = onnx_helper.make_graph([add_node], "model", [y], [z], [bias])
    opset = [onnx_helper.make_opsetid("", 14)]
    model = onnx_helper.make_model(graph, opset_imports=opset)
    model.ir_version = 8
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def toy_graphs(tmp_path: Path) -> tuple[Path, Path]:
    """Write toy pre/model ONNX files and return their paths."""
    pre_path = tmp_path / "pre.onnx"
    model_path = tmp_path / "model.onnx"
    pre_model = _make_scale_model(in_name="input", out_name="output")
    main_model = _make_add_model(in_name="model_input", out_name="model_output")
    onnx.save(pre_model, str(pre_path))
    onnx.save(main_model, str(model_path))
    return pre_path, model_path


# ---------------------------------------------------------------------------
# merge_onnx_models — happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_produces_valid_onnx(
    toy_graphs: tuple[Path, Path], tmp_path: Path
) -> None:
    """Merged graph passes onnx.checker.check_model."""
    pre_path, model_path = toy_graphs
    merged_path = tmp_path / "merged.onnx"

    merge_onnx_models(pre_path, model_path, merged_path)

    assert merged_path.exists()
    merged = onnx.load(str(merged_path))
    onnx.checker.check_model(merged)  # raises if invalid


@pytest.mark.unit
def test_merge_output_path_is_created(
    toy_graphs: tuple[Path, Path], tmp_path: Path
) -> None:
    """Merged file is created at the requested output path."""
    pre_path, model_path = toy_graphs
    nested = tmp_path / "subdir" / "merged.onnx"

    merge_onnx_models(pre_path, model_path, nested)

    assert nested.exists()


@pytest.mark.unit
def test_merge_prefix_applied_to_preprocessing_graph(
    toy_graphs: tuple[Path, Path], tmp_path: Path
) -> None:
    """The preprocessing graph's input is renamed with the configured prefix."""
    pre_path, model_path = toy_graphs
    merged_path = tmp_path / "merged.onnx"

    merge_onnx_models(pre_path, model_path, merged_path, prefix="zscore_")

    merged = onnx.load(str(merged_path))
    input_names = [i.name for i in merged.graph.input]
    # The preprocessing graph's original input name was "input";
    # after prefixing it becomes "zscore_input".
    msg = f"Expected 'zscore_input' in merged inputs {input_names}"
    assert any("zscore_input" in name for name in input_names), msg


@pytest.mark.unit
def test_merge_parity_within_atol(
    toy_graphs: tuple[Path, Path], tmp_path: Path
) -> None:
    """Chained ORT sessions and the merged graph produce identical outputs.

    For the toy graphs::

        merged(X) == (X * 2.0) + 1.0

    A batch of random floats in [0, 1) is used.
    """
    pytest.importorskip("onnxruntime", reason="onnxruntime required for parity check")

    pre_path, model_path = toy_graphs
    merged_path = tmp_path / "merged.onnx"
    merge_onnx_models(pre_path, model_path, merged_path)

    rng = np.random.default_rng(42)
    sample = rng.random((8, 4)).astype(np.float32)

    # verify_merge_parity raises MergeError on failure; no assertion needed here.
    verify_merge_parity(pre_path, model_path, merged_path, sample_input=sample)


@pytest.mark.unit
def test_merged_output_values_match_analytical_expectation(
    toy_graphs: tuple[Path, Path], tmp_path: Path
) -> None:
    """Merged graph output equals the analytical result (X * 2 + 1)."""
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime required")

    pre_path, model_path = toy_graphs
    merged_path = tmp_path / "merged.onnx"
    merge_onnx_models(pre_path, model_path, merged_path)

    sample = np.ones((3, 4), dtype=np.float32)
    sess = ort.InferenceSession(str(merged_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    output = sess.run(None, {in_name: sample})[0]

    # scale=2, bias=1 → each element becomes 1*2+1 = 3.0
    expected = np.full((3, 4), 3.0, dtype=np.float32)
    np.testing.assert_allclose(output, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# merge_onnx_models — error cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_raises_on_missing_preprocessing_file(tmp_path: Path) -> None:
    """MergeError raised when the preprocessing path does not exist."""
    with pytest.raises(MergeError, match="Failed to load preprocessing model"):
        merge_onnx_models(
            preprocessing_path=tmp_path / "nonexistent_pre.onnx",
            model_path=tmp_path / "nonexistent_model.onnx",
            output_path=tmp_path / "merged.onnx",
        )


@pytest.mark.unit
def test_merge_raises_on_missing_model_file(
    toy_graphs: tuple[Path, Path], tmp_path: Path
) -> None:
    """MergeError raised when the model path does not exist."""
    pre_path, _ = toy_graphs
    with pytest.raises(MergeError, match="Failed to load model"):
        merge_onnx_models(
            preprocessing_path=pre_path,
            model_path=tmp_path / "nonexistent_model.onnx",
            output_path=tmp_path / "merged.onnx",
        )


@pytest.mark.unit
def test_merge_parity_detects_shape_mismatch_at_inference(tmp_path: Path) -> None:
    """verify_merge_parity raises MergeError when shapes mismatch at inference.

    Note: ``onnx.compose.merge_models`` itself does not validate shapes at
    graph-build time (ONNX defers shape checking to runtime).  Shape
    incompatibilities surface when ORT tries to run the merged session.
    A 4-feature preprocessing graph feeding a 2-feature model will fail at
    the ``Add`` node inside the model graph.
    """
    pytest.importorskip("onnxruntime", reason="onnxruntime required for parity check")

    pre_path = tmp_path / "pre.onnx"
    model_path = tmp_path / "model.onnx"
    merged_path = tmp_path / "merged.onnx"
    onnx.save(
        _make_scale_model("input", "output", n_features=4),
        str(pre_path),
    )
    onnx.save(
        _make_add_model("model_input", "model_output", n_features=2),
        str(model_path),
    )

    # The merge itself succeeds at graph-build time…
    merge_onnx_models(pre_path, model_path, merged_path)

    # …but inference fails due to the shape mismatch, surfaced as MergeError.
    sample = np.ones((1, 4), dtype=np.float32)
    with pytest.raises(MergeError):
        verify_merge_parity(pre_path, model_path, merged_path, sample_input=sample)


# ---------------------------------------------------------------------------
# merge_onnx_models — exception handlers (mock-based)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_raises_on_compose_failure(
    toy_graphs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MergeError wraps exceptions raised by onnx.compose.merge_models."""
    pre_path, model_path = toy_graphs

    import onnx.compose as _compose

    monkeypatch.setattr(
        _compose,
        "merge_models",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("compose boom")),
    )

    with pytest.raises(MergeError, match="onnx.compose.merge_models failed"):
        merge_onnx_models(pre_path, model_path, tmp_path / "merged.onnx")


@pytest.mark.unit
def test_merge_raises_on_checker_failure(
    toy_graphs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MergeError wraps exceptions raised by onnx.checker.check_model.

    onnx.compose.merge_models calls check_model internally before returning.
    We let that internal call succeed (call 0) and fail on the explicit call
    in merge_onnx_models (call 1).
    """
    pre_path, model_path = toy_graphs

    original_check = onnx.checker.check_model
    call_count: list[int] = [0]

    def _raise_on_second(*a: object, **kw: object) -> None:
        call_count[0] += 1
        if call_count[0] <= 1:
            original_check(*a, **kw)  # type: ignore[arg-type]
        else:
            raise RuntimeError("checker boom")

    monkeypatch.setattr(onnx.checker, "check_model", _raise_on_second)

    with pytest.raises(MergeError, match="Merged model failed ONNX checker validation"):
        merge_onnx_models(pre_path, model_path, tmp_path / "merged.onnx")


@pytest.mark.unit
def test_merge_raises_on_save_failure(
    toy_graphs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MergeError wraps exceptions raised by onnx.save."""
    pre_path, model_path = toy_graphs

    import onnx as _onnx

    monkeypatch.setattr(
        _onnx,
        "save",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(MergeError, match="Failed to save merged model"):
        merge_onnx_models(pre_path, model_path, tmp_path / "merged.onnx")


# ---------------------------------------------------------------------------
# verify_merge_parity — error cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_merge_parity_raises_on_missing_ort(
    toy_graphs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MergeError raised when onnxruntime is not installed."""
    pre_path, model_path = toy_graphs
    merged_path = tmp_path / "merged.onnx"
    merge_onnx_models(pre_path, model_path, merged_path)

    import builtins

    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "onnxruntime":
            raise ImportError("onnxruntime blocked for test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    sample = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(MergeError, match="onnxruntime"):
        verify_merge_parity(pre_path, model_path, merged_path, sample_input=sample)
