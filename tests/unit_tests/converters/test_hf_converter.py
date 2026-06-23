"""Unit tests for the HuggingFace converter dependency guard (D2).

The conversion path itself needs the multi-GB ``[hf]`` extra (see the integration
test). These unit tests assert the public surface is wired and that a clear,
typed :class:`DependencyError` is raised when ``optimum`` is absent — the contract
that keeps the core package installable without the heavy stack.
"""

from __future__ import annotations

from importlib.util import find_spec

from pytest import raises as pytest_raises

from onnx_converter.converters import convert_hf_to_onnx
from onnx_converter.errors import DependencyError


def test_public_export_is_wired() -> None:
    """The lazy wrapper is exported from the converters package surface."""
    from onnx_converter import converters

    assert "convert_hf_to_onnx" in converters.__all__
    assert callable(convert_hf_to_onnx)


def test_missing_optimum_raises_dependency_error(tmp_path: object) -> None:
    """Without optimum installed, conversion raises a typed DependencyError."""
    if find_spec("optimum") is not None:  # pragma: no cover — only with [hf] extra
        return
    with pytest_raises(DependencyError, match="optimum is required"):
        convert_hf_to_onnx("some-model", str(tmp_path))
