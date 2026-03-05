"""Shared test helpers for converter tests."""

from __future__ import annotations

from pytest import MonkeyPatch as pytest_MonkeyPatch

from onnx_converter import postprocess as onnx_converter_postprocess
from onnx_converter.adapters import (
    parity_checkers as onnx_converter_adapters_parity_checkers,
)


class FakeParityChecker:
    """Fake parity checker for testing."""

    def check(self, *args: object, **kwargs: object) -> None:
        """Accept parity-check inputs without performing validation."""
        # Intentionally no-op for converter dependency isolation in unit tests.
        del args, kwargs


def mock_converter_dependencies(
    monkeypatch: pytest_MonkeyPatch,
    framework: str = "torch",
) -> None:
    """Mock converter, postprocessor, and parity checker dependencies.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    framework : str, default="torch"
        Framework name (``"torch"`` or ``"tensorflow"``).

    """
    # Mock postprocess functions to avoid loading ONNX files
    monkeypatch.setattr(
        onnx_converter_postprocess,
        "add_standard_metadata",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        onnx_converter_postprocess,
        "add_onnx_metadata",
        lambda *args, **kwargs: None,
    )

    # Mock parity checker to avoid dependencies
    if framework == "torch":
        monkeypatch.setattr(
            onnx_converter_adapters_parity_checkers,
            "TorchParityChecker",
            FakeParityChecker,
        )
    elif framework == "tensorflow":
        monkeypatch.setattr(
            onnx_converter_adapters_parity_checkers,
            "TensorflowParityChecker",
            FakeParityChecker,
        )
