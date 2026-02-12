"""Shared test helpers for converter tests."""

from __future__ import annotations

import pytest


class FakeParityChecker:
    """Fake parity checker for testing."""
    def check(self, *args, **kwargs):
        pass


def mock_converter_dependencies(monkeypatch: pytest.MonkeyPatch, framework: str = "torch") -> None:
    """Mock converter, postprocessor, and parity checker dependencies.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        framework: Framework name ("torch" or "tensorflow")
    """
    import onnx_converter.postprocess
    import onnx_converter.adapters.parity_checkers
    
    # Mock postprocess functions to avoid loading ONNX files
    monkeypatch.setattr(onnx_converter.postprocess, "add_standard_metadata", lambda **kwargs: None)
    monkeypatch.setattr(onnx_converter.postprocess, "add_onnx_metadata", lambda *args, **kwargs: None)
    
    # Mock parity checker to avoid dependencies
    if framework == "torch":
        monkeypatch.setattr(onnx_converter.adapters.parity_checkers, "TorchParityChecker", FakeParityChecker)
    elif framework == "tensorflow":
        monkeypatch.setattr(onnx_converter.adapters.parity_checkers, "TensorflowParityChecker", FakeParityChecker)
