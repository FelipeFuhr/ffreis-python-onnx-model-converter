"""Integration tests for the HuggingFace converter backend (D2).

Exercises the real ``optimum.exporters.onnx`` export path plus an ONNX Runtime
parity check. Skipped unless the multi-GB ``[hf]`` extra (optimum + transformers)
is installed; the tiny ``hf-internal-testing`` model keeps the download small when
it does run.
"""

from __future__ import annotations

from pathlib import Path

from pytest import importorskip as pytest_importorskip

# A purpose-built tiny model for tests (a few hundred KB) — keeps the download cheap.
_TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"


def test_hf_convert_and_parity(tmp_path: Path) -> None:
    """Export a tiny HF classifier to ONNX and assert ORT logit parity."""
    pytest_importorskip("optimum")
    pytest_importorskip("transformers")
    pytest_importorskip("onnxruntime")

    from onnx_converter.converters.hf_converter import (
        convert_hf_to_onnx,
        verify_hf_onnx_parity,
    )

    onnx_path = convert_hf_to_onnx(
        _TINY_MODEL,
        str(tmp_path),
        task="text-classification",
    )

    assert Path(onnx_path).exists()
    assert Path(onnx_path).stat().st_size > 0

    max_diff = verify_hf_onnx_parity(
        _TINY_MODEL,
        onnx_path,
        sample_texts=["a great film", "a terrible film"],
        max_length=16,
    )
    assert max_diff < 1e-3
