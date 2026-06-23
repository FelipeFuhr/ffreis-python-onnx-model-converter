"""HuggingFace-Transformers-to-ONNX conversion utilities (D2).

Exports a HuggingFace model to ONNX through ``optimum.exporters.onnx.main_export``,
which auto-detects the task from ``model.config.model_type`` (via ``optimum``'s
``TasksManager``) and selects the right ONNX export configuration. This is the
supported, maintained export path for transformers — far more robust than a hand
``torch.onnx.export`` of a tokenizer-fed model.

``optimum`` + ``transformers`` are a multi-GB install and are intentionally NOT a
hard dependency of this package (mirroring the ``tf_legacy`` isolation policy in
AGENTS.md). They are imported lazily inside the functions, so importing this module
never requires them; a clear :class:`DependencyError` is raised at call time when
absent. Install with ``pip install 'optimum[exporters]' transformers``.
"""

from __future__ import annotations

from pathlib import Path

from onnx_converter.errors import ConversionError
from onnx_converter.errors import DependencyError
from onnx_converter.errors import ParityError

_OPTIMUM_HINT = "install with `pip install 'optimum[exporters]' transformers`"


def convert_hf_to_onnx(
    model_name_or_path: str,
    output_dir: str,
    *,
    task: str = "auto",
    opset: int | None = None,
    device: str = "cpu",
) -> str:
    """Export a HuggingFace model to ONNX and return the produced ``.onnx`` path.

    Parameters
    ----------
    model_name_or_path : str
        A HuggingFace Hub model id or a local directory holding the model + config.
    output_dir : str
        Directory the ONNX artifact(s) are written to (created if missing).
    task : str, default="auto"
        Export task. ``"auto"`` lets ``optimum`` infer it from
        ``model.config.model_type`` (e.g. ``text-classification`` for
        ``AutoModelForSequenceClassification``). Pass an explicit task to override.
    opset : int | None, default=None
        Target ONNX opset; ``None`` uses ``optimum``'s default for the task.
    device : str, default="cpu"
        Device used during export tracing.

    Returns
    -------
    str
        Path to the produced ONNX model file (``model.onnx`` for single-graph
        tasks; the first ``*.onnx`` alphabetically for multi-graph exports).

    Raises
    ------
    DependencyError
        If ``optimum``/``transformers`` are not installed.
    ConversionError
        If the export produced no ``.onnx`` file.
    """
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as exc:
        raise DependencyError(
            f"optimum is required for HuggingFace ONNX export; {_OPTIMUM_HINT}"
        ) from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    main_export(
        model_name_or_path,
        output=str(out),
        task=task,
        opset=opset,
        device=device,
    )

    model_onnx = out / "model.onnx"
    if model_onnx.exists():
        return str(model_onnx)
    # Seq2seq / encoder-decoder tasks emit encoder_model.onnx / decoder_model.onnx.
    candidates = sorted(out.glob("*.onnx"))
    if not candidates:
        raise ConversionError(f"optimum produced no .onnx file in {out}")
    return str(candidates[0])


def verify_hf_onnx_parity(
    model_name_or_path: str,
    onnx_path: str,
    sample_texts: list[str],
    *,
    max_length: int = 128,
    atol: float = 1e-3,
) -> float:
    """Compare source-model logits to ONNX Runtime logits for text classification.

    Tokenizes ``sample_texts`` with the model's tokenizer, runs the original
    transformers ``AutoModelForSequenceClassification`` and an ONNX Runtime session
    on the exported graph, and asserts the logits match within ``atol``.

    Parameters
    ----------
    model_name_or_path : str
        The same model id/path passed to :func:`convert_hf_to_onnx`.
    onnx_path : str
        Path to the exported ONNX model.
    sample_texts : list[str]
        Texts to score through both paths.
    max_length : int, default=128
        Tokenizer truncation/padding length.
    atol : float, default=1e-3
        Absolute tolerance for the logit comparison.

    Returns
    -------
    float
        The maximum absolute difference between the two logit tensors.

    Raises
    ------
    DependencyError
        If ``transformers``/``onnxruntime`` are not installed.
    ParityError
        If the logits diverge beyond ``atol``.
    """
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise DependencyError(
            f"transformers + onnxruntime required for HF parity checks; {_OPTIMUM_HINT}"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    encoded = tokenizer(
        sample_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    source = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    source.eval()
    with torch.no_grad():
        source_logits = source(**encoded).logits.cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {inp.name: encoded[inp.name].cpu().numpy() for inp in session.get_inputs()}
    onnx_logits = session.run(None, feeds)[0]

    max_diff = float(np.max(np.abs(source_logits - onnx_logits)))
    if not np.allclose(source_logits, onnx_logits, atol=atol):
        raise ParityError(
            f"HuggingFace ONNX parity violated: max abs diff {max_diff} > {atol}"
        )
    return max_diff
