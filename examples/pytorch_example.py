#!/usr/bin/env python3
"""Example script for converting a tiny PyTorch model to ONNX."""

from __future__ import annotations

from pathlib import Path

from numpy import abs as np_abs
from numpy import allclose as np_allclose
from numpy import array as np_array
from numpy import float32 as np_float32
from numpy import max as np_max
from onnx import checker as onnx_checker
from onnx import load as onnx_load
from onnxruntime import InferenceSession as ort_InferenceSession
from torch import Tensor as torch_Tensor
from torch import from_numpy as torch_from_numpy
from torch import manual_seed as torch_manual_seed
from torch import no_grad as torch_no_grad
from torch.nn import Linear as nn_Linear
from torch.nn import Module as nn_Module
from torch.nn import ReLU as nn_ReLU
from torch.nn import Sequential as nn_Sequential

from onnx_converter import convert_pytorch_to_onnx


class TinyNet(nn_Module):
    """Small deterministic network for conversion/inference checks."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn_Sequential(
            nn_Linear(4, 8),
            nn_ReLU(),
            nn_Linear(8, 3),
        )

    def forward(self, x: torch_Tensor) -> torch_Tensor:
        """Run a forward pass through the tiny network."""
        return self.net(x)


def main() -> None:
    """Convert and verify a tiny PyTorch model."""
    print("=" * 60)
    print("PyTorch to ONNX Conversion Example")
    print("=" * 60)

    torch_manual_seed(7)
    model = TinyNet().eval()

    output_path = Path("outputs/tinynet.onnx")
    output_path.parent.mkdir(exist_ok=True)

    convert_pytorch_to_onnx(
        model=model,
        output_path=str(output_path),
        input_shape=(1, 4),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )

    if not output_path.exists():
        raise SystemExit("FAIL: ONNX file was not created.")

    onnx_model = onnx_load(str(output_path))
    onnx_checker.check_model(onnx_model)
    print("ONNX graph validated.")

    batch = np_array(
        [[0.1, -0.2, 0.3, 0.4], [0.5, -0.6, 0.7, -0.8]],
        dtype=np_float32,
    )

    with torch_no_grad():
        torch_out = model(torch_from_numpy(batch)).cpu().numpy()

    session = ort_InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["output"], {"input": batch})[0]

    if torch_out.shape != onnx_out.shape:
        raise SystemExit(
            f"FAIL: shape mismatch (torch={torch_out.shape}, onnx={onnx_out.shape})."
        )

    max_abs_diff = float(np_max(np_abs(torch_out - onnx_out)))
    print(f"Max abs diff: {max_abs_diff:.8f}")
    if not np_allclose(torch_out, onnx_out, atol=1e-5, rtol=1e-4):
        raise SystemExit(
            "FAIL: output mismatch "
            f"(max_abs_diff={max_abs_diff:.8f}, atol=1e-5, rtol=1e-4)."
        )

    print(f"PASS: {output_path}")


if __name__ == "__main__":
    main()
