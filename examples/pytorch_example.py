#!/usr/bin/env python3
"""Example script for converting a pretrained PyTorch model to ONNX."""
import torch
import torchvision.models as models
from onnx_converter import convert_pytorch_to_onnx
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    """Run the PyTorch-to-ONNX conversion example."""
    print("=" * 60)
    print("PyTorch to ONNX Conversion Example")
    print("=" * 60)
    
    # Load a pretrained ResNet18 model
    print("\n1. Loading pretrained ResNet18 model...")
    model = models.resnet18(pretrained=True)
    model.eval()
    print("   ✓ Model loaded")
    
    # Define output path
    output_path = "outputs/resnet18.onnx"
    os.makedirs("outputs", exist_ok=True)
    
    # Convert to ONNX
    print("\n2. Converting to ONNX format...")
    input_shape = (1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
    
    convert_pytorch_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=input_shape,
        input_names=["image"],
        output_names=["class_probabilities"],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'class_probabilities': {0: 'batch_size'}
        },
        opset_version=14
    )
    
    # Verify the ONNX model
    print("\n3. Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("   ✓ ONNX model is valid")
    
    # Test inference with ONNX Runtime
    print("\n4. Testing inference with ONNX Runtime...")
    import onnxruntime as ort
    import numpy as np
    
    # Create session
    session = ort.InferenceSession(output_path)
    
    # Create random input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {"image": dummy_input})
    print(f"   ✓ Inference successful! Output shape: {outputs[0].shape}")
    
    print("\n" + "=" * 60)
    print(f"SUCCESS! ONNX model saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
