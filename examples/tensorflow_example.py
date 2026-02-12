#!/usr/bin/env python3
"""Example script for converting a pretrained TensorFlow/Keras model to ONNX."""
import tensorflow as tf
from onnx_converter import convert_tensorflow_to_onnx
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    """Run the TensorFlow/Keras-to-ONNX conversion example."""
    print("=" * 60)
    print("TensorFlow/Keras to ONNX Conversion Example")
    print("=" * 60)
    
    # Load a pretrained MobileNetV2 model
    print("\n1. Loading pretrained MobileNetV2 model...")
    model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    print("   ✓ Model loaded")
    
    # Define output path
    output_path = "outputs/mobilenet_v2.onnx"
    os.makedirs("outputs", exist_ok=True)
    
    # Convert to ONNX
    print("\n2. Converting to ONNX format...")
    input_signature = [tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image")]
    
    convert_tensorflow_to_onnx(
        model=model,
        output_path=output_path,
        input_signature=input_signature,
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
    
    # Create random input (normalized to [0, 1])
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy_input})
    print(f"   ✓ Inference successful! Output shape: {outputs[0].shape}")
    
    print("\n" + "=" * 60)
    print(f"SUCCESS! ONNX model saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
