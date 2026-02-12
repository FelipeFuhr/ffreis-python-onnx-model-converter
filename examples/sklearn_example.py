#!/usr/bin/env python3
"""Example script for converting scikit-learn models and pipelines to ONNX."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from skl2onnx.common.data_types import FloatTensorType
from onnx_converter import convert_sklearn_to_onnx
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_simple_classifier() -> None:
    """Run a RandomForest scikit-learn conversion example."""
    print("\n" + "=" * 60)
    print("Example 1: Random Forest Classifier")
    print("=" * 60)
    
    # Load data and train model
    print("\n1. Training Random Forest model on Iris dataset...")
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    print("   ✓ Model trained")
    
    # Define output path
    output_path = "outputs/rf_classifier.onnx"
    os.makedirs("outputs", exist_ok=True)
    
    # Convert to ONNX
    print("\n2. Converting to ONNX format...")
    initial_types = [('input', FloatTensorType([None, 4]))]  # 4 features
    
    convert_sklearn_to_onnx(
        model=model,
        output_path=output_path,
        initial_types=initial_types
    )
    
    # Verify and test
    print("\n3. Testing inference with ONNX Runtime...")
    import onnxruntime as ort
    import numpy as np
    
    session = ort.InferenceSession(output_path)
    test_input = X[:5].astype(np.float32)
    predictions = session.run(None, {"input": test_input})
    print(f"   ✓ Inference successful! Predictions: {predictions[0]}")
    
    print(f"\n   SUCCESS! Model saved to: {output_path}")


def example_pipeline() -> None:
    """Run a pipeline conversion example with preprocessing."""
    print("\n" + "=" * 60)
    print("Example 2: Pipeline (Scaler + Random Forest)")
    print("=" * 60)
    
    # Load data and create pipeline
    print("\n1. Creating and training pipeline...")
    X, y = load_iris(return_X_y=True)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    pipeline.fit(X, y)
    print("   ✓ Pipeline trained")
    
    # Define output path
    output_path = "outputs/pipeline.onnx"
    os.makedirs("outputs", exist_ok=True)
    
    # Convert to ONNX
    print("\n2. Converting pipeline to ONNX format...")
    initial_types = [('input', FloatTensorType([None, 4]))]
    
    convert_sklearn_to_onnx(
        model=pipeline,
        output_path=output_path,
        initial_types=initial_types
    )
    
    # Verify and test
    print("\n3. Testing inference with ONNX Runtime...")
    import onnxruntime as ort
    import numpy as np
    
    session = ort.InferenceSession(output_path)
    test_input = X[:5].astype(np.float32)
    predictions = session.run(None, {"input": test_input})
    print(f"   ✓ Inference successful! Predictions: {predictions[0]}")
    
    print(f"\n   SUCCESS! Pipeline saved to: {output_path}")


def main() -> None:
    """Run all scikit-learn conversion examples."""
    print("=" * 60)
    print("Scikit-learn to ONNX Conversion Examples")
    print("=" * 60)
    
    # Run examples
    example_simple_classifier()
    example_pipeline()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
