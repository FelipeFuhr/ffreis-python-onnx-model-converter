"""Custom error types for ONNX conversion workflows."""


class ConversionError(RuntimeError):
    """Raised when a conversion fails in a user-facing way."""
