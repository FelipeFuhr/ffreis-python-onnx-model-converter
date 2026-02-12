"""Pydantic schemas for runtime validation of conversion inputs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator


class TorchFileConversionConfig(BaseModel):
    """Validated input for file-based PyTorch conversion."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    output_path: Path
    input_shape: tuple[int, ...]
    opset_version: int = Field(default=14, ge=1)
    allow_unsafe: bool = False

    @field_validator("input_shape")
    @classmethod
    def _validate_input_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("input_shape must contain at least one dimension.")
        if any(dim <= 0 for dim in value):
            raise ValueError("input_shape dimensions must be positive integers.")
        return value


class TensorflowFileConversionConfig(BaseModel):
    """Validated input for file-based TensorFlow conversion."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    output_path: Path
    opset_version: int = Field(default=14, ge=1)


class SklearnFileConversionConfig(BaseModel):
    """Validated input for file-based scikit-learn conversion."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    output_path: Path
    n_features: int = Field(gt=0)
    allow_unsafe: bool = False
