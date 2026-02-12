"""Application-layer result objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from typing import Optional


@dataclass(frozen=True)
class ConversionResult:
    """Structured conversion outcome."""

    output_path: Path
    framework: str
    source_path: Path
    metadata: Optional[Mapping[str, str]] = None
