"""Shared pytest configuration and marker assignment."""

from __future__ import annotations

from pathlib import Path

from pytest import Config as pytest_Config
from pytest import Item as pytest_Item
from pytest import mark as pytest_mark


def pytest_collection_modifyitems(
    config: pytest_Config, items: list[pytest_Item]
) -> None:
    """Attach suite markers based on test file path."""
    del config
    for item in items:
        path = Path(str(item.fspath))
        parts = set(path.parts)
        if "e2e_tests" in parts:
            item.add_marker(pytest_mark.e2e)
        elif "integration_tests" in parts:
            item.add_marker(pytest_mark.integration)
        elif "unit_tests" in parts:
            item.add_marker(pytest_mark.unit)
