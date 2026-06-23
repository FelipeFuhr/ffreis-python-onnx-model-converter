# ruff: noqa: D101, D102, E501
# flake8: noqa: E501
"""Cover convert_custom_file use case end-to-end with a fake plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from onnx_converter.application import use_cases
from onnx_converter.application.use_cases import ConversionResult
from onnx_converter.application.use_cases import convert_custom_file


class FakePlugin:
    name = "fake-plugin"

    def __init__(self) -> None:
        self.convert_calls: list[tuple[Path, Path, dict[str, Any]]] = []

    def convert(
        self,
        *,
        model_path: Path,
        output_path: Path,
        options: dict[str, Any],
    ) -> Path:
        self.convert_calls.append((model_path, output_path, dict(options)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"onnx-bytes")
        return output_path


class FakeRegistry:
    def __init__(self, plugin: FakePlugin) -> None:
        self.plugin = plugin
        self.resolve_calls: list[dict[str, Any]] = []

    def resolve(
        self,
        *,
        model_path: Path,
        model_type: str | None,
        plugin_name: str | None,
        options: dict[str, Any],
    ) -> FakePlugin:
        self.resolve_calls.append(
            {
                "model_path": model_path,
                "model_type": model_type,
                "plugin_name": plugin_name,
                "options": dict(options),
            }
        )
        return self.plugin


class TestConvertCustomFile:
    def test_resolves_and_runs_plugin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plugin = FakePlugin()
        registry = FakeRegistry(plugin)
        monkeypatch.setattr(
            use_cases, "create_default_registry", lambda extra_modules=None: registry
        )
        result = convert_custom_file(
            model_path=tmp_path / "model.bin",
            output_path=tmp_path / "out.onnx",
            model_type="custom",
            plugin_name="fake-plugin",
            plugin_modules=["my.module"],
            options={"foo": "bar"},
        )
        assert isinstance(result, ConversionResult)
        assert result.framework == "plugin:fake-plugin"
        assert result.output_path == tmp_path / "out.onnx"
        assert result.metadata is None

    def test_registry_resolve_receives_arguments(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plugin = FakePlugin()
        registry = FakeRegistry(plugin)
        monkeypatch.setattr(
            use_cases, "create_default_registry", lambda extra_modules=None: registry
        )
        convert_custom_file(
            model_path=tmp_path / "model.bin",
            output_path=tmp_path / "out.onnx",
            model_type="custom",
            plugin_name="fake-plugin",
            plugin_modules=None,
            options={"key": 42},
        )
        call = registry.resolve_calls[0]
        assert call["model_type"] == "custom"
        assert call["plugin_name"] == "fake-plugin"
        assert call["options"] == {"key": 42}

    def test_plugin_convert_receives_arguments(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plugin = FakePlugin()
        registry = FakeRegistry(plugin)
        monkeypatch.setattr(
            use_cases, "create_default_registry", lambda extra_modules=None: registry
        )
        convert_custom_file(
            model_path=tmp_path / "model.bin",
            output_path=tmp_path / "out.onnx",
            model_type=None,
            plugin_name=None,
            plugin_modules=None,
            options={"opt": True},
        )
        assert len(plugin.convert_calls) == 1
        mp, op, opts = plugin.convert_calls[0]
        assert mp == tmp_path / "model.bin"
        assert op == tmp_path / "out.onnx"
        assert opts == {"opt": True}

    def test_options_copied_not_mutated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plugin = FakePlugin()
        registry = FakeRegistry(plugin)
        monkeypatch.setattr(
            use_cases, "create_default_registry", lambda extra_modules=None: registry
        )
        original = {"foo": "bar"}
        convert_custom_file(
            model_path=tmp_path / "m.bin",
            output_path=tmp_path / "o.onnx",
            model_type=None,
            plugin_name=None,
            plugin_modules=None,
            options=original,
        )
        # convert_custom_file builds option_map = dict(options); even if plugin
        # mutates its received options, the caller's dict should be untouched.
        assert original == {"foo": "bar"}
