"""Unit tests for adapter loaders and parity checker implementations."""

from __future__ import annotations

from pathlib import Path
from sys import modules as sys_modules
from types import SimpleNamespace as types_SimpleNamespace

from numpy import array as np_array
from numpy import array_equal as np_array_equal
from numpy import float32 as np_float32
from numpy import ndarray as np_ndarray
from numpy import save as np_save
from pytest import MonkeyPatch as pytest_MonkeyPatch
from pytest import raises as pytest_raises

from onnx_converter.adapters.loaders import (
    SklearnModelLoader,
    TensorflowModelLoader,
    TorchModelLoader,
)
from onnx_converter.adapters.parity_checkers import (
    SklearnParityChecker,
    TensorflowParityChecker,
    TorchParityChecker,
)
from onnx_converter.application.options import ParityOptions
from onnx_converter.errors import (
    ParityError,
    UnsafeLoadError,
    UnsupportedModelError,
)


def test_torch_loader_rejects_checkpoint_dict(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Raise UnsupportedModelError for checkpoint-like dict payloads."""
    fake_torch = types_SimpleNamespace(
        jit=types_SimpleNamespace(
            load=lambda _: (_ for _ in ()).throw(ValueError("x"))
        ),
        load=lambda *_args, **_kwargs: {"state_dict": {}},
    )
    monkeypatch.setitem(sys_modules, "torch", fake_torch)

    with pytest_raises(UnsupportedModelError, match="checkpoint"):
        TorchModelLoader().load(tmp_path / "m.pt", allow_unsafe=True)


def test_torch_loader_dependency_error(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Raise dependency error when torch cannot be imported."""
    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest_raises(Exception, match="PyTorch is required"):
        TorchModelLoader().load(tmp_path / "m.pt")


def test_torch_loader_requires_allow_unsafe_for_fallback(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Raise UnsafeLoadError when safe fallback fails and unsafe mode is off."""
    fake_torch = types_SimpleNamespace(
        jit=types_SimpleNamespace(
            load=lambda _: (_ for _ in ()).throw(ValueError("x"))
        ),
        load=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    monkeypatch.setitem(sys_modules, "torch", fake_torch)

    with pytest_raises(UnsafeLoadError, match="safe torch.load fallback"):
        TorchModelLoader().load(tmp_path / "m.pt", allow_unsafe=False)


def test_torch_loader_unsafe_fallback_returns_model(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Load using unsafe fallback when explicitly allowed."""
    state = {"calls": 0}

    def fake_load(
        _path: object,
        *,
        map_location: str = "cpu",
        weights_only: bool = True,
    ) -> object:
        del map_location
        state["calls"] += 1
        if weights_only:
            raise ValueError("safe path fails")
        return object()

    fake_torch = types_SimpleNamespace(
        jit=types_SimpleNamespace(
            load=lambda _: (_ for _ in ()).throw(ValueError("x"))
        ),
        load=fake_load,
    )
    monkeypatch.setitem(sys_modules, "torch", fake_torch)

    model = TorchModelLoader().load(tmp_path / "m.pt", allow_unsafe=True)
    assert model is not None
    assert state["calls"] == 2


def test_tensorflow_loader_directory_and_file_paths(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Return string for SavedModel directories and load keras file paths."""
    loaded: dict[str, str] = {}
    fake_tf = types_SimpleNamespace(
        keras=types_SimpleNamespace(
            models=types_SimpleNamespace(
                load_model=lambda path: loaded.setdefault("path", path)
            )
        )
    )
    monkeypatch.setitem(sys_modules, "tensorflow", fake_tf)

    model_dir = tmp_path / "saved_model"
    model_dir.mkdir()
    model_file = tmp_path / "model.keras"
    model_file.write_text("x", encoding="utf-8")

    loader = TensorflowModelLoader()
    assert loader.load(model_dir) == str(model_dir)
    assert loader.load(model_file) == str(model_file)


def test_tensorflow_loader_dependency_error(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Raise dependency error when tensorflow cannot be imported."""
    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tensorflow":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest_raises(Exception, match="TensorFlow is required"):
        TensorflowModelLoader().load(tmp_path / "m.keras")


def test_sklearn_loader_paths_and_errors(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Cover sklearn loader success and failure branches."""
    loader = SklearnModelLoader()
    with pytest_raises(UnsafeLoadError):
        loader.load(tmp_path / "x.pkl", allow_unsafe=False)

    monkeypatch.setitem(
        sys_modules, "joblib", types_SimpleNamespace(load=lambda p: ("joblib", p))
    )
    assert loader.load(tmp_path / "x.pkl", allow_unsafe=True)[0] == "joblib"

    skops_io = types_SimpleNamespace(load=lambda p: ("skops", p))
    monkeypatch.setitem(sys_modules, "skops.io", skops_io)
    assert loader.load(tmp_path / "x.skops", allow_unsafe=False)[0] == "skops"

    monkeypatch.delitem(sys_modules, "skops.io", raising=False)
    with pytest_raises(Exception, match="skops is required"):
        loader.load(tmp_path / "x.skops", allow_unsafe=False)

    monkeypatch.setitem(sys_modules, "joblib", None)
    with pytest_raises(Exception, match="joblib is required"):
        loader.load(tmp_path / "x.pkl", allow_unsafe=True)

    with pytest_raises(UnsupportedModelError, match="Unsupported model file extension"):
        loader.load(tmp_path / "x.abc", allow_unsafe=True)


def test_torch_parity_checker_dependency_error(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Raise ParityError when torch is unavailable."""
    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    parity = ParityOptions(input_path=tmp_path / "x.npy")

    with pytest_raises(ParityError, match="requires torch"):
        TorchParityChecker().check(
            model=object(), onnx_path=tmp_path / "m.onnx", parity=parity
        )


def test_torch_parity_checker_success(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Compute expected torch output and forward it to tensor parity checker."""
    import onnx_converter.adapters.parity_checkers as module

    parity_input = np_array([[1.0, 2.0]], dtype=np_float32)
    monkeypatch.setattr(module, "load_parity_input", lambda _: parity_input)
    captured: dict[str, object] = {}

    def fake_check_tensor_parity(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(module, "check_tensor_parity", fake_check_tensor_parity)

    class FakeTensor:
        """Simple tensor-like object supporting expected torch chain."""

        def to(self, _dtype: object) -> FakeTensor:
            return self

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def numpy(self) -> np_ndarray:
            return np_array([[3.0, 4.0]], dtype=np_float32)

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

    fake_torch = types_SimpleNamespace(
        no_grad=lambda: _NoGrad(),
        from_numpy=lambda arr: FakeTensor(),
        float32="float32",
    )
    monkeypatch.setitem(sys_modules, "torch", fake_torch)

    class FakeModel:
        def __call__(self, _tensor: FakeTensor) -> tuple[FakeTensor]:
            return (FakeTensor(),)

    parity = ParityOptions(input_path=tmp_path / "x.npy", atol=1e-6, rtol=1e-5)
    TorchParityChecker().check(FakeModel(), tmp_path / "m.onnx", parity)
    assert captured["label"] == "PyTorch"


def test_tensorflow_parity_checker_saved_model_path_not_supported(
    tmp_path: Path,
) -> None:
    """Reject parity checks for SavedModel path strings."""
    parity = ParityOptions(input_path=tmp_path / "x.npy")
    np_save(tmp_path / "x.npy", np_array([[1.0, 2.0]], dtype=np_float32))
    with pytest_raises(ParityError, match="not supported yet"):
        TensorflowParityChecker().check(
            model="saved_model", onnx_path=tmp_path / "m.onnx", parity=parity
        )


def test_tensorflow_parity_checker_success(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Handle tuple/list model outputs and pass first tensor to parity checker."""
    import onnx_converter.adapters.parity_checkers as module

    parity_input = np_array([[1.0, 2.0]], dtype=np_float32)
    monkeypatch.setattr(module, "load_parity_input", lambda _: parity_input)
    captured: dict[str, object] = {}

    def fake_check_tensor_parity(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(module, "check_tensor_parity", fake_check_tensor_parity)

    class FakeOutput:
        def numpy(self) -> np_ndarray:
            return np_array([[5.0, 6.0]], dtype=np_float32)

    class FakeModel:
        def __call__(
            self, _inputs: np_ndarray, training: bool = False
        ) -> tuple[FakeOutput]:
            del training
            return (FakeOutput(),)

    parity = ParityOptions(input_path=tmp_path / "x.npy", atol=1e-6, rtol=1e-5)
    TensorflowParityChecker().check(FakeModel(), tmp_path / "m.onnx", parity)
    assert captured["label"] == "TensorFlow"


def test_sklearn_parity_checker_calls_backend(
    monkeypatch: pytest_MonkeyPatch, tmp_path: Path
) -> None:
    """Forward loaded parity input into sklearn parity backend helper."""
    calls: dict[str, object] = {}
    parity_input = np_array([[1.0, 2.0]], dtype=np_float32)
    np_save(tmp_path / "x.npy", parity_input)

    import onnx_converter.adapters.parity_checkers as module

    monkeypatch.setattr(module, "load_parity_input", lambda _: parity_input)

    def fake_check_sklearn_parity(**kwargs: object) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(module, "check_sklearn_parity", fake_check_sklearn_parity)
    parity = ParityOptions(input_path=tmp_path / "x.npy", atol=1e-6, rtol=1e-5)
    model = object()
    onnx_path = tmp_path / "m.onnx"
    SklearnParityChecker().check(model=model, onnx_path=onnx_path, parity=parity)

    assert calls["model"] is model
    assert calls["onnx_path"] == onnx_path
    assert np_array_equal(calls["parity_input"], parity_input)


def test_sklearn_parity_checker_skips_when_input_missing(tmp_path: Path) -> None:
    """Return early when parity input is not configured."""
    parity = ParityOptions(input_path=None)
    SklearnParityChecker().check(
        model=object(), onnx_path=tmp_path / "m.onnx", parity=parity
    )
