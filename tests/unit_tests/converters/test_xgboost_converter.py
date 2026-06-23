"""Unit tests for the XGBoost-to-ONNX converter.

Tests validate:
- Missing onnxmltools raises a clear ConversionError (import-guard path).
- Invalid n_features raises ConversionError before any model is touched.
- The public API function delegates to the impl.
- Happy path: classifier + regressor produce a valid ONNX file with parity.
- The top-level ``onnx_converter.convert_xgboost_to_onnx`` wrapper re-exports
  the same behaviour.

These tests use ``pytest.importorskip`` so a missing xgboost/onnxmltools install
causes a clear skip rather than a collection failure.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestXgboostConverterImportGuard:
    """Validate the ConversionError raised when onnxmltools is missing."""

    def test_missing_onnxmltools_raises_conversion_error(self, tmp_path: Path) -> None:
        """A missing onnxmltools import must surface as ConversionError."""
        from onnx_converter.errors import ConversionError

        mock_model = MagicMock()

        with patch.dict("sys.modules", {"onnxmltools": None, "skl2onnx": None}):
            from importlib import reload

            import onnx_converter.converters.xgboost_converter as xgb_mod

            reload(xgb_mod)
            with pytest.raises(ConversionError, match="onnxmltools"):
                xgb_mod.convert_xgboost_to_onnx(
                    model=mock_model,
                    output_path=str(tmp_path / "out.onnx"),
                    n_features=4,
                )


class TestXgboostConverterValidation:
    """Validate input validation errors before conversion."""

    def test_invalid_n_features_raises(self, tmp_path: Path) -> None:
        """n_features <= 0 must raise ConversionError immediately."""
        from onnx_converter.converters.xgboost_converter import convert_xgboost_to_onnx
        from onnx_converter.errors import ConversionError

        mock_model = MagicMock()
        with pytest.raises(
            ConversionError, match="n_features must be a positive integer"
        ):
            convert_xgboost_to_onnx(
                model=mock_model,
                output_path=str(tmp_path / "out.onnx"),
                n_features=0,
            )

    def test_negative_n_features_raises(self, tmp_path: Path) -> None:
        """Negative n_features must also raise ConversionError."""
        from onnx_converter.converters.xgboost_converter import convert_xgboost_to_onnx
        from onnx_converter.errors import ConversionError

        mock_model = MagicMock()
        with pytest.raises(
            ConversionError, match="n_features must be a positive integer"
        ):
            convert_xgboost_to_onnx(
                model=mock_model,
                output_path=str(tmp_path / "out.onnx"),
                n_features=-1,
            )


class TestXgboostConverterHappyPath:
    """Integration-style tests requiring xgboost, onnxmltools, and onnxruntime."""

    xgboost = pytest.importorskip("xgboost", reason="xgboost not installed")
    onnxmltools = pytest.importorskip("onnxmltools", reason="onnxmltools not installed")
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    np = pytest.importorskip("numpy", reason="numpy not installed")

    def _make_classifier(self, seed: int = 42) -> object:
        """Return a fitted XGBClassifier on synthetic 2-class data."""
        import numpy as np
        from xgboost import XGBClassifier

        rng = np.random.default_rng(seed)
        X = rng.standard_normal((80, 6)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        clf = XGBClassifier(
            n_estimators=5,
            max_depth=2,
            random_state=seed,
            eval_metric="logloss",
        )
        clf.fit(X, y)
        return clf

    def _make_regressor(self, seed: int = 42) -> object:
        """Return a fitted XGBRegressor on synthetic data."""
        import numpy as np
        from xgboost import XGBRegressor

        rng = np.random.default_rng(seed)
        X = rng.standard_normal((80, 6)).astype(np.float32)
        y = rng.standard_normal(80).astype(np.float32)
        reg = XGBRegressor(n_estimators=5, max_depth=2, random_state=seed)
        reg.fit(X, y)
        return reg

    def test_classifier_produces_onnx_file(self, tmp_path: Path) -> None:
        """XGBClassifier export must produce a non-empty ONNX file."""
        from onnx_converter.converters.xgboost_converter import convert_xgboost_to_onnx

        model = self._make_classifier()
        out = convert_xgboost_to_onnx(
            model=model,
            output_path=str(tmp_path / "clf.onnx"),
            n_features=6,
        )
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_regressor_produces_onnx_file(self, tmp_path: Path) -> None:
        """XGBRegressor export must produce a non-empty ONNX file."""
        from onnx_converter.converters.xgboost_converter import convert_xgboost_to_onnx

        model = self._make_regressor()
        out = convert_xgboost_to_onnx(
            model=model,
            output_path=str(tmp_path / "reg.onnx"),
            n_features=6,
        )
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_top_level_api_delegates(self, tmp_path: Path) -> None:
        """The top-level ``onnx_converter.convert_xgboost_to_onnx`` must work."""
        from onnx_converter import convert_xgboost_to_onnx

        model = self._make_classifier()
        out = convert_xgboost_to_onnx(
            model=model,
            output_path=str(tmp_path / "api_clf.onnx"),
            n_features=6,
        )
        assert Path(out).exists()

    def test_output_is_valid_onnx_graph(self, tmp_path: Path) -> None:
        """The produced ONNX file must load without error under ONNX Runtime."""
        import onnxruntime as ort

        from onnx_converter.converters.xgboost_converter import convert_xgboost_to_onnx

        model = self._make_regressor()
        out = convert_xgboost_to_onnx(
            model=model,
            output_path=str(tmp_path / "valid.onnx"),
            n_features=6,
        )
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        assert sess.get_inputs()[0].name is not None

    def test_regressor_onnx_parity(self, tmp_path: Path) -> None:
        """XGBRegressor ONNX predictions must match sklearn API within atol=1e-4."""
        import numpy as np
        import onnxruntime as ort
        from xgboost import XGBRegressor

        from onnx_converter.converters.xgboost_converter import convert_xgboost_to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 6)).astype(np.float32)
        y = rng.standard_normal(40).astype(np.float32)
        X_test = rng.standard_normal((20, 6)).astype(np.float32)

        model = XGBRegressor(n_estimators=5, max_depth=2, random_state=0)
        model.fit(X, y)
        xgb_preds = model.predict(X_test)

        out = convert_xgboost_to_onnx(
            model=model,
            output_path=str(tmp_path / "parity.onnx"),
            n_features=6,
        )
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        onnx_preds = sess.run(None, {input_name: X_test})[0].flatten()

        max_diff = float(np.max(np.abs(xgb_preds - onnx_preds)))
        err_msg = f"XGBoost/ONNX parity failed: max diff = {max_diff}"
        assert np.allclose(xgb_preds, onnx_preds, atol=1e-4), err_msg
