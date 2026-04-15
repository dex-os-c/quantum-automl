"""
quantum_automl.explainability
==============================

Classical Explainable AI (XAI) on top of quantum model predictions.

Why classical XAI on quantum models?
-------------------------------------
Quantum circuits are fundamentally black boxes from a data-science perspective.
SHAP (SHapley Additive exPlanations) treats the model as a function
``f: R^d → R`` and computes feature importance without needing to inspect
the circuit internals.  This gives data scientists the interpretability they
need regardless of whether the underlying model is classical or quantum.

Requirements
------------
Install the ``advanced`` extras::

    pip install "quantum-automl[advanced]"

or just::

    pip install shap
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class QuantumExplainer:
    """
    SHAP-based explainer for fitted quantum AutoML models.

    Uses ``shap.KernelExplainer`` (model-agnostic) which works with any
    sklearn-compatible estimator — including VQC, QSVC, VQR, and QSVR.

    Parameters
    ----------
    automl_model : QuantumAutoClassifier | QuantumAutoRegressor
        A **fitted** AutoML estimator.
    background_samples : int
        Number of background samples passed to KernelExplainer.
        Fewer is faster but less accurate.  Recommended: 20-50.

    Examples
    --------
    >>> from quantum_automl import QuantumAutoClassifier
    >>> from quantum_automl.explainability import QuantumExplainer
    >>> clf.fit(X_train, y_train)
    >>> explainer = QuantumExplainer(clf, background_samples=20)
    >>> shap_values = explainer.explain(X_test[:10])
    >>> explainer.summary_plot(shap_values, X_test[:10])
    """

    def __init__(
        self,
        automl_model: Any,
        background_samples: int = 20,
    ) -> None:
        try:
            import shap as _shap
            self._shap = _shap
        except ImportError as exc:
            raise ImportError(
                "shap is required for QuantumExplainer. "
                "Install it with: pip install shap\n"
                "or: pip install 'quantum-automl[advanced]'"
            ) from exc

        if not hasattr(automl_model, "best_model_"):
            raise ValueError(
                "automl_model must be a fitted QuantumAutoClassifier or "
                "QuantumAutoRegressor (call .fit() first)."
            )

        self.automl_model = automl_model
        self.background_samples = background_samples
        self._explainer: Any = None
        self._background_data: np.ndarray | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(self, X_background: Any) -> "QuantumExplainer":
        """
        Fit the KernelExplainer on background data.

        Parameters
        ----------
        X_background : array-like, shape (n_samples, n_features)
            Reference data for SHAP baseline.  Use a representative sample
            of the training set (e.g. ``shap.sample(X_train, 50)``).

        Returns
        -------
        self
        """
        X_arr = np.array(X_background, dtype=np.float64)
        if X_arr.shape[0] > self.background_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_arr.shape[0], size=self.background_samples, replace=False)
            X_arr = X_arr[idx]

        self._background_data = X_arr
        predict_fn = self._get_predict_fn()
        self._explainer = self._shap.KernelExplainer(predict_fn, X_arr)
        logger.info(
            "QuantumExplainer fitted on %d background samples.", X_arr.shape[0]
        )
        return self

    def explain(
        self,
        X: Any,
        nsamples: int = 100,
    ) -> Any:
        """
        Compute SHAP values for ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        nsamples : int
            Coalitions per sample.  More = more accurate but slower.
            Keep at 50-100 for low-end hardware.

        Returns
        -------
        shap_values : np.ndarray or list of np.ndarray
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit(X_background) before .explain().")
        X_arr = np.array(X, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = self._explainer.shap_values(X_arr, nsamples=nsamples)
        return shap_values

    def summary_plot(
        self,
        shap_values: Any,
        X: Any,
        feature_names: list[str] | None = None,
        plot_type: str = "bar",
    ) -> None:
        """
        Render a SHAP summary plot.

        Requires ``matplotlib``.

        Parameters
        ----------
        shap_values : np.ndarray or list
        X : array-like
        feature_names : list[str] | None
        plot_type : str
            "bar" | "dot" | "violin"
        """
        X_arr = np.array(X, dtype=np.float64)
        names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]
        try:
            self._shap.summary_plot(
                shap_values,
                X_arr,
                feature_names=names,
                plot_type=plot_type,
                show=True,
            )
        except Exception as exc:
            warnings.warn(
                f"Could not render summary_plot: {exc}\n"
                "Ensure matplotlib is installed: pip install matplotlib",
                UserWarning,
                stacklevel=2,
            )

    def waterfall_plot(
        self,
        shap_values: Any,
        sample_index: int = 0,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Render a SHAP waterfall plot for a single prediction.

        Parameters
        ----------
        shap_values : np.ndarray
        sample_index : int
        feature_names : list[str] | None
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit() first.")

        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            warnings.warn(
                "matplotlib is required for waterfall_plot.",
                UserWarning,
                stacklevel=2,
            )
            return

        sv = shap_values
        if isinstance(sv, list):
            sv = sv[0]  # first class for multiclass

        names = feature_names or [f"feature_{i}" for i in range(sv.shape[1])]
        explanation = self._shap.Explanation(
            values=sv[sample_index],
            base_values=self._explainer.expected_value,
            data=self._background_data[0] if self._background_data is not None else None,
            feature_names=names,
        )
        self._shap.waterfall_plot(explanation)

    def feature_importance(
        self,
        shap_values: Any,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Compute mean absolute SHAP values as a feature importance dict.

        Parameters
        ----------
        shap_values : np.ndarray or list
        feature_names : list[str] | None

        Returns
        -------
        dict[str, float] : feature → mean |SHAP value|, sorted descending
        """
        sv = shap_values
        if isinstance(sv, list):
            sv = np.mean(np.abs(np.array(sv)), axis=0)
        importance = np.mean(np.abs(sv), axis=0)
        n = len(importance)
        names = feature_names or [f"feature_{i}" for i in range(n)]
        result = {name: float(imp) for name, imp in zip(names, importance)}
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_predict_fn(self) -> Any:
        """Return a single predict function suitable for KernelExplainer."""
        problem_type = getattr(
            self.automl_model, "_PROBLEM_TYPE", "classification"
        )
        model = self.automl_model

        if problem_type == "classification":
            # Try predict_proba first; fall back to predict
            if hasattr(model, "predict_proba"):
                try:
                    # Test it works
                    _ = model.predict_proba(
                        np.zeros((1, self.automl_model._profile_.n_features))
                    )
                    return model.predict_proba
                except Exception:
                    pass
            return lambda X: model.predict(X).astype(float)
        return model.predict
