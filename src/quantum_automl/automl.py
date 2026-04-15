"""
quantum_automl.automl
=====================

High-level AutoML entry points that orchestrate the full pipeline:
data analysis → preprocessing → model search → final fit.

Both :class:`QuantumAutoClassifier` and :class:`QuantumAutoRegressor` are
sklearn-compatible estimators (they inherit ``BaseEstimator`` and the
appropriate mixin), so they slot seamlessly into ``Pipeline``,
``GridSearchCV``, etc.

Low-end hardware tips
---------------------
* Set ``max_qubits=4`` and ``max_iter=50`` for a fast first run.
* Use ``search_strategy="grid"`` on machines without much RAM.
* Pass ``subsample=200`` to limit training data during search; the final
  model is then retrained on the full dataset.
* The default ``reps=1`` keeps circuit depth minimal — increase only when
  you have access to more compute.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from quantum_automl.data import QuantumDataAnalyzer, DataProfile, MAX_SAFE_QUBITS
from quantum_automl.search import QuantumModelSearch, SearchReport

logger = logging.getLogger(__name__)


# ── Shared base ──────────────────────────────────────────────────────────────

class _QuantumAutoBase(BaseEstimator):
    """
    Internal base class shared by the classifier and regressor.

    Parameters
    ----------
    max_qubits : int
        Maximum circuit width.  Recommended: 4–6 for laptops.
    max_iter : int
        VQC/VQR optimiser iterations per candidate model.
    reps : int
        Circuit layer repetitions (keep at 1 for speed).
    cv_folds : int
        Cross-validation folds used during search.
    search_strategy : str
        "grid" (exhaustive) or "optuna" (Bayesian, needs optuna installed).
    n_trials : int
        Optuna trial count (ignored for grid search).
    subsample : int | None
        Randomly subsample training data to this many rows during search.
        The winning model is then retrained on the full dataset.
    include_kernel_models : bool
        Include QSVC/QSVR (quantum kernel) models in the search space.
    early_stop_threshold : float | None
        Stop search early if a model exceeds this CV score.
    seed : int
        Global random seed.
    verbose : bool
        Print progress.
    """

    # Subclasses override these
    _PROBLEM_TYPE: str = ""

    def __init__(
        self,
        max_qubits: int = MAX_SAFE_QUBITS,
        max_iter: int = 100,
        reps: int = 1,
        cv_folds: int = 3,
        search_strategy: str = "grid",
        n_trials: int = 20,
        subsample: int | None = None,
        include_kernel_models: bool = True,
        early_stop_threshold: float | None = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.max_qubits = max_qubits
        self.max_iter = max_iter
        self.reps = reps
        self.cv_folds = cv_folds
        self.search_strategy = search_strategy
        self.n_trials = n_trials
        self.subsample = subsample
        self.include_kernel_models = include_kernel_models
        self.early_stop_threshold = early_stop_threshold
        self.seed = seed
        self.verbose = verbose

    # ── Internals ────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.info(msg)

    def _validate_input(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """Basic input validation — returns NumPy arrays."""
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y)
        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array, got shape {X_arr.shape}."
            )
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, "
                f"got X:{X_arr.shape[0]} y:{y_arr.shape[0]}."
            )
        if X_arr.shape[0] < 2 * self.cv_folds:
            raise ValueError(
                f"Dataset has only {X_arr.shape[0]} samples which is too few "
                f"for {self.cv_folds}-fold CV.  "
                f"Reduce cv_folds or provide more data."
            )
        return X_arr, y_arr

    def _subsample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.subsample is not None and X.shape[0] > self.subsample:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(X.shape[0], size=self.subsample, replace=False)
            self._log(
                f"  ⚡ Subsampling: {X.shape[0]} → {self.subsample} rows for search."
            )
            return X[idx], y[idx]
        return X, y

    def _run_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_qubits: int,
    ) -> SearchReport:
        """Dispatch to the correct search class."""
        common_kwargs = dict(
            problem_type=self._PROBLEM_TYPE,
            n_qubits=n_qubits,
            max_iter=self.max_iter,
            reps=self.reps,
            cv_folds=self.cv_folds,
            seed=self.seed,
            verbose=self.verbose,
        )

        if self.search_strategy == "optuna":
            try:
                from quantum_automl.search import OptunaQuantumSearch, _OPTUNA_AVAILABLE
                if not _OPTUNA_AVAILABLE:
                    raise ImportError
                searcher = OptunaQuantumSearch(
                    **common_kwargs,
                    n_trials=self.n_trials,
                )
                return searcher.search(X, y)
            except ImportError:
                warnings.warn(
                    "optuna not installed; falling back to grid search.",
                    UserWarning,
                    stacklevel=3,
                )

        # Default: grid search
        searcher = QuantumModelSearch(
            **common_kwargs,
            include_kernel_models=self.include_kernel_models,
            early_stop_threshold=self.early_stop_threshold,
        )
        return searcher.search(X, y)

    def _fit_best_model(
        self,
        best_model: Any,
        X_full: np.ndarray,
        y_full: np.ndarray,
    ) -> None:
        """Re-fit the winning model on the complete training set."""
        self._log("  🔧 Fitting best model on full training data …")
        t0 = time.time()
        best_model.fit(X_full, y_full)
        self._log(f"  ✓ Done in {time.time() - t0:.1f}s")

    # ── Shared fit logic ─────────────────────────────────────────────────────

    def _fit(self, X: Any, y: Any) -> "_QuantumAutoBase":
        t_total = time.time()
        X_arr, y_arr = self._validate_input(X, y)

        # ── 1. Data analysis ─────────────────────────────────────────────────
        self._log("\n⚛️  quantum-automl — starting AutoML pipeline")
        self._log("─" * 50)
        self._log("📊 Analysing dataset …")
        analyzer = QuantumDataAnalyzer(
            max_qubits=self.max_qubits, random_state=self.seed
        )
        profile: DataProfile = analyzer.analyze(X_arr, y_arr)
        if self.verbose:
            print(profile)

        # ── 2. Preprocessing (full dataset) ──────────────────────────────────
        self._log("⚙️  Preprocessing …")
        X_proc, y_proc = analyzer.preprocess(X_arr, y_arr, fit=True)
        self._analyzer_ = analyzer
        self._profile_ = profile

        # ── 3. Subsample for search ──────────────────────────────────────────
        X_search, y_search = self._subsample(X_proc, y_proc)

        # ── 4. Model search ──────────────────────────────────────────────────
        report = self._run_search(X_search, y_search, profile.recommended_qubits)
        self.search_report_ = report
        self.best_params_ = {
            "feature_map": report.best_result.spec.feature_map_name,
            "ansatz": report.best_result.spec.ansatz_name,
            "model_type": report.best_result.spec.model_type,
            "optimizer": report.best_result.spec.optimizer_name,
            "n_qubits": report.best_result.spec.n_qubits,
            "reps": report.best_result.spec.reps,
        }
        self.best_score_ = report.best_result.cv_score_mean

        # ── 5. Refit on full preprocessed data ───────────────────────────────
        best_model = report.best_result.model
        self._fit_best_model(best_model, X_proc, y_proc)
        self.best_model_ = best_model

        total_time = time.time() - t_total
        self._log(
            f"\n🎉 AutoML complete in {total_time:.1f}s\n"
            f"   Best model : {report.best_result.spec.name}\n"
            f"   CV score   : {self.best_score_:.4f}"
        )
        return self

    def _predict_raw(self, X: Any) -> np.ndarray:
        """Preprocess X and call predict on the best model."""
        check_is_fitted(self, "best_model_")
        X_arr = np.array(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_proc, _ = self._analyzer_.preprocess(X_arr, np.zeros(X_arr.shape[0]), fit=False)
        return self.best_model_.predict(X_proc)


# ── Classifier ───────────────────────────────────────────────────────────────

class QuantumAutoClassifier(_QuantumAutoBase, ClassifierMixin):
    """
    Automated quantum classifier.

    Finds, trains, and evaluates the best quantum classification model for
    your dataset with a single call to ``fit()``.

    Parameters
    ----------
    max_qubits : int, default 6
        Maximum circuit width.  Use 4 for fast iteration on low-end hardware.
    max_iter : int, default 100
        VQC optimiser iterations per candidate model.
    reps : int, default 1
        Circuit layer repetitions.
    cv_folds : int, default 3
        Cross-validation folds.
    search_strategy : str, default "grid"
        "grid" or "optuna".
    n_trials : int, default 20
        Optuna trial count (ignored for grid search).
    subsample : int | None, default None
        Sub-sample training rows for search speed.
    include_kernel_models : bool, default True
        Include QSVC models.
    early_stop_threshold : float | None, default None
        Stop when CV accuracy ≥ this value.
    seed : int, default 42
    verbose : bool, default True

    Attributes
    ----------
    best_model_ : sklearn-compatible estimator
        The fitted best model.
    best_params_ : dict
        Configuration of the best model.
    best_score_ : float
        Mean CV accuracy of the best model.
    search_report_ : SearchReport
        Full search history.
    classes_ : np.ndarray
        Unique class labels.

    Examples
    --------
    >>> from quantum_automl import QuantumAutoClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    >>> clf = QuantumAutoClassifier(max_qubits=4, max_iter=50, verbose=True)
    >>> clf.fit(X, y)
    >>> y_pred = clf.predict(X[:5])
    """

    _PROBLEM_TYPE = "classification"

    def fit(self, X: Any, y: Any) -> "QuantumAutoClassifier":
        """
        Run the full AutoML pipeline for classification.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        self._fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        raw = self._predict_raw(X)
        # Decode integer predictions back to original labels if needed
        return self._analyzer_.inverse_transform_labels(raw)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities (only available for VQC models).

        Parameters
        ----------
        X : array-like

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)

        Raises
        ------
        AttributeError
            If the best model does not support ``predict_proba``.
        """
        check_is_fitted(self, "best_model_")
        if not hasattr(self.best_model_, "predict_proba"):
            raise AttributeError(
                f"{type(self.best_model_).__name__} does not support "
                "`predict_proba`.  Use VQC-based models for probability outputs."
            )
        X_arr = np.array(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_proc, _ = self._analyzer_.preprocess(
            X_arr, np.zeros(X_arr.shape[0]), fit=False
        )
        return self.best_model_.predict_proba(X_proc)

    def score(self, X: Any, y: Any) -> float:
        """Return mean accuracy on (X, y)."""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.array(y)))


# ── Regressor ────────────────────────────────────────────────────────────────

class QuantumAutoRegressor(_QuantumAutoBase, RegressorMixin):
    """
    Automated quantum regressor.

    Finds, trains, and evaluates the best quantum regression model for
    your dataset with a single call to ``fit()``.

    Parameters
    ----------
    max_qubits : int, default 6
    max_iter : int, default 100
    reps : int, default 1
    cv_folds : int, default 3
    search_strategy : str, default "grid"
    n_trials : int, default 20
    subsample : int | None, default None
    include_kernel_models : bool, default True
    early_stop_threshold : float | None, default None
    seed : int, default 42
    verbose : bool, default True

    Attributes
    ----------
    best_model_ : sklearn-compatible estimator
    best_params_ : dict
    best_score_ : float
        Mean CV R² of the best model.
    search_report_ : SearchReport

    Examples
    --------
    >>> from quantum_automl import QuantumAutoRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=80, n_features=4, noise=0.1, random_state=0)
    >>> reg = QuantumAutoRegressor(max_qubits=4, max_iter=50, verbose=True)
    >>> reg.fit(X, y)
    >>> y_pred = reg.predict(X[:5])
    """

    _PROBLEM_TYPE = "regression"

    def fit(self, X: Any, y: Any) -> "QuantumAutoRegressor":
        """
        Run the full AutoML pipeline for regression.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        return self._fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict target values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        return self._predict_raw(X)

    def score(self, X: Any, y: Any) -> float:
        """Return R² score on (X, y)."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return float(r2_score(np.array(y), y_pred))
