"""
quantum_automl.data
===================

Data analysis and preprocessing utilities for quantum machine learning.

Key design decisions for **low-end hardware**
---------------------------------------------
* PCA dimensionality reduction is applied whenever ``n_features > max_qubits``
  so the quantum circuit never grows beyond what a CPU simulator can handle.
* All feature scaling uses ``MinMaxScaler`` to map inputs to [0, π] — the
  natural angular range for Pauli rotation gates.
* The qubit recommendation heuristic caps at ``MAX_SAFE_QUBITS = 6`` by
  default; this keeps statevector simulation memory under ~512 MB.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
# Statevector simulator RAM ≈ 16 * 2^n bytes.
# n=6  → 1 KB (trivial), n=10 → 16 KB, n=16 → 1 MB, n=20 → 16 MB, n=26 → 1 GB
# We cap at 6 for low-end safety; users can override via max_qubits.
MAX_SAFE_QUBITS: int = 6
MIN_QUBITS: int = 2


@dataclass
class DataProfile:
    """Structured summary returned by :meth:`QuantumDataAnalyzer.analyze`."""

    problem_type: str  # "classification" | "regression"
    n_samples: int
    n_features: int
    n_classes: int | None  # None for regression
    class_labels: list[Any] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    has_missing: bool = False
    recommended_qubits: int = MIN_QUBITS
    needs_pca: bool = False
    target_dtype: str = "float64"
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            "── DataProfile ─────────────────────────────",
            f"  Problem type      : {self.problem_type}",
            f"  Samples / Features: {self.n_samples} / {self.n_features}",
        ]
        if self.n_classes is not None:
            lines.append(f"  Classes           : {self.n_classes} → {self.class_labels}")
        lines += [
            f"  Recommended qubits: {self.recommended_qubits}",
            f"  PCA needed        : {self.needs_pca}",
            f"  Missing values    : {self.has_missing}",
        ]
        if self.notes:
            lines.append("  Notes:")
            for note in self.notes:
                lines.append(f"    • {note}")
        lines.append("─────────────────────────────────────────────")
        return "\n".join(lines)


class QuantumDataAnalyzer:
    """
    Analyses a dataset and prepares it for quantum model search.

    Parameters
    ----------
    max_qubits : int, optional
        Hard cap on the number of qubits used.  Defaults to
        :data:`MAX_SAFE_QUBITS` (6) for low-end hardware.
    scale_range : tuple[float, float], optional
        Target range after MinMaxScaling.  Defaults to ``(0, np.pi)`` which
        maps each feature to a full rotation period on the Bloch sphere.
    random_state : int, optional
        Seed for reproducible PCA / train-test splits.
    """

    def __init__(
        self,
        max_qubits: int = MAX_SAFE_QUBITS,
        scale_range: tuple[float, float] = (0.0, float(np.pi)),
        random_state: int = 42,
    ) -> None:
        if max_qubits < MIN_QUBITS:
            raise ValueError(
                f"max_qubits must be >= {MIN_QUBITS}; got {max_qubits}."
            )
        self.max_qubits = max_qubits
        self.scale_range = scale_range
        self.random_state = random_state

        # Set after fit
        self._scaler: MinMaxScaler | None = None
        self._pca: PCA | None = None
        self._label_encoder: LabelEncoder | None = None
        self._profile: DataProfile | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze(self, X: Any, y: Any) -> DataProfile:
        """
        Inspect the dataset and return a :class:`DataProfile`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        DataProfile
        """
        X_arr, y_arr = self._to_numpy(X, y)
        notes: list[str] = []

        # ── Missing values ───────────────────────────────────────────────────
        has_missing = bool(np.isnan(X_arr).any() or np.isnan(y_arr).any())
        if has_missing:
            notes.append(
                "Missing values detected — they will be imputed with column means."
            )

        # ── Problem type ─────────────────────────────────────────────────────
        problem_type, n_classes, class_labels = self._infer_problem_type(y_arr)
        if problem_type == "classification" and n_classes is not None and n_classes > 10:
            warnings.warn(
                f"Detected {n_classes} classes. "
                "Quantum classifiers are most effective for ≤10 classes.",
                UserWarning,
                stacklevel=2,
            )
            notes.append(
                f"Large number of classes ({n_classes}). "
                "Consider reducing via label grouping."
            )

        # ── Qubits ───────────────────────────────────────────────────────────
        n_features = X_arr.shape[1]
        rec_qubits = self.recommend_qubits(n_features)
        needs_pca = n_features > self.max_qubits

        if needs_pca:
            notes.append(
                f"n_features ({n_features}) > max_qubits ({self.max_qubits}). "
                f"PCA will reduce to {rec_qubits} components."
            )

        # ── Large dataset warning ────────────────────────────────────────────
        n_samples = X_arr.shape[0]
        if n_samples > 500:
            notes.append(
                f"Dataset has {n_samples} samples. "
                "Quantum simulators are slow on large datasets. "
                "Consider passing a random subsample for AutoML search, "
                "then fine-tuning on the full set."
            )

        feature_names: list[str] = []
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)

        self._profile = DataProfile(
            problem_type=problem_type,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_labels=list(class_labels) if class_labels is not None else [],
            feature_names=feature_names,
            has_missing=has_missing,
            recommended_qubits=rec_qubits,
            needs_pca=needs_pca,
            target_dtype=str(y_arr.dtype),
            notes=notes,
        )
        return self._profile

    def recommend_qubits(self, n_features: int) -> int:
        """
        Return the number of qubits best suited for ``n_features`` inputs.

        Heuristic
        ---------
        * Use exactly n_features qubits if it fits within max_qubits.
        * Otherwise cap at max_qubits (PCA will handle the reduction).
        * Always enforce the minimum of MIN_QUBITS (2).

        Parameters
        ----------
        n_features : int

        Returns
        -------
        int
        """
        n_qubits = min(n_features, self.max_qubits)
        n_qubits = max(n_qubits, MIN_QUBITS)
        # Round down to even number for RealAmplitudes / EfficientSU2
        # which work best with even qubit counts (CX gate pairs)
        if n_qubits > 2 and n_qubits % 2 != 0:
            n_qubits -= 1
        logger.debug(
            "recommend_qubits: n_features=%d → n_qubits=%d", n_features, n_qubits
        )
        return n_qubits

    def preprocess(
        self,
        X: Any,
        y: Any,
        fit: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features and targets for quantum circuits.

        Steps
        -----
        1. Convert to NumPy.
        2. Impute missing values (mean imputation).
        3. MinMaxScale features to ``scale_range``.
        4. (Optional) PCA if n_features > max_qubits.
        5. For classification: LabelEncode targets to contiguous integers.

        Parameters
        ----------
        X : array-like
        y : array-like
        fit : bool
            If True, fit scalers/PCA on this data (training).
            If False, transform only (inference).

        Returns
        -------
        X_proc : np.ndarray, shape (n_samples, n_qubits)
        y_proc : np.ndarray, shape (n_samples,)
        """
        X_arr, y_arr = self._to_numpy(X, y)

        # ── 1. Missing value imputation ──────────────────────────────────────
        if np.isnan(X_arr).any():
            col_means = np.nanmean(X_arr, axis=0)
            inds = np.where(np.isnan(X_arr))
            X_arr[inds] = np.take(col_means, inds[1])

        # ── 2. Feature scaling ───────────────────────────────────────────────
        if fit:
            self._scaler = MinMaxScaler(feature_range=self.scale_range)
            X_arr = self._scaler.fit_transform(X_arr)
        else:
            if self._scaler is None:
                raise RuntimeError("Call preprocess(fit=True) before inference.")
            X_arr = self._scaler.transform(X_arr)

        # ── 3. PCA (only when needed) ────────────────────────────────────────
        if self._profile is not None and self._profile.needs_pca:
            n_components = self._profile.recommended_qubits
            if fit:
                self._pca = PCA(
                    n_components=n_components, random_state=self.random_state
                )
                X_arr = self._pca.fit_transform(X_arr)
                explained = self._pca.explained_variance_ratio_.sum()
                logger.info(
                    "PCA: %d → %d features, %.1f%% variance retained.",
                    self._profile.n_features,
                    n_components,
                    explained * 100,
                )
                if explained < 0.80:
                    warnings.warn(
                        f"PCA retains only {explained:.1%} of variance. "
                        "Consider increasing max_qubits for better accuracy.",
                        UserWarning,
                        stacklevel=2,
                    )
                # Re-scale PCA output to scale_range
                self._pca_scaler = MinMaxScaler(feature_range=self.scale_range)
                X_arr = self._pca_scaler.fit_transform(X_arr)
            else:
                if self._pca is None:
                    raise RuntimeError("PCA not fitted. Call preprocess(fit=True) first.")
                X_arr = self._pca.transform(X_arr)
                X_arr = self._pca_scaler.transform(X_arr)

        # ── 4. Target encoding ───────────────────────────────────────────────
        if self._profile is not None and self._profile.problem_type == "classification":
            if fit:
                self._label_encoder = LabelEncoder()
                y_arr = self._label_encoder.fit_transform(y_arr)
            else:
                if self._label_encoder is None:
                    raise RuntimeError(
                        "LabelEncoder not fitted. Call preprocess(fit=True) first."
                    )
                y_arr = self._label_encoder.transform(y_arr)

        return X_arr.astype(np.float64), y_arr

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode integer class labels back to original format."""
        if self._label_encoder is None:
            return y_encoded
        
        if np.isscalar(y_encoded)or (isinstance(y_encoded, np.ndarray)and y_encoded.ndim == 0):
            y_encoded = np.array([y_encoded])
        return self._label_encoder.inverse_transform(y_encoded.astype(int))

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """Convert pandas / list inputs to float64 NumPy arrays."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.array(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = np.array(y)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        return X_arr, y_arr

    @staticmethod
    def _infer_problem_type(
        y: np.ndarray,
    ) -> tuple[str, int | None, np.ndarray | None]:
        """
        Heuristically determine whether ``y`` represents a classification or
        regression target.

        Returns
        -------
        problem_type : str
        n_classes : int | None
        class_labels : np.ndarray | None
        """
        # Integer / boolean arrays with <= 20 unique values → classification
        is_int_like = np.issubdtype(y.dtype, np.integer) or np.issubdtype(
            y.dtype, np.bool_
        )
        unique_vals = np.unique(y)
        n_unique = len(unique_vals)

        if is_int_like or (n_unique <= 20 and n_unique / len(y) < 0.05):
            return "classification", n_unique, unique_vals
        return "regression", None, None
