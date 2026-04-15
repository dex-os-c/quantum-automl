"""
quantum_automl.cluster
=======================

Unsupervised quantum clustering via quantum kernel k-means.

The algorithm:
1. Build a quantum kernel matrix  K[i,j] = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²
   using one of the feature map candidates.
2. Run sklearn's ``SpectralClustering`` (kernel="precomputed") on K.
3. Repeat for each feature map and pick the one with the best
   silhouette score.

Low-end hardware notes
-----------------------
* Kernel matrix computation is O(n²) — expensive for large datasets.
  Always set ``subsample`` to ≤ 100 on laptops.
* ``max_qubits=4`` is recommended.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from quantum_automl.data import QuantumDataAnalyzer, MAX_SAFE_QUBITS
from quantum_automl.models import QuantumModelFactory

logger = logging.getLogger(__name__)


class QuantumAutoCluster(BaseEstimator, ClusterMixin):
    """
    Automated quantum kernel clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_qubits : int, default 4
        Qubit budget.  Keep at 4 for low-end hardware.
    reps : int, default 1
        Circuit repetitions.
    subsample : int | None, default 100
        Limit dataset size for kernel computation.
    seed : int, default 42
    verbose : bool, default True

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster label for each training sample.
    best_feature_map_name_ : str
    best_score_ : float
        Silhouette score of the best clustering.

    Examples
    --------
    >>> from quantum_automl.cluster import QuantumAutoCluster
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=60, n_features=4, centers=3, random_state=0)
    >>> qc = QuantumAutoCluster(n_clusters=3, max_qubits=4, subsample=60)
    >>> qc.fit(X)
    >>> print(qc.labels_)
    """

    def __init__(
        self,
        n_clusters: int = 2,
        max_qubits: int = MAX_SAFE_QUBITS,
        reps: int = 1,
        subsample: int | None = 100,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_qubits = max_qubits
        self.reps = reps
        self.subsample = subsample
        self.seed = seed
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.info(msg)

    def fit(self, X: Any, y: Any = None) -> "QuantumAutoCluster":
        """
        Fit quantum kernel clustering.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        X_arr = np.array(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self._log("\n⚛️  QuantumAutoCluster — fitting")
        self._log(f"   n_samples={X_arr.shape[0]}, n_features={X_arr.shape[1]}")

        # Subsample for kernel computation
        n_orig = X_arr.shape[0]
        if self.subsample is not None and n_orig > self.subsample:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n_orig, self.subsample, replace=False)
            X_arr = X_arr[idx]
            self._sample_indices_ = idx
            self._log(f"   Subsampled: {n_orig} → {len(idx)} rows")
        else:
            self._sample_indices_ = np.arange(n_orig)

        # Preprocess
        analyzer = QuantumDataAnalyzer(max_qubits=self.max_qubits, random_state=self.seed)
        analyzer._profile_ = type("_P", (), {
            "needs_pca": X_arr.shape[1] > self.max_qubits,
            "recommended_qubits": analyzer.recommend_qubits(X_arr.shape[1]),
            "n_features": X_arr.shape[1],
            "problem_type": "regression",  # dummy for preprocess
        })()
        scaler = MinMaxScaler(feature_range=(0, float(np.pi)))
        X_proc = scaler.fit_transform(X_arr)
        self._scaler_ = scaler

        n_qubits = analyzer.recommend_qubits(X_arr.shape[1])
        if X_arr.shape[1] > n_qubits:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_qubits, random_state=self.seed)
            X_proc = pca.fit_transform(X_proc)
            X_proc = MinMaxScaler(feature_range=(0, float(np.pi))).fit_transform(X_proc)
            self._pca_ = pca
        else:
            self._pca_ = None

        factory = QuantumModelFactory(n_qubits=n_qubits, reps=self.reps, seed=self.seed)
        feature_maps = factory.get_feature_maps()

        best_score = -1.0
        best_labels: np.ndarray | None = None
        best_fm_name = ""

        for fm_name, fm in feature_maps:
            self._log(f"   Trying feature map: {fm_name} …")
            try:
                K = self._compute_kernel_matrix(fm, X_proc)
                # Clip to [0, 1] for numerical stability
                K = np.clip(K, 0, 1)
                # Make symmetric and add small diagonal for PSD
                K = 0.5 * (K + K.T)
                np.fill_diagonal(K, 1.0)

                clusterer = SpectralClustering(
                    n_clusters=self.n_clusters,
                    affinity="precomputed",
                    random_state=self.seed,
                    assign_labels="kmeans",
                )
                labels = clusterer.fit_predict(K)

                if len(np.unique(labels)) < 2:
                    self._log(f"     ✗ Only 1 cluster found — skipping.")
                    continue

                score = silhouette_score(X_proc, labels, metric="cosine")
                self._log(f"     ✓ Silhouette = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_fm_name = fm_name

            except Exception as exc:  # noqa: BLE001
                self._log(f"     ✗ Failed: {exc}")
                logger.warning("Feature map %s failed: %s", fm_name, exc)

        if best_labels is None:
            raise RuntimeError(
                "All feature maps failed during clustering.  "
                "Try reducing max_qubits or subsample."
            )

        self.labels_ = best_labels
        self.best_feature_map_name_ = best_fm_name
        self.best_score_ = best_score
        self._n_qubits_ = n_qubits

        self._log(
            f"\n✅ Best clustering: {best_fm_name}  "
            f"silhouette={best_score:.4f}"
        )
        return self

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X, y)
        return self.labels_

    def _compute_kernel_matrix(
        self, feature_map: Any, X: np.ndarray
    ) -> np.ndarray:
        """
        Compute the n×n quantum kernel matrix using FidelityQuantumKernel.
        """
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        sampler = StatevectorSampler()
        kernel = FidelityQuantumKernel(
            feature_map=feature_map,
            sampler=sampler,
        )
        return kernel.evaluate(x_vec=X)
