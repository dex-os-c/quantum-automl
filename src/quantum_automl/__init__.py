"""
quantum-automl
==============

Automated Machine Learning for Quantum Computing.

Philosophy: **"You bring the data, we build the quantum model."**

Designed to run on low-end hardware (no GPU required) while remaining
compatible with real IBM Quantum hardware for production upgrades.

Public API
----------
>>> from quantum_automl import QuantumAutoClassifier, QuantumAutoRegressor
>>> clf = QuantumAutoClassifier(max_qubits=4, max_iter=10, verbose=True)
>>> clf.fit(X_train, y_train)
>>> y_pred = clf.predict(X_test)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "quantum-automl contributors"
__license__ = "MIT"

# ── Public classes ──────────────────────────────────────────────────────────
from quantum_automl.automl import QuantumAutoClassifier, QuantumAutoRegressor  # noqa: E402

# ── Advanced / optional ─────────────────────────────────────────────────────
try:
    from quantum_automl.cluster import QuantumAutoCluster  # noqa: F401
    _cluster_available = True
except ImportError:
    _cluster_available = False

try:
    from quantum_automl.explainability import QuantumExplainer  # noqa: F401
    _xai_available = True
except ImportError:
    _xai_available = False

__all__ = [
    "__version__",
    "QuantumAutoClassifier",
    "QuantumAutoRegressor",
]
if _cluster_available:
    __all__.append("QuantumAutoCluster")
if _xai_available:
    __all__.append("QuantumExplainer")
