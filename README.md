# ⚛️ quantum-automl

> **"You bring the data, we build the quantum model."**

[![PyPI version](https://img.shields.io/pypi/v/quantum-automl.svg)](https://pypi.org/project/quantum-automl/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An **end-to-end AutoML framework for Quantum Machine Learning** built on
[Qiskit](https://qiskit.org/) and
[qiskit-machine-learning](https://qiskit-community.github.io/qiskit-machine-learning/).

Most quantum ML libraries give you powerful building blocks but leave the hard
work to you: *which feature map? which ansatz? how many qubits? which
optimizer?* `quantum-automl` answers all of those questions automatically — the
same way `auto-sklearn` or TPOT do for classical ML.

---

## ✨ Highlights

| Feature | Details |
|---|---|
| **One-liner API** | `clf.fit(X, y)` — that's it |
| **Sklearn-compatible** | Plugs into `Pipeline`, `GridSearchCV`, etc. |
| **Low-end hardware first** | Runs on any modern laptop (no GPU needed) |
| **Smart search** | Grid search + optional Optuna Bayesian search |
| **Kernel & variational** | Searches VQC, VQR, QSVC, QSVR automatically |
| **Explainability** | SHAP-based feature importance on quantum models |
| **Unsupervised** | `QuantumAutoCluster` for clustering tasks |
| **Real hardware ready** | One call to `set_backend("ibm_brisbane")` |

---

## 🚀 Quick Install

```bash
pip install quantum-automl
```

With visualisation and XAI extras (matplotlib, SHAP):

```bash
pip install "quantum-automl[advanced]"
```

For development:

```bash
git clone https://github.com/quantum-automl/quantum-automl
cd quantum-automl
pip install -e ".[dev]"
```

---

## ⚡ Quick Start

```python
from quantum_automl import QuantumAutoClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Prepare data
X, y = make_classification(n_samples=120, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Fit — the library does everything else
clf = QuantumAutoClassifier(
    max_qubits=4,    # circuit width (4 is safe on any laptop)
    max_iter=50,     # optimizer iterations per candidate model
    verbose=True,    # show search progress
)
clf.fit(X_train, y_train)

# 3. Predict & evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Best model: {clf.best_params_}")
```

### Regression

```python
from quantum_automl import QuantumAutoRegressor
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
reg = QuantumAutoRegressor(max_qubits=4, max_iter=50, verbose=True)
reg.fit(X[:80], y[:80])
print(f"R²: {reg.score(X[80:], y[80:]):.4f}")
```

### Unsupervised Clustering

```python
from quantum_automl.cluster import QuantumAutoCluster
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=60, n_features=4, centers=3, random_state=0)
qc = QuantumAutoCluster(n_clusters=3, max_qubits=4, subsample=60)
labels = qc.fit_predict(X)
print(f"Silhouette: {qc.best_score_:.4f}")
```

### Explainability (requires `shap`)

```python
from quantum_automl.explainability import QuantumExplainer

explainer = QuantumExplainer(clf, background_samples=20)
explainer.fit(X_train)
shap_values = explainer.explain(X_test[:10])
print(explainer.feature_importance(shap_values))
explainer.summary_plot(shap_values, X_test[:10])
```

---

## 🔧 Configuration Reference

### `QuantumAutoClassifier` / `QuantumAutoRegressor`

| Parameter | Default | Description |
|---|---|---|
| `max_qubits` | `6` | Circuit width. **Use 4 on laptops.** |
| `max_iter` | `100` | Optimizer iterations per candidate. Lower = faster. |
| `reps` | `1` | Circuit layer repeats. Keep at 1 for speed. |
| `cv_folds` | `3` | Cross-validation folds. |
| `search_strategy` | `"grid"` | `"grid"` or `"optuna"`. |
| `n_trials` | `20` | Optuna trial count (ignored for grid). |
| `subsample` | `None` | Row limit during search (e.g. 100). |
| `include_kernel_models` | `True` | Include QSVC / QSVR. |
| `early_stop_threshold` | `None` | Stop when score ≥ threshold. |
| `seed` | `42` | Global random seed. |
| `verbose` | `True` | Show progress. |

### Fitted Attributes

After `.fit()`:

| Attribute | Type | Description |
|---|---|---|
| `best_model_` | estimator | The fitted best model |
| `best_params_` | dict | Configuration of the best model |
| `best_score_` | float | Mean CV score |
| `search_report_` | `SearchReport` | Full search history |

---

## 🖥️ Low-End Hardware Guide

`quantum-automl` is specifically designed to run on laptops and free cloud tiers
(Colab, Kaggle). Here is the recommended configuration:

```python
clf = QuantumAutoClassifier(
    max_qubits=4,           # keeps statevector RAM under 256 MB
    max_iter=50,            # reduces per-model training time
    reps=1,                 # shallow circuits = faster simulation
    cv_folds=3,             # 3-fold is plenty for search
    subsample=100,          # search on 100 rows, refit on all
    include_kernel_models=True,  # QSVC is fast on small datasets
    early_stop_threshold=0.90,   # stop if 90% accuracy found
)
```

**Memory requirements** for the statevector simulator:

| Qubits | RAM needed |
|---|---|
| 4 | ~0.5 KB |
| 6 | ~8 KB |
| 8 | ~128 KB |
| 10 | ~2 MB |
| 14 | ~512 MB |
| 16 | ~2 GB |
| 20 | ~32 GB |

```python
from quantum_automl.utils import check_hardware_compatibility
check_hardware_compatibility(n_qubits=8)   # prints safety warning
```

---

## 🏭 Real IBM Hardware

```python
from quantum_automl import QuantumAutoClassifier
from quantum_automl.utils import set_backend, get_runtime_options

# Connect to IBM Quantum
set_backend("ibm_brisbane", token="YOUR_IBM_TOKEN")

# Enable error mitigation (recommended for real hardware)
options = get_runtime_options(mitigation_level=2)

clf = QuantumAutoClassifier(max_qubits=4, max_iter=30, verbose=True)
clf.fit(X_train, y_train)
```

---

## 📐 How It Works

```
fit(X, y)
    │
    ▼
QuantumDataAnalyzer
  • Detect problem type (classification / regression)
  • Recommend qubit count
  • Scale features to [0, π]
  • PCA if n_features > max_qubits
    │
    ▼
QuantumModelSearch (grid or Optuna)
  • Candidate feature maps: ZFeatureMap, ZZFeatureMap, PauliFeatureMap
  • Candidate ansatzes:     RealAmplitudes, EfficientSU2, TwoLocal
  • Candidate kernel models: QSVC, QSVR
  • Cross-validate each with sklearn cross_val_score
    │
    ▼
Best model found → re-fit on full training data
    │
    ▼
predict(X_new)  →  preprocess → best_model_.predict → return
```

---

## 🔬 Comparison

| Feature | `quantum-automl` | `sQUlearn` | `qiskit-machine-learning` |
|---|---|---|---|
| AutoML (zero config) | ✅ | ❌ | ❌ |
| Sklearn API | ✅ | ✅ | ✅ |
| Auto feature map search | ✅ | ❌ | ❌ |
| Auto ansatz search | ✅ | ❌ | ❌ |
| Optuna Bayesian search | ✅ | ❌ | ❌ |
| SHAP explainability | ✅ | ❌ | ❌ |
| Clustering | ✅ | ❌ | ❌ |
| Low-end hardware tuning | ✅ | ⚠️ | ⚠️ |
| IBM hardware integration | ✅ | ✅ | ✅ |

---

## 🤝 Contributing

Contributions are warmly welcome!

1. **Fork** the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev deps: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/`
5. Run linting: `ruff check src/ && black --check src/`
6. Open a **Pull Request** describing your changes.

### Ideas for contributions

- Additional ansatz templates (e.g. hardware-efficient circuits)
- Support for `qiskit-ibm-runtime` V2 primitives
- Async / parallel model evaluation
- Integration with Weights & Biases for search tracking
- Noise-aware circuit optimisation with `mitiq`
- Quantum Neural Network (QNN) search

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 📚 References

- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
- [Variational Quantum Classifier (Havlíček et al. 2019)](https://www.nature.com/articles/s41586-019-0980-2)
- [Quantum Kernel Methods (Schuld & Killoran 2019)](https://arxiv.org/abs/1803.07128)
- [Optuna](https://optuna.org/)
- [SHAP](https://shap.readthedocs.io/)
