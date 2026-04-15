"""
quantum_automl.search
=====================

Model search and hyperparameter optimisation.

Two strategies are provided:

1. :class:`QuantumModelSearch` — exhaustive grid search over the
   pre-defined catalogue of feature maps × ansatzes.  Suitable for small
   searches on low-end hardware.

2. :class:`OptunaQuantumSearch` — Bayesian optimisation via Optuna.
   Smarter about exploration/exploitation; typically finds good configs
   with fewer model evaluations.

Low-end hardware notes
----------------------
* ``n_jobs=1`` is used throughout — parallelism requires spawning new
  processes, each of which would reload the full Qiskit stack.  On a
  laptop the overhead usually exceeds any speed gain.
* Cross-validation fold count defaults to 3.  Increase to 5 for more
  reliable scores once you have GPU / remote hardware.
* An ``early_stop_threshold`` can abort search when a model exceeds a
  user-defined score — useful on time-constrained machines.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from quantum_automl.models import QuantumModelFactory, ModelSpec

logger = logging.getLogger(__name__)


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Stores the outcome of a single model evaluation."""

    spec: ModelSpec
    model: Any
    cv_score_mean: float
    cv_score_std: float
    wall_time_s: float
    error: str | None = None


@dataclass
class SearchReport:
    """Aggregated report from a full search run."""

    best_result: SearchResult
    all_results: list[SearchResult] = field(default_factory=list)
    total_time_s: float = 0.0
    n_evaluated: int = 0
    n_failed: int = 0
    search_strategy: str = "grid"

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            "── SearchReport ────────────────────────────",
            f"  Strategy        : {self.search_strategy}",
            f"  Models evaluated: {self.n_evaluated}",
            f"  Models failed   : {self.n_failed}",
            f"  Total time      : {self.total_time_s:.1f}s",
            "  Best model      :",
            f"    Name          : {self.best_result.spec.name}",
            f"    CV score      : {self.best_result.cv_score_mean:.4f} "
            f"± {self.best_result.cv_score_std:.4f}",
            f"    Time          : {self.best_result.wall_time_s:.1f}s",
            "─────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ── Grid search ──────────────────────────────────────────────────────────────

class QuantumModelSearch:
    """
    Exhaustive search over the quantum model catalogue.

    Parameters
    ----------
    problem_type : str
        "classification" or "regression"
    n_qubits : int
        Number of qubits for all candidate circuits.
    max_iter : int
        Maximum VQC/VQR optimiser iterations per model.
    reps : int
        Circuit layer repetitions (keep 1 for low-end hardware).
    cv_folds : int
        Number of cross-validation folds.
    scoring : str | None
        Sklearn scoring string.  ``None`` uses accuracy for classification
        and r2 for regression.
    early_stop_threshold : float | None
        If a model achieves ``>= early_stop_threshold`` CV score, stop
        immediately and return it as the best.
    seed : int
        Random seed.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        problem_type: str = "classification",
        n_qubits: int = 4,
        max_iter: int = 100,
        reps: int = 1,
        cv_folds: int = 3,
        scoring: str | None = None,
        early_stop_threshold: float | None = None,
        include_kernel_models: bool = True,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.problem_type = problem_type
        self.n_qubits = n_qubits
        self.max_iter = max_iter
        self.reps = reps
        self.cv_folds = cv_folds
        self.scoring = scoring or (
            "accuracy" if problem_type == "classification" else "r2"
        )
        self.early_stop_threshold = early_stop_threshold
        self.include_kernel_models = include_kernel_models
        self.seed = seed
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.info(msg)

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_candidates: int | None = None,
    ) -> SearchReport:
        """
        Run the grid search and return a :class:`SearchReport`.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_qubits)
            Pre-processed feature matrix.
        y : np.ndarray, shape (n_samples,)
            Pre-processed target vector.
        max_candidates : int | None
            Limit the number of models evaluated (useful for quick tests).

        Returns
        -------
        SearchReport
        """
        factory = QuantumModelFactory(
            n_qubits=self.n_qubits,
            max_iter=self.max_iter,
            reps=self.reps,
            seed=self.seed,
        )

        if self.problem_type == "classification":
            candidates = factory.candidate_classifiers(
                include_kernel=self.include_kernel_models
            )
        else:
            candidates = factory.candidate_regressors(
                include_kernel=self.include_kernel_models
            )

        if max_candidates is not None:
            candidates = candidates[:max_candidates]

        self._log(
            f"\n🔍 Starting quantum model search "
            f"({len(candidates)} candidates, {self.cv_folds}-fold CV, "
            f"scoring={self.scoring})"
        )

        cv = (
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            if self.problem_type == "classification"
            else KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        )

        all_results: list[SearchResult] = []
        best_result: SearchResult | None = None
        search_start = time.time()

        for i, (spec, model) in enumerate(candidates, 1):
            self._log(
                f"  [{i:02d}/{len(candidates):02d}] Evaluating {spec.name} …"
            )
            t0 = time.time()
            result = self._evaluate(model, spec, X, y, cv)
            elapsed = time.time() - t0
            result.wall_time_s = elapsed

            all_results.append(result)

            if result.error is None:
                score_str = (
                    f"{result.cv_score_mean:.4f} ± {result.cv_score_std:.4f}"
                )
                self._log(
                    f"         ✓ CV {self.scoring}: {score_str}  "
                    f"({elapsed:.1f}s)"
                )
                if best_result is None or result.cv_score_mean > best_result.cv_score_mean:
                    best_result = result

                if (
                    self.early_stop_threshold is not None
                    and result.cv_score_mean >= self.early_stop_threshold
                ):
                    self._log(
                        f"  ⚡ Early stop: score {result.cv_score_mean:.4f} "
                        f">= threshold {self.early_stop_threshold}"
                    )
                    break
            else:
                self._log(f"         ✗ Failed: {result.error}")

        total_time = time.time() - search_start
        n_failed = sum(1 for r in all_results if r.error is not None)

        if best_result is None:
            raise RuntimeError(
                "All candidate models failed during search. "
                "Check logs for details.  "
                "Try reducing max_qubits or checking your data dimensions."
            )

        self._log(
            f"\n✅ Search complete in {total_time:.1f}s\n"
            f"   Best: {best_result.spec.name} — "
            f"{self.scoring}={best_result.cv_score_mean:.4f}"
        )

        report = SearchReport(
            best_result=best_result,
            all_results=all_results,
            total_time_s=total_time,
            n_evaluated=len(all_results),
            n_failed=n_failed,
            search_strategy="grid",
        )
        return report

    def _evaluate(
        self,
        model: Any,
        spec: ModelSpec,
        X: np.ndarray,
        y: np.ndarray,
        cv: Any,
    ) -> SearchResult:
        """Run cross-validation for a single model, catching all exceptions."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                factory = QuantumModelFactory(n_qubits=self.n_qubits, max_iter=self.max_iter,reps=self.reps, seed = self.seed,)
                scores = []
                for train_idx, val_idx in cv.split(X,y):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    fm_cls = QuantumModelFactory.FEATURE_MAP_CATALOGUE[spec.feature_map_name]
                    fm = fm_cls(feature_dimension=self.n_qubits, reps=self.reps)
                    if spec.model_type == "vqc":
                        az_cls = QuantumModelFactory.ANSATZ_CATALOGUE[spec.ansatz_name]
                        az = az_cls(num_qubits=self.n_qubits, reps=self.reps, entanglement = "linear")
                        fold_model = factory.build_vqc(fm, az, spec.optimizer_name)
                    elif spec.model_type =="qsvc":
                        fold_model = factory.build_qsvc(fm)
                    fold_model.fit(X_tr, y_tr)
                    scores.append(fold_model.score(X_val, y_val))
                scores = np.array(scores)
                    
            return SearchResult(
                spec=spec,
                model=model,
                cv_score_mean=float(np.mean(scores)),
                cv_score_std=float(np.std(scores)),
                wall_time_s=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model %s raised: %s", spec.name, exc)
            return SearchResult(
                spec=spec,
                model=model,
                cv_score_mean=-np.inf,
                cv_score_std=0.0,
                wall_time_s=0.0,
                error=str(exc),
            )


# ── Optuna search ────────────────────────────────────────────────────────────

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    class OptunaQuantumSearch:
        """
        Bayesian hyperparameter search using Optuna.

        Searches over:
        * feature map type
        * ansatz type
        * number of circuit reps (1, 2)
        * optimiser (COBYLA, SPSA)
        * kernel SVM regularisation C

        Parameters
        ----------
        problem_type : str
        n_qubits : int
        max_iter : int
            Max VQC/VQR optimiser iterations.
        cv_folds : int
        n_trials : int
            Number of Optuna trials.
        scoring : str | None
        seed : int
        verbose : bool
        """

        def __init__(
            self,
            problem_type: str = "classification",
            n_qubits: int = 4,
            max_iter: int = 100,
            cv_folds: int = 3,
            n_trials: int = 20,
            scoring: str | None = None,
            seed: int = 42,
            verbose: bool = False,
        ) -> None:
            self.problem_type = problem_type
            self.n_qubits = n_qubits
            self.max_iter = max_iter
            self.cv_folds = cv_folds
            self.n_trials = n_trials
            self.scoring = scoring or (
                "accuracy" if problem_type == "classification" else "r2"
            )
            self.seed = seed
            self.verbose = verbose

        def _log(self, msg: str) -> None:
            if self.verbose:
                print(msg)
            logger.info(msg)

        def search(
            self,
            X: np.ndarray,
            y: np.ndarray,
        ) -> SearchReport:
            """
            Run Optuna Bayesian search.

            Parameters
            ----------
            X : np.ndarray
            y : np.ndarray

            Returns
            -------
            SearchReport
            """
            cv = (
                StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.seed
                )
                if self.problem_type == "classification"
                else KFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.seed
                )
            )

            all_results: list[SearchResult] = []
            search_start = time.time()

            def objective(trial: optuna.Trial) -> float:
                fm_name = trial.suggest_categorical(
                    "feature_map",
                    ["ZFeatureMap", "ZZFeatureMap", "PauliFeatureMap"],
                )
                reps = trial.suggest_int("reps", 1, 2)
                model_type = trial.suggest_categorical(
                    "model_type",
                    ["vqc", "qsvc"] if self.problem_type == "classification"
                    else ["vqr", "qsvr"],
                )

                factory = QuantumModelFactory(
                    n_qubits=self.n_qubits,
                    max_iter=self.max_iter,
                    reps=reps,
                    seed=self.seed,
                )
                fm_instance = factory.get_feature_maps()
                fm_map = {name: inst for name, inst in fm_instance}
                fm = fm_map[fm_name]

                try:
                    if model_type in ("vqc", "vqr"):
                        az_name = trial.suggest_categorical(
                            "ansatz",
                            ["RealAmplitudes", "EfficientSU2", "TwoLocal"],
                        )
                        opt_name = trial.suggest_categorical(
                            "optimizer", ["COBYLA", "SPSA"]
                        )
                        az_map = {name: inst for name, inst in factory.get_ansatzes()}
                        az = az_map[az_name]

                        if model_type == "vqc":
                            model = factory.build_vqc(fm, az, opt_name)
                        else:
                            model = factory.build_vqr(fm, az, opt_name)

                        spec = ModelSpec(
                            name=f"{model_type.upper()}_{fm_name}_{az_name}_{opt_name}",
                            feature_map_name=fm_name,
                            ansatz_name=az_name,
                            model_type=model_type,
                            n_qubits=self.n_qubits,
                            reps=reps,
                            optimizer_name=opt_name,
                        )
                    else:
                        C = trial.suggest_float("C", 0.01, 100.0, log=True)
                        if model_type == "qsvc":
                            model = factory.build_qsvc(fm, C=C)
                        else:
                            model = factory.build_qsvr(fm, C=C)
                        spec = ModelSpec(
                            name=f"{model_type.upper()}_{fm_name}_C{C:.2f}",
                            feature_map_name=fm_name,
                            ansatz_name=None,
                            model_type=model_type,
                            n_qubits=self.n_qubits,
                            reps=reps,
                            optimizer_name="N/A",
                        )

                    t0 = time.time()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        scores = cross_val_score(
                            model, X, y, cv=cv, scoring=self.scoring, n_jobs=1
                        )
                    elapsed = time.time() - t0
                    mean_score = float(np.mean(scores))

                    result = SearchResult(
                        spec=spec,
                        model=model,
                        cv_score_mean=mean_score,
                        cv_score_std=float(np.std(scores)),
                        wall_time_s=elapsed,
                    )
                    all_results.append(result)
                    self._log(
                        f"  Trial {trial.number:03d}: {spec.name} → "
                        f"{self.scoring}={mean_score:.4f} ({elapsed:.1f}s)"
                    )
                    return mean_score

                except Exception as exc:  # noqa: BLE001
                    logger.warning("Trial %d failed: %s", trial.number, exc)
                    return -np.inf

            self._log(
                f"\n🔬 Starting Optuna search "
                f"({self.n_trials} trials, {self.cv_folds}-fold CV)"
            )
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            study = optuna.create_study(
                direction="maximize", sampler=sampler
            )
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            total_time = time.time() - search_start
            successful = [r for r in all_results if r.error is None]

            if not successful:
                raise RuntimeError("All Optuna trials failed.")

            best_result = max(successful, key=lambda r: r.cv_score_mean)
            self._log(
                f"\n✅ Optuna search complete in {total_time:.1f}s\n"
                f"   Best: {best_result.spec.name} — "
                f"{self.scoring}={best_result.cv_score_mean:.4f}"
            )

            return SearchReport(
                best_result=best_result,
                all_results=all_results,
                total_time_s=total_time,
                n_evaluated=len(all_results),
                n_failed=len(all_results) - len(successful),
                search_strategy="optuna",
            )

    _OPTUNA_AVAILABLE = True

except ImportError:
    _OPTUNA_AVAILABLE = False
    logger.debug("optuna not found; OptunaQuantumSearch is unavailable.")
