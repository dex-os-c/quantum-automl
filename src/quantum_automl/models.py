"""
quantum_automl.models
=====================

Quantum model factory — creates Qiskit feature maps, ansatzes, and
assembled VQC / VQR / QSVC models.

Low-end hardware optimisations
-------------------------------
* ``reps=1`` is used by default on all circuits.  This keeps gate depth
  minimal, reducing simulation time exponentially.
* ``StatevectorSampler`` / ``StatevectorEstimator`` from ``qiskit.primitives``
  are used as the default backends — they run entirely in NumPy (no Aer
  required) and are typically 10-100× faster for small circuits.
* ``max_iter`` for COBYLA/SPSA is capped at a user-configurable value so
  training stays tractable on a laptop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

# Qiskit circuits
from qiskit.circuit.library import (
    EfficientSU2,
    PauliFeatureMap,
    RealAmplitudes,
    TwoLocal,
    ZFeatureMap,
    ZZFeatureMap,
)

# Qiskit primitives (no Aer needed — runs on pure NumPy)
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

# Qiskit algorithms
from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B

# Qiskit Machine Learning
try:
    from qiskit_machine_learning.algorithms import VQC, VQR
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC, QSVR
    _QML_AVAILABLE = True
except ImportError as _err:
    _QML_AVAILABLE = False
    _QML_IMPORT_ERROR = _err

logger = logging.getLogger(__name__)

# ── Named tuples / dataclasses ───────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Describes a single candidate model configuration."""

    name: str
    feature_map_name: str
    ansatz_name: str | None  # None for kernel methods
    model_type: str           # "vqc" | "vqr" | "qsvc" | "qsvr"
    n_qubits: int
    reps: int
    optimizer_name: str


# ── Factory ──────────────────────────────────────────────────────────────────

class QuantumModelFactory:
    """
    Constructs Qiskit machine learning models from named components.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for all circuits.
    max_iter : int
        Maximum number of optimiser iterations for VQC / VQR.
    reps : int
        Number of circuit repetitions (layers).  Keep at 1 for low-end
        hardware; increase to 2-3 for better expressibility on real devices.
    seed : int
        Random seed for optimisers and initial parameters.
    """

    # Catalogues ─────────────────────────────────────────────────────────────
    FEATURE_MAP_CATALOGUE: dict[str, type] = {
        "ZZFeatureMap": ZZFeatureMap,
        "ZFeatureMap": ZFeatureMap,
        "PauliFeatureMap": PauliFeatureMap,
    }

    ANSATZ_CATALOGUE: dict[str, type] = {
        "RealAmplitudes": RealAmplitudes,
        "EfficientSU2": EfficientSU2,
        "TwoLocal": TwoLocal,
    }

    OPTIMIZER_CATALOGUE: dict[str, Any] = {
        "COBYLA": COBYLA,
        "SPSA": SPSA,
        "L_BFGS_B": L_BFGS_B,
    }

    def __init__(
        self,
        n_qubits: int,
        max_iter: int = 100,
        reps: int = 1,
        seed: int = 42,
    ) -> None:
        if not _QML_AVAILABLE:
            raise ImportError(
                "qiskit-machine-learning is required for QuantumModelFactory. "
                f"Original error: {_QML_IMPORT_ERROR}"
            )
        self.n_qubits = n_qubits
        self.max_iter = max_iter
        self.reps = reps
        self.seed = seed

    # ── Feature maps ─────────────────────────────────────────────────────────

    def get_feature_maps(self) -> list[tuple[str, Any]]:
        """
        Return a list of ``(name, feature_map_instance)`` for this qubit count.

        For low-end hardware the list is ordered from fastest to slowest:
        ``ZFeatureMap`` (depth 1) → ``ZZFeatureMap`` (depth ∝ n) →
        ``PauliFeatureMap`` (depth ∝ n²).
        """
        n = self.n_qubits
        maps = [
            (
                "ZFeatureMap",
                ZFeatureMap(feature_dimension=n, reps=self.reps),
            ),
            (
                "ZZFeatureMap",
                ZZFeatureMap(feature_dimension=n, reps=self.reps, entanglement="linear"),
            ),
            (
                "PauliFeatureMap",
                PauliFeatureMap(
                    feature_dimension=n,
                    reps=self.reps,
                    paulis=["Z", "ZZ"],
                ),
            ),
        ]
        logger.debug("get_feature_maps: returning %d feature maps", len(maps))
        return maps

    # ── Ansatzes ─────────────────────────────────────────────────────────────

    def get_ansatzes(self) -> list[tuple[str, Any]]:
        """
        Return a list of ``(name, ansatz_instance)`` for this qubit count.

        Ordered from fewest to most parameters (fastest → richest).
        """
        n = self.n_qubits
        ansatzes = [
            (
                "RealAmplitudes",
                RealAmplitudes(num_qubits=n, reps=self.reps, entanglement="linear"),
            ),
            (
                "EfficientSU2",
                EfficientSU2(num_qubits=n, reps=self.reps, entanglement="linear"),
            ),
            (
                "TwoLocal",
                TwoLocal(
                    num_qubits=n,
                    rotation_blocks=["ry", "rz"],
                    entanglement_blocks="cx",
                    reps=self.reps,
                    entanglement="linear",
                ),
            ),
        ]
        logger.debug("get_ansatzes: returning %d ansatzes", len(ansatzes))
        return ansatzes

    # ── Optimizers ───────────────────────────────────────────────────────────

    def _make_optimizer(self, name: str = "COBYLA") -> Any:
        """
        Instantiate a Qiskit optimiser.

        COBYLA is the default for low-end hardware — it is gradient-free,
        robust, and converges well with few iterations.
        SPSA is preferred when running on real hardware (shot noise).
        """
        cls = self.OPTIMIZER_CATALOGUE.get(name, COBYLA)
        kwargs: dict[str, Any] = {"maxiter": self.max_iter}
        if name == "SPSA":
            kwargs["max_trials"] = self.max_iter
            del kwargs["maxiter"]
        return cls(**kwargs)

    # ── VQC (classifier) ─────────────────────────────────────────────────────

    def build_vqc(
        self,
        feature_map: Any,
        ansatz: Any,
        optimizer_name: str = "COBYLA",
    ) -> VQC:
        """
        Assemble a :class:`qiskit_machine_learning.algorithms.VQC`.

        Parameters
        ----------
        feature_map : QuantumCircuit
        ansatz : QuantumCircuit
        optimizer_name : str

        Returns
        -------
        VQC
        """
        sampler = StatevectorSampler()
        optimizer = self._make_optimizer(optimizer_name)
        np.random.seed(self.seed)
        initial_point = (
            np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
        )
        vqc = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
        )
        logger.debug(
            "Built VQC: feature_map=%s, ansatz=%s, params=%d",
            feature_map.__class__.__name__,
            ansatz.__class__.__name__,
            ansatz.num_parameters,
        )
        return vqc

    # ── VQR (regressor) ──────────────────────────────────────────────────────

    def build_vqr(
        self,
        feature_map: Any,
        ansatz: Any,
        optimizer_name: str = "COBYLA",
    ) -> VQR:
        """
        Assemble a :class:`qiskit_machine_learning.algorithms.VQR`.

        Parameters
        ----------
        feature_map : QuantumCircuit
        ansatz : QuantumCircuit
        optimizer_name : str

        Returns
        -------
        VQR
        """
        estimator = StatevectorEstimator()
        optimizer = self._make_optimizer(optimizer_name)
        np.random.seed(self.seed)
        initial_point = (
            np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
        )
        vqr = VQR(
            estimator=estimator,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
        )
        logger.debug("Built VQR with %d parameters.", ansatz.num_parameters)
        return vqr

    # ── QSVC (kernel classifier) ─────────────────────────────────────────────

    def build_qsvc(
        self,
        feature_map: Any,
        C: float = 1.0,
    ) -> QSVC:
        """
        Assemble a :class:`qiskit_machine_learning.algorithms.QSVC`.

        The quantum kernel is computed with a
        :class:`qiskit_machine_learning.kernels.FidelityQuantumKernel` using
        the ``StatevectorSampler`` primitive (fast, CPU-only).

        Parameters
        ----------
        feature_map : QuantumCircuit
        C : float
            Regularisation parameter passed to the inner SVM.

        Returns
        -------
        QSVC
        """
        sampler = StatevectorSampler()
        kernel = FidelityQuantumKernel(
            feature_map=feature_map,
            kernel = FidelityQuantumKernel(feature_map=feature_map)
        )
        qsvc = QSVC(quantum_kernel=kernel, C=C)
        logger.debug("Built QSVC with kernel feature_map=%s", feature_map.__class__.__name__)
        return qsvc

    # ── QSVR (kernel regressor) ──────────────────────────────────────────────

    def build_qsvr(
        self,
        feature_map: Any,
        C: float = 1.0,
    ) -> QSVR:
        """
        Assemble a :class:`qiskit_machine_learning.algorithms.QSVR`.

        Parameters
        ----------
        feature_map : QuantumCircuit
        C : float

        Returns
        -------
        QSVR
        """
        sampler = StatevectorSampler()
        kernel = FidelityQuantumKernel(
            feature_map=feature_map,
            kernel = FidelityQuantumKernel(feature_map=feature_map)
        )
        qsvr = QSVR(quantum_kernel=kernel, C=C)
        logger.debug("Built QSVR with kernel feature_map=%s", feature_map.__class__.__name__)
        return qsvr

    # ── Full catalogue for AutoML search ─────────────────────────────────────

    def candidate_classifiers(
        self, include_kernel: bool = True
    ) -> list[tuple[ModelSpec, Any]]:
        """
        Generate all candidate classifier model instances for search.

        Returns a list of ``(ModelSpec, sklearn-compatible estimator)`` pairs.
        """
        candidates: list[tuple[ModelSpec, Any]] = []
        feature_maps = self.get_feature_maps()
        ansatzes = self.get_ansatzes()

        for fm_name, fm in feature_maps:
            # VQC variants
            for az_name, az in ansatzes:
                for opt_name in ["COBYLA"]:   # keep search space small by default
                    try:
                        model = self.build_vqc(fm, az, opt_name)
                        spec = ModelSpec(
                            name=f"VQC_{fm_name}_{az_name}_{opt_name}",
                            feature_map_name=fm_name,
                            ansatz_name=az_name,
                            model_type="vqc",
                            n_qubits=self.n_qubits,
                            reps=self.reps,
                            optimizer_name=opt_name,
                        )
                        candidates.append((spec, model))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Skipping VQC %s/%s: %s", fm_name, az_name, exc)

            # QSVC (one per feature map)
            if include_kernel:
                try:
                    model = self.build_qsvc(fm)
                    spec = ModelSpec(
                        name=f"QSVC_{fm_name}",
                        feature_map_name=fm_name,
                        ansatz_name=None,
                        model_type="qsvc",
                        n_qubits=self.n_qubits,
                        reps=self.reps,
                        optimizer_name="N/A",
                    )
                    candidates.append((spec, model))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping QSVC %s: %s", fm_name, exc)

        logger.info("Generated %d candidate classifiers.", len(candidates))
        return candidates

    def candidate_regressors(
        self, include_kernel: bool = True
    ) -> list[tuple[ModelSpec, Any]]:
        """
        Generate all candidate regressor model instances for search.
        """
        candidates: list[tuple[ModelSpec, Any]] = []
        feature_maps = self.get_feature_maps()
        ansatzes = self.get_ansatzes()

        for fm_name, fm in feature_maps:
            for az_name, az in ansatzes:
                try:
                    model = self.build_vqr(fm, az)
                    spec = ModelSpec(
                        name=f"VQR_{fm_name}_{az_name}",
                        feature_map_name=fm_name,
                        ansatz_name=az_name,
                        model_type="vqr",
                        n_qubits=self.n_qubits,
                        reps=self.reps,
                        optimizer_name="COBYLA",
                    )
                    candidates.append((spec, model))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping VQR %s/%s: %s", fm_name, az_name, exc)

            if include_kernel:
                try:
                    model = self.build_qsvr(fm)
                    spec = ModelSpec(
                        name=f"QSVR_{fm_name}",
                        feature_map_name=fm_name,
                        ansatz_name=None,
                        model_type="qsvr",
                        n_qubits=self.n_qubits,
                        reps=self.reps,
                        optimizer_name="N/A",
                    )
                    candidates.append((spec, model))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping QSVR %s: %s", fm_name, exc)

        logger.info("Generated %d candidate regressors.", len(candidates))
        return candidates
