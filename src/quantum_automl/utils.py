"""
quantum_automl.utils
====================

Utility functions for backend configuration, circuit visualisation,
logging setup, and performance profiling.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np

logger = logging.getLogger(__name__)

# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """
    Configure the ``quantum_automl`` logger.

    Parameters
    ----------
    level : str
        One of "DEBUG", "INFO", "WARNING", "ERROR".

    Examples
    --------
    >>> from quantum_automl.utils import setup_logging
    >>> setup_logging("DEBUG")
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    )
    pkg_logger = logging.getLogger("quantum_automl")
    pkg_logger.setLevel(numeric)
    if not pkg_logger.handlers:
        pkg_logger.addHandler(handler)


# ── Backend management ───────────────────────────────────────────────────────

# Registry of known simulator backends
_SIMULATORS = {
    "statevector_simulator": "statevector_simulator",
    "aer_simulator": "aer_simulator",
    "qasm_simulator": "qasm_simulator",
    # Statevector primitives from qiskit.primitives (default — no Aer needed)
    "statevector": "statevector",
}

_CURRENT_BACKEND: dict[str, Any] = {"name": "statevector", "instance": None}


def set_backend(backend_name: str, **kwargs: Any) -> Any:
    """
    Configure the default execution backend for quantum circuits.

    Supports simulators (CPU-only, no special hardware) and real IBM
    Quantum devices (requires ``qiskit-ibm-runtime`` and valid credentials).

    Parameters
    ----------
    backend_name : str
        One of:

        * ``"statevector"`` — pure-NumPy statevector (default, fastest on CPU)
        * ``"statevector_simulator"`` — Aer statevector
        * ``"aer_simulator"`` — Aer general-purpose simulator
        * ``"qasm_simulator"`` — shot-based QASM simulator
        * Any IBM Quantum device name, e.g. ``"ibm_brisbane"``

    **kwargs :
        Extra arguments forwarded to the backend constructor / ``IBMProvider``.

    Returns
    -------
    backend : Aer backend or IBMQBackend or None for primitives

    Examples
    --------
    >>> from quantum_automl.utils import set_backend
    >>> set_backend("aer_simulator")                        # Aer
    >>> set_backend("ibm_brisbane", token="MY_TOKEN")       # Real hardware
    """
    name = backend_name.lower().strip()

    if name in ("statevector", "statevector_primitive"):
        # Pure-NumPy primitives, no Aer needed
        _CURRENT_BACKEND["name"] = "statevector"
        _CURRENT_BACKEND["instance"] = None
        logger.info("Backend set to pure-NumPy StatevectorSampler/Estimator.")
        return None

    # ── Aer simulators ───────────────────────────────────────────────────────
    if name in _SIMULATORS:
        try:
            from qiskit_aer import AerSimulator
            backend = AerSimulator(method=name.replace("_simulator", ""))
            _CURRENT_BACKEND["name"] = name
            _CURRENT_BACKEND["instance"] = backend
            logger.info("Aer backend set: %s", name)
            return backend
        except ImportError:
            warnings.warn(
                "qiskit-aer is not installed.  "
                "Falling back to pure-NumPy statevector primitives.",
                UserWarning,
                stacklevel=2,
            )
            _CURRENT_BACKEND["name"] = "statevector"
            _CURRENT_BACKEND["instance"] = None
            return None

    # ── IBM Quantum (real hardware) ───────────────────────────────────────────
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, EstimatorV2
        token = kwargs.pop("token", None)
        if token:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum", token=token, overwrite=True
            )
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend(name, **kwargs)
        _CURRENT_BACKEND["name"] = name
        _CURRENT_BACKEND["instance"] = backend
        logger.info("IBM Quantum backend set: %s", name)
        print(
            f"⚠️  Real hardware selected: {name}\n"
            "   Remember to enable error mitigation (see utils.get_runtime_options)."
        )
        return backend
    except ImportError:
        raise ImportError(
            "Real IBM hardware requires qiskit-ibm-runtime. "
            "Install it with: pip install qiskit-ibm-runtime"
        ) from None
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to IBM Quantum backend '{name}': {exc}"
        ) from exc


def get_current_backend() -> dict[str, Any]:
    """Return the currently configured backend info dict."""
    return dict(_CURRENT_BACKEND)


def get_runtime_options(mitigation_level: int = 1) -> dict[str, Any]:
    """
    Return Qiskit Runtime options dict with error mitigation enabled.

    Useful when using real IBM hardware.

    Parameters
    ----------
    mitigation_level : int
        0 = none, 1 = readout mitigation only (fast),
        2 = TREX + ZNE (slower, more accurate).

    Returns
    -------
    dict
    """
    options: dict[str, Any] = {
        "resilience_level": mitigation_level,
        "optimization_level": 1,
    }
    if mitigation_level >= 2:
        options["resilience"] = {
            "measure_mitigation": True,
            "zne_mitigation": True,
            "zne": {"noise_factors": [1, 3, 5]},
        }
    return options


# ── Circuit visualisation ─────────────────────────────────────────────────────

def visualize_circuit(
    model: Any,
    style: str = "text",
    fold: int = 80,
) -> Any:
    """
    Draw the quantum circuit(s) of a fitted model.

    Parameters
    ----------
    model : fitted QuantumAutoClassifier | QuantumAutoRegressor | VQC | VQR
        Can be either a raw Qiskit ML model or a fitted AutoML estimator.
    style : str
        "text" (always works, no extra deps), "mpl" (needs matplotlib),
        "latex" (needs pylatexenc + pdflatex).
    fold : int
        Line-fold width for text style.

    Returns
    -------
    QuantumCircuit.draw() output (str for text, Figure for mpl).
    """
    # Unwrap AutoML wrapper
    if hasattr(model, "best_model_"):
        model = model.best_model_

    circuit = _extract_circuit(model)
    if circuit is None:
        warnings.warn(
            "Could not extract circuit from model.  "
            "The model may not expose its internal circuit.",
            UserWarning,
            stacklevel=2,
        )
        return None

    draw_kwargs: dict[str, Any] = {"output": style, "fold": fold}
    if style == "mpl":
        draw_kwargs["style"] = "iqp"
    return circuit.draw(**draw_kwargs)


def _extract_circuit(model: Any) -> Any | None:
    """
    Try common attribute paths used by Qiskit ML models to expose circuits.
    """
    # VQC / VQR expose the bound circuit after fitting
    for attr in ("circuit", "_circuit", "ansatz", "_ansatz"):
        if hasattr(model, attr):
            circuit = getattr(model, attr)
            if hasattr(circuit, "draw"):
                return circuit
    # QSVC / QSVR expose the feature map via the kernel
    if hasattr(model, "quantum_kernel"):
        fm = getattr(model.quantum_kernel, "feature_map", None)
        if fm is not None and hasattr(fm, "draw"):
            return fm
    return None


# ── Timing context manager ───────────────────────────────────────────────────

@contextmanager
def timer(label: str = "Block") -> Generator[dict[str, float], None, None]:
    """
    Simple wall-clock timer context manager.

    Examples
    --------
    >>> with timer("Search") as t:
    ...     # ... do work ...
    ...     pass
    >>> print(f"Elapsed: {t['elapsed']:.2f}s")
    """
    info: dict[str, float] = {"elapsed": 0.0}
    t0 = time.perf_counter()
    try:
        yield info
    finally:
        info["elapsed"] = time.perf_counter() - t0
        logger.debug("%s took %.3fs", label, info["elapsed"])


# ── Memory estimate ───────────────────────────────────────────────────────────

def estimate_simulator_memory_mb(n_qubits: int) -> float:
    """
    Estimate the RAM required for statevector simulation.

    Parameters
    ----------
    n_qubits : int

    Returns
    -------
    float : estimated memory in megabytes
    """
    # Statevector has 2^n complex128 amplitudes → 16 bytes each
    n_amplitudes = 2 ** n_qubits
    bytes_required = n_amplitudes * 16  # complex128 = 16 bytes
    return bytes_required / (1024 ** 2)


def check_hardware_compatibility(n_qubits: int, verbose: bool = True) -> bool:
    """
    Warn if the requested qubit count may exceed available RAM.

    Parameters
    ----------
    n_qubits : int
    verbose : bool

    Returns
    -------
    bool : True if safe, False if potentially dangerous
    """
    mem_mb = estimate_simulator_memory_mb(n_qubits)
    safe = mem_mb < 512

    if verbose:
        emoji = "✅" if safe else "⚠️ "
        print(
            f"{emoji} n_qubits={n_qubits}: statevector requires "
            f"~{mem_mb:.1f} MB  "
            f"({'safe for CPU' if safe else 'may be slow or OOM on low-end hardware'})"
        )
    return safe


# ── Quick benchmark ───────────────────────────────────────────────────────────

def benchmark_simulator(n_qubits_range: list[int] | None = None) -> dict[int, float]:
    """
    Run a simple benchmark to find the largest qubit count the machine can
    handle comfortably.

    Parameters
    ----------
    n_qubits_range : list[int] | None
        Qubit counts to test.  Defaults to [2, 4, 6, 8].

    Returns
    -------
    dict[int, float] : mapping of n_qubits → seconds per statevector simulation
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    if n_qubits_range is None:
        n_qubits_range = [2, 4, 6, 8]

    results: dict[int, float] = {}
    print("⏱  Simulator benchmark:")
    for n in n_qubits_range:
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure_all()

        sampler = StatevectorSampler()
        t0 = time.perf_counter()
        try:
            job = sampler.run([qc])
            job.result()
            elapsed = time.perf_counter() - t0
            status = f"{elapsed * 1000:.1f} ms"
        except Exception as exc:  # noqa: BLE001
            elapsed = float("inf")
            status = f"FAILED ({exc})"

        results[n] = elapsed
        mem_mb = estimate_simulator_memory_mb(n)
        print(f"  n={n:2d}  {status}  (~{mem_mb:.0f} MB)")

    return results
