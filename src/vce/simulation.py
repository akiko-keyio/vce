# vce/simulation.py ─ Advanced Monte‑Carlo test‑bed for LS‑VCE
# ================================================================
# This module synthesises *real‑world like* observation scenarios to
# stress‑test the Variance Component Estimators implemented in
# ``vce.core`` (HelmertVCE, LSVCE, …).  It supersedes the older
# experimental script and now covers a far wider range of stochastic
# settings – from simple block‑diagonal group variances to fully
# correlated, banded or even random SPD covariances – while remaining
# **self‑contained** and reproducible.
#
#  ──────────────────────────────────────────────────────────────
#  References (abbr. used in docstrings)
#  ──────────────────────────────────────────────────────────────
#  TAS 2008 : Teunissen & Amiri‑Simkooei (2008)
#  AS 2007  : Amiri‑Simkooei PhD thesis (2007)
#
#  Formula tags like «Eq. (2) TAS 2008» in comments point to the
#  definition of Q_y  =  Q₀ + Σ σ_k Q_k  used throughout.
#  ------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Sequence

import numpy as np
import scipy.linalg as la
import scipy.stats
from numpy.random import Generator, default_rng
from scipy.stats import ortho_group

# ――― VCE estimators ---------------------------------------------------
from vce.core import HelmertVCE, LSVCE  # noqa: F401

__all__ = [
    "Scenario",
    "generate_design_matrix",
    "generate_q_blocks",
    "simulate_y",
    "run_estimators",
    "monte_carlo",
]

# ---------------------------------------------------------------------
# 1. Scenario specification
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Scenario:
    """High‑level *recipe* for a Monte‑Carlo run.

    Parameters
    ----------
    name : str
        Free‑form scenario label.
    m : int
        Number of observations.
    r_dim : int
        Rank of the design matrix A (number of unknown parameters).
    sigma_true : Sequence[float]
        True variance components **σ** (size *p*).
    n_trials : int, default=100
        Independent Monte‑Carlo replications.
    q_structure : Literal["disjoint", "overlapping", "toeplitz", "banded", "random"]
        Structural pattern of {Q_k}. See :func:`generate_q_blocks`.
    A_cond : float | None, default=None
        Target condition number of the design matrix.  *None* ⇒ draw
        standard normal matrix.
    outlier_params : dict | None, default=None
        ``{"fraction": 0.05, "magnitude": 10.0}`` injects outliers in the
        synthetic observations (Robustness test).
    q0_factor : float | None, default=None
        Optional scaling factor «α» to create a *known* covariance part
        **Q₀ = α·Q₁** (*Eq. (2) TAS 2008*).
    seed : int | None, default=None
        NumPy RNG seed for full reproducibility.
    """

    name: str
    m: int
    r_dim: int
    sigma_true: Sequence[float]
    n_trials: int = 100
    q_structure: Literal[
        "disjoint", "overlapping", "toeplitz", "banded", "random"
    ] = "disjoint"
    A_cond: Optional[float] = None
    outlier_params: Optional[Dict[str, float]] = None
    q0_factor: Optional[float] = None
    seed: Optional[int] = None


# ---------------------------------------------------------------------
# 2. Design matrix generator
# ---------------------------------------------------------------------

def generate_design_matrix(
    m: int,
    r_dim: int,
    rng: Generator,
    cond: Optional[float] = None,
) -> np.ndarray:
    """Return **A** ∈ R^{m×r} with *controlled* condition number.

    * If ``cond is None`` – draw a standard iid N(0,1) matrix.
    * Else – build ``A = U Σ Vᵀ`` with singular values spread logarithmically
      between 1 and *cond*.
    """

    if cond is None:
        return rng.standard_normal(size=(m, r_dim))

    U = ortho_group.rvs(dim=m, random_state=rng)
    V = ortho_group.rvs(dim=r_dim, random_state=rng)
    # singular values range linearly → numeric condition = cond
    Σ = np.linspace(1.0, cond, r_dim)
    S = np.zeros((m, r_dim))
    S[: r_dim, : r_dim] = np.diag(Σ)
    return U @ S @ V.T


# ---------------------------------------------------------------------
# 3. Covariance block generator
# ---------------------------------------------------------------------

def _toeplitz_block(m: int, ρ: float) -> np.ndarray:
    """Toeplitz(ρ^{|i−j|}) positive‑definite block (1×1 component)."""

    idx = np.arange(m)
    return ρ ** np.abs(np.subtract.outer(idx, idx))


def _banded_block(m: int, bandwidth: int) -> np.ndarray:
    """Unit‑diagonal SPD banded block (half‑bandwidth = *bandwidth*)."""

    Q = np.eye(m)
    for k in range(1, bandwidth + 1):
        np.fill_diagonal(Q[k:], 0.8 ** k)
        np.fill_diagonal(Q[:, k:], 0.8 ** k)
    return Q


def _random_spd(m: int, rng: Generator) -> np.ndarray:
    """Random *scaled* SPD matrix via Wishart draw."""

    X = rng.standard_normal(size=(m, 2 * m))
    Q = X @ X.T  # Wishart(df=2m)
    Q /= np.trace(Q) / m  # normalise average variance → 1
    return Q


def generate_q_blocks(
    m: int,
    structure: Literal[
        "disjoint", "overlapping", "toeplitz", "banded", "random",
    ],
    rng: Optional[Generator] = None,
) -> list[np.ndarray]:
    """Return a *list* {Q_k} that spans most practical noise patterns.

    *disjoint*      – **block‑diagonal, non‑overlapping** group variances
    *overlapping*   – three partially overlapping groups (old default)
    *toeplitz*      – AR(1)‑like correlation (ρ = 0.8)
    *banded*        – tri‑ or pentadiagonal correlation
    *random*        – fully populated random SPD matrices
    """

    rng = default_rng() if rng is None else rng

    if structure == "disjoint":
        sizes = (m // 3, m // 3, m - 2 * (m // 3))
        q_blocks: list[np.ndarray] = []
        cursor = 0
        for size in sizes:
            blk = np.zeros((m, m))
            idx = slice(cursor, cursor + size)
            blk[idx, idx] = np.eye(size)
            q_blocks.append(blk)
            cursor += size
        return q_blocks

    if structure == "overlapping":
        q1, q2, q3 = np.zeros((m, m)), np.zeros((m, m)), np.zeros((m, m))
        mid1, mid2 = m // 2, 3 * m // 4
        q1[:mid1, :mid1] = np.eye(mid1)
        q2[m // 4 : mid2, m // 4 : mid2] = np.eye(mid2 - m // 4)
        q3[mid2:, mid2:] = np.eye(m - mid2)
        return [q1, q2, q3]

    if structure == "toeplitz":
        return [_toeplitz_block(m, 0.8)]

    if structure == "banded":
        # build *two* variance components: bandwidth‑1 and bandwidth‑3
        return [_banded_block(m, 1), _banded_block(m, 3)]

    if structure == "random":
        return [_random_spd(m, rng) for _ in range(3)]

    raise ValueError(f"Unknown q_structure: {structure}")


# ---------------------------------------------------------------------
# 4. Observation generator  (y = A b + e)
# ---------------------------------------------------------------------

def simulate_y(
    A: np.ndarray,
    b: np.ndarray,
    Q_y: np.ndarray,
    rng: Generator,
    outlier_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Draw one **y** sample and optionally contaminate with outliers.

    *Baseline*:  e ~ N(0, Q_y)  (TAS 2008 Eq. (1)).
    *Outliers*:  add *k* large deviations (L/σ multiplier) at random indices.
    """

    y = A @ b + rng.multivariate_normal(np.zeros(A.shape[0]), Q_y)

    if outlier_params is not None:
        frac = float(outlier_params.get("fraction", 0.05))
        mag = float(outlier_params.get("magnitude", 10.0))
        n_out = int(frac * A.shape[0])
        if n_out > 0:
            idx = rng.choice(A.shape[0], n_out, replace=False)
            # scale magnitude by average σ (trace(Q_y)/m) for comparability
            noise_std = np.sqrt(np.trace(Q_y) / A.shape[0])
            y[idx] += mag * noise_std * rng.standard_normal(size=n_out)
    return y


# ---------------------------------------------------------------------
# 5. Wrapper – run all available estimators
# ---------------------------------------------------------------------

def _available_estimators():
    """Return *name → class* mapping of every VCE estimator shipped."""

    estimators: Dict[str, Any] = {
        "helmert": HelmertVCE,
        "lsvce": LSVCE,
    }
    return estimators


def run_estimators(
    A: np.ndarray,
    q_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    Q0: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fit **all** shipped estimators on a *single* observation vector."""

    out: Dict[str, Dict[str, Any]] = {}
    for name, cls in _available_estimators().items():
        est = cls(A, q_blocks, Q0=Q0, verbose=False)
        est.fit(y, sigma0=np.ones(len(q_blocks)))
        out[name] = {
            "sigma": est.sigma_.copy(),
            "cov_theo": est.cov_theo_.copy(),
            "chi2": float(est.chi2_),
            "n_iter": int(est.n_iter_),
            "converged": bool(est.converged_),
        }
    return out


# ---------------------------------------------------------------------
# 6. Monte‑Carlo driver
# ---------------------------------------------------------------------

def monte_carlo(scn: Scenario) -> Dict[str, Dict[str, np.ndarray]]:
    """Run *n_trials* synthetic experiments under *Scenario scn*.

    Returned structure  →  dict[estimator][metric].  Metrics:
        • sigma      – (N, p)   stacked σ̂ per trial
        • cov_theo   – (N, p, p) theoretical Cov{σ̂}
        • chi2       – (N,)      residual χ² value
        • n_iter     – (N,)      iteration count
        • converged  – (N,)      boolean flag
    """

    rng = default_rng(scn.seed)

    # 1. experiment design ------------------------------------------------
    A = generate_design_matrix(scn.m, scn.r_dim, rng, cond=scn.A_cond)
    b_true = rng.standard_normal(scn.r_dim)

    q_blocks = generate_q_blocks(scn.m, scn.q_structure, rng)
    p = len(q_blocks)
    if len(scn.sigma_true) != p:
        raise ValueError("Length of sigma_true must match number of Q_blocks")

    Q0 = None if scn.q0_factor is None else scn.q0_factor * q_blocks[0]
    Q_stochastic = sum(σ * Qk for σ, Qk in zip(scn.sigma_true, q_blocks))
    Q_y_true = (0 if Q0 is None else Q0) + Q_stochastic  # Eq. (2) TAS 2008

    # 2. storage allocation ----------------------------------------------
    metrics = ("sigma", "cov_theo", "chi2", "n_iter", "converged")
    results: Dict[str, Dict[str, list]] = {
        est: {m: [] for m in metrics} for est in _available_estimators()
    }

    # 3. Monte‑Carlo loop -------------------------------------------------
    for _ in range(scn.n_trials):
        y = simulate_y(A, b_true, Q_y_true, rng, scn.outlier_params)
        trial_stats = run_estimators(A, q_blocks, y, Q0)
        for est, stats in trial_stats.items():
            for m in metrics:
                results[est][m].append(stats[m])

    # 4. stack → ndarray --------------------------------------------------
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for est, data in results.items():
        out[est] = {
            "sigma": np.vstack(data["sigma"]),
            "cov_theo": np.stack(data["cov_theo"]),
            "chi2": np.array(data["chi2"], float),
            "n_iter": np.array(data["n_iter"], int),
            "converged": np.array(data["converged"], bool),
        }
    return out


# ---------------------------------------------------------------------
# 7. Quick self‑test (executed only when run as script) ----------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scn = Scenario(
        name="demo",
        m=10,
        r_dim=10,
        sigma_true=(3.0, 1.5, 0.5),
        n_trials=50,
        q_structure="overlapping",
        A_cond=50,
        outlier_params={"fraction": 0.0, "magnitude": 8.0},
        q0_factor=0.2,
        seed=42,
    )

    logging.info("Running demo scenario …")
    stats = monte_carlo(scn)
    for est, res in stats.items():
        logging.info(
            "%s: mean σ̂ = %s",
            est,
            np.mean(res["sigma"], axis=0).round(2),
        )
