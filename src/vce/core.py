# -*- coding: utf-8 -*-
"""
Variance‑Component Estimation (VCE) Library — revised
======================================================
Fixes & upgrades after formal cross‑check with Teunissen & Amiri‑Simkooei
(2007) and Amiri‑Simkooei (2007):

• **Bug‑fix (LSVCEPlus)**  — the right‑hand vector rₖ must use the *known*
  part Q₀ of the covariance model (Eq. 4.105, thesis; Eq. 36, journal), **not** Q_y.
  If Q₀ is absent (the common case), it is 0.
• Added user‑settable attribute **Q0** to supply Q₀ when needed.
• Minor: docstrings cite exact equations; improved type hints; left Helmert &
  unit‑weight LS‑VCE untouched (they were already correct).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Optional, Literal

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def orthogonal_projector(
    A: np.ndarray, Q_inv: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return P = A(AᵀQ⁻¹A)⁻¹AᵀQ⁻¹ and P_perp = I − P."""
    At = A.T
    N = At @ Q_inv @ A
    # Avoid forming the inverse explicitly for better performance
    M = np.linalg.solve(N, At @ Q_inv)
    P = A @ M
    return P, np.eye(A.shape[0]) - P


def trace_of_product(*mats: np.ndarray) -> float:
    """Return ``trace(M₁⋯Mₙ)`` using ``multi_dot`` for speed."""
    prod = np.linalg.multi_dot(mats)
    return float(np.trace(prod))


# ---------------------------------------------------------------------------
# Base scaffold (unchanged)
# ---------------------------------------------------------------------------


@dataclass
class VCEBase:
    A: np.ndarray
    Q_blocks: Sequence[np.ndarray]
    max_iter: int = 50
    tol: float = 1e-10
    verbose: bool = False

    sigma_: np.ndarray = field(init=False, repr=False)
    converged_: bool = field(init=False, default=False)
    n_iter_: int = field(init=False, default=0)
    residual_: np.ndarray = field(init=False, repr=False)
    chi2_: float = field(init=False, default=float("nan"))
    cov_theo_: np.ndarray = field(init=False, repr=False)

    # subclasses implement one iteration -------------------------------------
    def _update_step(
        self, y: np.ndarray, P_perp: np.ndarray, e: np.ndarray
    ) -> np.ndarray:  # noqa: N802
        raise NotImplementedError

    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray:
        """Return theoretical covariance of sigma-hat."""
        raise NotImplementedError

    # driver loop -------------------------------------------------------------
    def fit(self, y: np.ndarray, sigma0: Optional[Sequence[float]] = None):
        y = np.asarray(y, float)
        m = y.size
        if any(Q.shape != (m, m) for Q in self.Q_blocks):
            raise ValueError("Every Q_k must be m×m and match y.")

        self.sigma_ = (
            np.asarray(sigma0, float)
            if sigma0 is not None
            else np.array([y @ Q @ y / np.trace(Q) for Q in self.Q_blocks])
        )

        for it in range(1, self.max_iter + 1):
            Q_y = sum(s * Q for s, Q in zip(self.sigma_, self.Q_blocks))
            Q_inv = np.linalg.inv(Q_y)
            _, P_perp = orthogonal_projector(self.A, Q_inv)
            e = P_perp @ y
            sigma_new = self._update_step(y, P_perp, e)
            if np.allclose(self.sigma_, sigma_new, rtol=self.tol, atol=self.tol):
                self.converged_ = True
                self.n_iter_ = it
                break
            if self.verbose:
                print(f"iter {it:02d}: σ = {sigma_new}")
            self.sigma_ = sigma_new
            self.n_iter_ = it
        Q_y = self.predict_Q()
        Q_inv = np.linalg.inv(Q_y)
        _, P_perp = orthogonal_projector(self.A, Q_inv)
        self.residual_ = P_perp @ y
        self.chi2_ = float(self.residual_ @ Q_inv @ self.residual_)
        self.cov_theo_ = self._compute_covariance(P_perp, Q_y)
        return self

    def predict_Q(self) -> np.ndarray:
        return sum(s * Q for s, Q in zip(self.sigma_, self.Q_blocks))

    @property
    def sigma(self) -> np.ndarray:  # pragma: no cover
        return self.sigma_


# ---------------------------------------------------------------------------
# 1. Helmert estimator (unchanged)
# ---------------------------------------------------------------------------


class HelmertVCE(VCEBase):
    method: Literal["helmert"] = "helmert"

    H_inv: np.ndarray = field(init=False, repr=False)

    def _update_step(self, y, P_perp, e):  # noqa: N802
        p = len(self.Q_blocks)
        E = [P_perp @ Q @ P_perp / s for Q, s in zip(self.Q_blocks, self.sigma_)]
        q = np.array([e @ (Ek @ e) for Ek in E])
        H = np.empty((p, p))
        for k in range(p):
            for j in range(p):
                H[k, j] = trace_of_product(E[k], P_perp, self.Q_blocks[j], P_perp)
        return np.linalg.solve(H, q)

    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray:
        p = len(self.Q_blocks)
        E = [P_perp @ Q @ P_perp / s for Q, s in zip(self.Q_blocks, self.sigma_)]
        H = np.empty((p, p))
        for k in range(p):
            for j in range(p):
                H[k, j] = trace_of_product(E[k], P_perp, self.Q_blocks[j], P_perp)
        self.H_inv = np.linalg.inv(H)
        return self.H_inv


# ---------------------------------------------------------------------------
# 2. LS‑VCE (unit weight) — unchanged
# ---------------------------------------------------------------------------


class LSVCE(VCEBase):
    method: Literal["lsvce"] = "lsvce"

    N_inv: np.ndarray = field(init=False, repr=False)

    def _update_step(self, y, P_perp, e):  # noqa: N802
        p = len(self.Q_blocks)
        l_vec = np.array([e @ (Q @ e) for Q in self.Q_blocks])
        N = np.empty((p, p))
        for k, Qk in enumerate(self.Q_blocks):
            for j, Ql in enumerate(self.Q_blocks):
                N[k, j] = trace_of_product(P_perp, Qk, P_perp, Ql)
        return np.linalg.solve(N, l_vec)

    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray:
        p = len(self.Q_blocks)
        N = np.empty((p, p))
        for k, Qk in enumerate(self.Q_blocks):
            for j, Ql in enumerate(self.Q_blocks):
                N[k, j] = trace_of_product(P_perp, Qk, P_perp, Ql)
        self.N_inv = np.linalg.inv(N)
        return self.N_inv


# ---------------------------------------------------------------------------
# 3. LSVCEPlus — corrected rₖ term (Eq. 4.105 / 36)
# ---------------------------------------------------------------------------


@dataclass
class LSVCEPlus(VCEBase):
    """BLUE/Minimum‑variance LS‑VCE for κ = 0.

    Parameters
    ----------
    Q0 : Optional[np.ndarray]
        Known part of the covariance model (defaults to 0).  Used in the
        second term of rₖ, see Amiri‑Simkooei thesis Eq. 4.105.
    """

    Q0: Optional[np.ndarray] = None  # known part (may be zero)
    method: Literal["lsvce_plus"] = "lsvce_plus"
    N_inv: np.ndarray = field(init=False, repr=False)

    def _update_step(self, y, P_perp, e):  # noqa: N802
        Q_y = self.predict_Q()
        Wy = (1.0 / np.sqrt(2.0)) * np.linalg.inv(Q_y)  # Eq. (49) with κ=0
        Q0 = self.Q0 if self.Q0 is not None else np.zeros_like(Q_y)

        p = len(self.Q_blocks)
        N = np.empty((p, p))
        r = np.empty(p)
        for k, Qk in enumerate(self.Q_blocks):
            for j, Ql in enumerate(self.Q_blocks):
                N[k, j] = trace_of_product(Qk, Wy, P_perp, Ql, Wy, P_perp)  # Eq. (52)
            # rₖ = eᵀ Wy Qk Wy e − ½ tr(Qk Wy P⊥ Q₀ Wy P⊥)  (Eq. 4.105)
            r[k] = e @ Wy @ Qk @ Wy @ e - 0.5 * trace_of_product(
                Qk, Wy, P_perp, Q0, Wy, P_perp
            )
        return np.linalg.solve(N, r)

    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray:
        Wy = (1.0 / np.sqrt(2.0)) * np.linalg.inv(Q_y)
        p = len(self.Q_blocks)
        N = np.empty((p, p))
        for k, Qk in enumerate(self.Q_blocks):
            for j, Ql in enumerate(self.Q_blocks):
                N[k, j] = trace_of_product(Qk, Wy, P_perp, Ql, Wy, P_perp)
        self.N_inv = np.linalg.inv(N)
        return self.N_inv


# ---------------------------------------------------------------------------
# Smoke‑test -----------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    m, r_dim, p = 120, 4, 3
    A = np.random.randn(m, r_dim)

    Q_blocks = []
    cursor = 0
    for size in (40, 40, 40):
        Qk = np.zeros((m, m))
        Qk[cursor : cursor + size, cursor : cursor + size] = 1.0
        Q_blocks.append(Qk)
        cursor += size

    sigma_true = np.array([5.0, 2.0, 1.0])
    Q_true = sum(s * Q for s, Q in zip(sigma_true, Q_blocks))
    b_true = np.array([1.2, -0.8, 0.5, 2.0])
    y = A @ b_true + np.random.multivariate_normal(np.zeros(m), Q_true)

    print("Helmert:")
    print(HelmertVCE(A, Q_blocks).fit(y).sigma)

    print("Unit‑weight LS‑VCE:")
    print(LSVCE(A, Q_blocks).fit(y).sigma)

    print("BLUE LS‑VCE (κ=0):")
    print(LSVCEPlus(A, Q_blocks).fit(y).sigma)
