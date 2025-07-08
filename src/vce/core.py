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
    """Return projectors onto ``range(A)`` and its complement.

    Parameters
    ----------
    A:
        Design matrix whose column space is the target subspace.
    Q_inv:
        Precision matrix defining the inner product.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(P, P_perp)`` where ``P`` projects onto ``range(A)`` and
        ``P_perp`` onto the orthogonal complement.
    """

    At = A.T  # transpose for reuse
    N = At @ Q_inv @ A  # normal matrix
    N_inv = np.linalg.inv(N)  # its inverse
    P = A @ (N_inv @ (At @ Q_inv))  # projector onto range(A)
    return P, np.eye(A.shape[0]) - P  # complementary projector


def trace_of_product(*mats: np.ndarray) -> float:
    """Return the trace of a matrix product.

    Multiplication is performed sequentially in the supplied order.
    """
    prod = mats[0]  # first factor
    for M in mats[1:]:
        prod = prod @ M  # accumulate product
    return float(np.trace(prod))


# ---------------------------------------------------------------------------
# Base scaffold (unchanged)
# ---------------------------------------------------------------------------


@dataclass
class VCEBase:
    """Common driver for variance-component estimators.

    Parameters
    ----------
    A : np.ndarray
        Design matrix relating observations to parameters.
    Q_blocks : Sequence[np.ndarray]
        Basis matrices defining the covariance model.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance for ``sigma`` updates.
    verbose : bool, optional
        If ``True``, print intermediate estimates.

    Attributes
    ----------
    sigma_ : np.ndarray
        Final variance-component estimates.
    converged_ : bool
        Whether the algorithm met the tolerance.
    n_iter_ : int
        Number of iterations executed.
    residual_ : np.ndarray
        Post-fit residual vector.
    chi2_ : float
        Residual chi-square statistic.
    cov_theo_ : np.ndarray
        Theoretical covariance matrix of ``sigma_``.
    """

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
        """Return updated variance components for one iteration."""
        raise NotImplementedError

    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray:
        """Compute the theoretical covariance of ``sigma_``."""
        raise NotImplementedError

    # driver loop -------------------------------------------------------------
    def fit(self, y: np.ndarray, sigma0: Optional[Sequence[float]] = None):
        """Iteratively estimate variance components from ``y``."""
        y = np.asarray(y, float)  # observation vector
        m = y.size  # number of observations
        if any(Q.shape != (m, m) for Q in self.Q_blocks):
            raise ValueError("Every Q_k must be m×m and match y.")

        # initial variance components
        self.sigma_ = (
            np.asarray(sigma0, float)
            if sigma0 is not None
            else np.array([y @ Q @ y / np.trace(Q) for Q in self.Q_blocks])
        )

        for it in range(1, self.max_iter + 1):  # main iteration loop
            Q_y = sum(
                s * Q for s, Q in zip(self.sigma_, self.Q_blocks)
            )  # build covariance
            Q_inv = np.linalg.inv(Q_y)  # precision matrix
            _, P_perp = orthogonal_projector(self.A, Q_inv)  # projectors
            e = P_perp @ y  # residual vector
            sigma_new = self._update_step(y, P_perp, e)  # subclass update
            if np.allclose(self.sigma_, sigma_new, rtol=self.tol, atol=self.tol):
                self.converged_ = True
                self.n_iter_ = it
                break
            if self.verbose:
                print(f"iter {it:02d}: σ = {sigma_new}")
            self.sigma_ = sigma_new
            self.n_iter_ = it
        Q_y = self.predict_Q()  # final covariance estimate
        Q_inv = np.linalg.inv(Q_y)
        _, P_perp = orthogonal_projector(self.A, Q_inv)
        self.residual_ = P_perp @ y
        self.chi2_ = float(self.residual_ @ Q_inv @ self.residual_)
        self.cov_theo_ = self._compute_covariance(P_perp, Q_y)
        return self

    def predict_Q(self) -> np.ndarray:
        """Return the predicted covariance matrix ``Q_y``."""
        return sum(s * Q for s, Q in zip(self.sigma_, self.Q_blocks))

    @property
    def sigma(self) -> np.ndarray:  # pragma: no cover
        """Variance component estimates."""
        return self.sigma_


# ---------------------------------------------------------------------------
# 1. Helmert estimator (unchanged)
# ---------------------------------------------------------------------------


class HelmertVCE(VCEBase):
    """Helmert variance-component estimator."""

    method: Literal["helmert"] = "helmert"

    H_inv: np.ndarray = field(init=False, repr=False)

    def _update_step(self, y, P_perp, e):  # noqa: N802
        """Compute next ``sigma`` using the Helmert formulation."""
        p = len(self.Q_blocks)
        E = [
            P_perp @ Q @ P_perp / s for Q, s in zip(self.Q_blocks, self.sigma_)
        ]  # Eq. matrix
        q = np.array([e @ (Ek @ e) for Ek in E])  # right-hand side
        H = np.empty((p, p))  # normal matrix
        for k in range(p):
            for j in range(p):
                # build H from traces of products
                H[k, j] = trace_of_product(E[k], P_perp, self.Q_blocks[j], P_perp)
        return np.linalg.solve(H, q)

    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray:
        """Return the covariance of the Helmert estimator."""
        p = len(self.Q_blocks)
        E = [
            P_perp @ Q @ P_perp / s for Q, s in zip(self.Q_blocks, self.sigma_)
        ]  # reuse E
        H = np.empty((p, p))  # assemble normal matrix
        for k in range(p):
            for j in range(p):
                # same trace expression as in update step
                H[k, j] = trace_of_product(E[k], P_perp, self.Q_blocks[j], P_perp)
        self.H_inv = np.linalg.inv(H)
        return self.H_inv


# ---------------------------------------------------------------------------
# 2. LS‑VCE (unit weight) — unchanged
# ---------------------------------------------------------------------------


class LSVCE(VCEBase):
    """Least-squares variance-component estimator."""

    method: Literal["lsvce"] = "lsvce"

    N_inv: np.ndarray = field(init=False, repr=False)

    def _update_step(self, y, P_perp, e):  # noqa: N802
        """Update ``sigma`` using the LS-VCE scheme."""
        p = len(self.Q_blocks)
        l_vec = np.array([e @ (Q @ e) for Q in self.Q_blocks])  # rhs vector
        N = np.empty((p, p))  # normal matrix
        for k, Qk in enumerate(self.Q_blocks):
            for j, Ql in enumerate(self.Q_blocks):
                # build N from traces of projected blocks
                N[k, j] = trace_of_product(P_perp, Qk, P_perp, Ql)
        return np.linalg.solve(N, l_vec)

    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray:
        """Return the covariance of the LS-VCE estimator."""
        p = len(self.Q_blocks)
        N = np.empty((p, p))  # same normal matrix as update step
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
        """Update ``sigma`` using the BLUE variant."""
        Q_y = self.predict_Q()
        Wy = (1.0 / np.sqrt(2.0)) * np.linalg.inv(Q_y)  # Eq. (49) with κ=0
        Q0 = self.Q0 if self.Q0 is not None else np.zeros_like(Q_y)

        p = len(self.Q_blocks)
        N = np.empty((p, p))  # normal matrix
        r = np.empty(p)  # right-hand side
        for k, Qk in enumerate(self.Q_blocks):
            for j, Ql in enumerate(self.Q_blocks):
                # Eq. (52) - build normal matrix
                N[k, j] = trace_of_product(Qk, Wy, P_perp, Ql, Wy, P_perp)
            # rₖ = eᵀ Wy Qk Wy e − ½ tr(Qk Wy P⊥ Q₀ Wy P⊥)  (Eq. 4.105)
            r[k] = e @ Wy @ Qk @ Wy @ e - trace_of_product(
                Qk, Wy, P_perp, Q0, Wy, P_perp
            )

        return np.linalg.solve(N, r)

    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray:
        """Return the covariance of the LS-VCE+ estimator."""
        Wy = (1.0 / np.sqrt(2.0)) * np.linalg.inv(Q_y)
        p = len(self.Q_blocks)
        N = np.empty((p, p))  # reuse normal matrix
        for k, Qk in enumerate(self.Q_blocks):
            for j, Ql in enumerate(self.Q_blocks):
                N[k, j] = trace_of_product(Qk, Wy, P_perp, Ql, Wy, P_perp)
        self.N_inv = np.linalg.inv(N)
        return self.N_inv
