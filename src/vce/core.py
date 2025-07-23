# vce/core.py ― Definitive version, fully documented
# ==================================================
# References (abbrev. used below):
#   ▸ TAS 2008  = Teunissen & Amiri-Simkooei (2008), “Least-Squares Variance Component Estimation”
#   ▸ AS 2007   = Amiri-Simkooei (2007), PhD thesis, Ch. 3–4   (Helmert VCE)
#
# Formula numbers in comments correspond to those publications
# (e.g. “Eq. (60) TAS 2008”).

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------
def oblique_projector(A: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the (generally *non-orthogonal*) projector **P** and its complement **P_perp**:

        P        = A (Aᵀ W A)⁻¹ Aᵀ W                       # Eq. (17) TAS 2008
        P_perp   = I − P                                     # Eq. (18) TAS 2008

    These satisfy P² = P but Pᵀ ≠ P in general.  W is the inverse of the
    observation covariance matrix, *i.e.* W = Q_y⁻¹.

    Parameters
    ----------
    A : (m, n) ndarray
        Design matrix.
    W : (m, m) ndarray
        Weight matrix (inverse covariance).

    Returns
    -------
    P : (m, m) ndarray
        Oblique projector onto range(A).
    P_perp : (m, m) ndarray
        Complementary projector onto the *residual* space.
    """
    AtW = A.T @ W
    N_inv = np.linalg.pinv(AtW @ A)
    P = A @ (N_inv @ AtW)
    return P, np.eye(A.shape[0]) - P


def trace_of_product(*mats: np.ndarray) -> float:
    """
    Compute **tr(M₁ M₂ … M_k)** with a single pass.

    The function exists only for readability; no numerical shortcut is taken.
    """
    prod = mats[0]
    for M in mats[1:]:
        prod = prod @ M
    return float(np.trace(prod))


# ---------------------------------------------------------------------
# Generic LS-VCE framework
# ---------------------------------------------------------------------
@dataclass
class VCEBase:
    r"""
    Abstract super-class for iterative *Least-Squares Variance Component
    Estimation* (LS-VCE).

    Functional model   :  y = A x + e                                # Eq. (1) TAS 2008
    Stochastic model   :  Q_y = Q₀ + Σ σ_k Q_k                       # Eq. (2) TAS 2008

    All concrete subclasses need to implement two hooks:

    * ``_update_step``   – one iteration of σ ← σ + Δσ
    * ``_compute_covariance`` – asymptotic Cov{σ̂} after convergence
    """

    # ---------------- public inputs ----------------
    A: np.ndarray
    Q_blocks: Sequence[np.ndarray]
    max_iter: int = 50
    tol: float = 1e-10
    verbose: bool = False
    Q0: Optional[np.ndarray] = None
    enforce_non_negative: bool = False
    singular_tol: float = 1e12

    # ---------------- results (populated by .fit) --
    sigma_: np.ndarray = field(init=False)  # shape (p,)
    converged_: bool = field(init=False, default=False)
    n_iter_: int = field(init=False, default=0)
    residual_: np.ndarray = field(init=False)  # shape (m,)
    chi2_: float = field(init=False, default=np.nan)
    cov_theo_: np.ndarray = field(init=False)  # shape (p, p)

    # ------------------------------------------------
    # life-cycle helpers
    # ------------------------------------------------
    def __post_init__(self) -> None:
        p = len(self.Q_blocks)
        self.sigma_ = np.full(p, np.nan)
        self.residual_ = np.full(self.A.shape[0], np.nan)
        self.cov_theo_ = np.full((p, p), np.nan)

    # ------------------------------------------------
    # linear-system solver
    # ------------------------------------------------
    def _solve_system(self, N_mat: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
        """
        Solve **N σ = r** – the normal-equation system of a VCE iteration.

        A fallback to *least-squares* is provided if N is singular; a warning is
        raised if N is ill-conditioned (cond > ``self.singular_tol``).
        """
        if np.linalg.cond(N_mat) > self.singular_tol and self.verbose:
            logging.warning("Normal matrix is ill-conditioned; solution may be unstable")

        try:
            return np.linalg.solve(N_mat, r_vec)
        except np.linalg.LinAlgError:
            if self.verbose:
                logging.warning("Normal matrix singular - using np.linalg.lstsq")
            try:
                return np.linalg.lstsq(N_mat, r_vec, rcond=None)[0]
            except np.linalg.LinAlgError:
                if self.verbose:
                    logging.error("SVD failed - keeping previous σ")
                return self.sigma_

    # ------------------------------------------------
    # main driver
    # ------------------------------------------------
    def fit(self, y: np.ndarray, sigma0: Optional[Sequence[float]] = None) -> "VCEBase":
        """
        Iterate until

          ‖σ^{(t)} − σ^{(t−1)}‖_∞  ≤  ``tol``

        or *max_iter* is reached.  On convergence, residuals, χ² and the
        theoretical covariance of σ̂ are computed.
        """
        y = np.asarray(y, float)
        p = len(self.Q_blocks)
        self.sigma_ = np.ones(p) if sigma0 is None else np.asarray(sigma0, float)

        for it in range(1, self.max_iter + 1):
            self.n_iter_ = it
            σ_prev = self.sigma_.copy()

            # 1. build current Q_y
            Q_y = self.predict_Q()
            if np.linalg.cond(Q_y) > self.singular_tol:
                if self.verbose:
                    logging.warning(f"Iter {it:02d}: Q_y ill-conditioned – aborting")
                self.converged_ = False
                return self

            Q_inv = np.linalg.pinv(Q_y)
            _, P_perp = oblique_projector(self.A, Q_inv)
            e = P_perp @ y                                          # ê  (Eq. (50) TAS 2008)

            # 2. one VCE update
            self.sigma_ = self._update_step(e, P_perp, Q_inv)

            # 3. optional non-negativity enforcement
            if self.enforce_non_negative:
                self.sigma_ = np.maximum(self.sigma_, 0.0)

            # 4. convergence check
            if np.allclose(σ_prev, self.sigma_, rtol=self.tol, atol=self.tol):
                self.converged_ = True
                break
            if self.verbose:
                logging.info(f"Iter {it:02d}: σ = {self.sigma_}")

        if not self.converged_ and self.verbose:
            logging.warning("LS-VCE did *not* converge")

        # 5. post-processing on success
        if self.converged_:
            try:
                Q_inv = np.linalg.pinv(self.predict_Q())
                _, P_perp = oblique_projector(self.A, Q_inv)
                self.residual_ = P_perp @ y
                self.chi2_ = float(self.residual_ @ Q_inv @ self.residual_)  # Eq. (56) TAS 2008
                self.cov_theo_ = self._compute_covariance(P_perp, self.predict_Q())
            except (np.linalg.LinAlgError, ValueError):
                self.chi2_ = np.nan
                self.cov_theo_[:] = np.nan
        return self

    # ------------------------------------------------
    # utilities
    # ------------------------------------------------
    def predict_Q(self) -> np.ndarray:
        """
        Return current **Q_y = Q₀ + Σ σ_k Q_k**             # Eq. (2) TAS 2008
        """
        Q_y = sum(σ * Q for σ, Q in zip(self.sigma_, self.Q_blocks))
        return Q_y if self.Q0 is None else Q_y + self.Q0

    # ----- subclass requirements ---------------------------------------
    def _update_step(self, e: np.ndarray, P_perp: np.ndarray, Q_y_inv: np.ndarray) -> np.ndarray: ...
    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------
# BIQUE / REML / “optimal” LS-VCE
# ---------------------------------------------------------------------
@dataclass
class LSVCE(VCEBase):
    """
    BIQUE (= REML for κ = 0) implementation of LS-VCE.

    *Normal equations* (κ = 0, A-model) — TAS 2008 Eq. (60–61):

        r_k   = ½ [ êᵀ R Q_k R ê  −  tr(P_⊥ Q₀ R P_⊥ Q_k R) ]       # Eq. (60)

        N_kl  = ½ · tr(P_⊥ Q_k R P_⊥ Q_l R)                         # Eq. (61)

    with  R = Q_y⁻¹,  P_⊥ = I − P.
    """
    method: Literal["bique", "reml", "optimal_lsvce"] = "bique"

    # ------------------------------------------------
    # one BIQUE iteration
    # ------------------------------------------------
    def _update_step(self, e: np.ndarray, P_perp: np.ndarray, Q_inv: np.ndarray) -> np.ndarray:  # noqa: N803
        p = len(self.Q_blocks)
        Q0_eff = np.zeros_like(self.Q_blocks[0]) if self.Q0 is None else self.Q0

        N = np.empty((p, p))
        r = np.empty(p)

        for k, Qk in enumerate(self.Q_blocks):
            # ---- right-hand side r_k  (Eq. (60) TAS 2008) -----------
            term1 = e @ Q_inv @ Qk @ Q_inv @ e
            term2 = trace_of_product(P_perp, Q0_eff, Q_inv, P_perp, Qk, Q_inv)
            r[k] = 0.5 * (term1 - term2)

            # ---- normal matrix N_kl  (Eq. (61) TAS 2008) -----------
            for l, Ql in enumerate(self.Q_blocks[k:], start=k):  # exploit symmetry
                term = trace_of_product(P_perp, Qk, Q_inv, P_perp, Ql, Q_inv)
                N[k, l] = N[l, k] = 0.5 * term

        return self._solve_system(N, r)

    # ------------------------------------------------
    # asymptotic covariance  (Sec. 5 TAS 2008)
    # ------------------------------------------------
    def _compute_covariance(self, P_perp: np.ndarray, Q_y: np.ndarray) -> np.ndarray:
        Q_inv = np.linalg.pinv(Q_y)
        p = len(self.Q_blocks)

        N = np.empty((p, p))
        for k, Qk in enumerate(self.Q_blocks):
            for l, Ql in enumerate(self.Q_blocks[k:], start=k):
                term = trace_of_product(P_perp, Qk, Q_inv, P_perp, Ql, Q_inv)  # Eq. (61)
                N[k, l] = N[l, k] = 0.5 * term
        return np.linalg.pinv(N)

# ---------------------------------------------------------------------
# Helmert-type LS-VCE  (AS 2007, Sec. 3–4)  — configurable version
# ---------------------------------------------------------------------
@dataclass
class HelmertVCE(VCEBase):
    """
    Helmert-type LS-VCE estimator.

    A *group-diagonal* noise model (block-diagonal Q_k) admits a closed-form
    one-step solution (AS 2007 §3.4).  In finite samples that one-step
    covariance is often underestimated.  This class therefore exposes the
    boolean flag **one_step**:

    • one_step = False (default) – use *step-by-step* iterations until the
      standard LS-VCE convergence criterion is met;

    • one_step = True  – if the model is group-diagonal **and** all initial
      σ_k > 0, perform exactly one closed-form update and return.

    Update equations (AS 2007):

        q_k  = êᵀ E_k ê                                   (Eq. 3.17)
        H_kl = tr(P_⊥ᵀ E_k P_⊥ Q_l)                       (Eq. 3.18)
        Cov{σ̂} = H⁻¹                                     (Eq. 4.14)
    """
    method: Literal["helmert"] = "helmert"
    one_step: bool = False        # expose one-step mode to the user

    # ------------------------------------------------
    # helpers
    # ------------------------------------------------
    def _has_group_structure(self, tol: float = 1e-12) -> bool:
        """Return True if all Q_k are diagonal and non-overlapping (AS 2007 §3.4)."""
        m = self.Q_blocks[0].shape[0]
        occupied = np.zeros(m, dtype=bool)
        for Qk in self.Q_blocks:
            if not np.allclose(Qk - np.diag(np.diagonal(Qk)), 0.0, atol=tol):
                return False
            mask = np.abs(np.diagonal(Qk)) > tol
            if np.any(occupied & mask):
                return False
            occupied |= mask
        return True

    def _make_Ek(self, Q_inv: np.ndarray) -> Sequence[np.ndarray]:
        """
        Build derivative matrices E_k.

            group-diagonal & σ_k>0 :  E_k = σ_k⁻¹ Q_k⁻¹      (Eq. 3.11)
            otherwise              :  E_k = Q_y⁻¹ Q_k Q_y⁻¹  (Eq. 3.16)
        """
        if self._has_group_structure() and np.all(self.sigma_ > 0.0):
            return [np.linalg.pinv(Qk) / σ for σ, Qk in zip(self.sigma_, self.Q_blocks)]
        return [Q_inv @ Qk @ Q_inv for Qk in self.Q_blocks]

    # ------------------------------------------------
    # driver override
    # ------------------------------------------------
    def fit(
        self,
        y: np.ndarray,
        sigma0: Optional[Sequence[float]] = None,
    ) -> "HelmertVCE":
        """
        If ``one_step`` and the model is group-diagonal, run a single closed-form
        update; otherwise fall back to the generic iterative routine.
        """
        do_one_step = (
            self.one_step
            and self._has_group_structure()
            and (sigma0 is None or np.all(np.asarray(sigma0) > 0.0))
        )

        if do_one_step:
            old_max = self.max_iter
            self.max_iter = 1
            try:
                super().fit(y, sigma0)  # may early-exit on ill-conditioned Q_y
                # explicit residuals and theoretical covariance
                Q_y = self.predict_Q()
                Q_inv = np.linalg.pinv(Q_y)
                _, P_perp = oblique_projector(self.A, Q_inv)
                self.residual_ = P_perp @ y
                self.chi2_ = float(self.residual_ @ Q_inv @ self.residual_)
                self.cov_theo_ = self._compute_covariance(P_perp, Q_y)
                self.converged_ = True
            finally:
                self.max_iter = old_max
            return self

        # step-by-step (default)
        return super().fit(y, sigma0)

    # ------------------------------------------------
    # core mathematics
    # ------------------------------------------------
    def _update_step(
        self,
        e: np.ndarray,
        P_perp: np.ndarray,
        Q_inv: np.ndarray,
    ) -> np.ndarray:  # noqa: N803
        """Solve H σ = q for the current iterate (AS 2007 Eq. 3.17–3.18)."""
        p = len(self.Q_blocks)
        Ek = self._make_Ek(Q_inv)

        H = np.empty((p, p))
        q = np.empty(p)

        for k, E_k in enumerate(Ek):
            q[k] = e @ E_k @ e
            for l in range(k, p):
                Ql = self.Q_blocks[l]
                Hkl = np.trace(P_perp.T @ E_k @ P_perp @ Ql)
                H[k, l] = H[l, k] = Hkl
        return self._solve_system(H, q)

    def _compute_covariance(
        self,
        P_perp: np.ndarray,
        Q_y: np.ndarray,
    ) -> np.ndarray:
        """Return Cov{σ̂} = H⁻¹   (AS 2007 Eq. 4.14)."""
        Q_inv  = np.linalg.pinv(Q_y)
        E_list = self._make_Ek(Q_inv)
        p      = len(self.Q_blocks)

        H = np.empty((p, p))
        for k, Ek in enumerate(E_list):
            for l in range(k, p):
                Ql  = self.Q_blocks[l]
                Hkl = np.trace(P_perp.T @ Ek @ P_perp @ Ql)   # Eq. 3.18
                H[k, l] = H[l, k] = Hkl
        return np.linalg.pinv(H)                              # Eq. 4.14

