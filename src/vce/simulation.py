"""Monte Carlo utilities for comparing VCE estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Sequence, Optional, Any

import scipy.stats

import numpy as np

from vce.core import HelmertVCE, LSVCE, LSVCEPlus
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd


@dataclass
class Scenario:
    """Parameters defining a Monte Carlo run."""

    m: int
    r_dim: int
    block_sizes: Sequence[int]
    sigma_true: Sequence[float]
    n_trials: int = 100
    seed: Optional[int] = None


@dataclass
class Metrics:
    """Evaluation summary for one estimator."""

    bias: np.ndarray
    variance: np.ndarray
    rmse: np.ndarray
    var_ratio: np.ndarray
    cover95: np.ndarray
    chi2_p: float
    iter_mean: float
    fail_rate: float


def generate_design_matrix(m: int, r_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Return an ``m`` × ``r`` random design matrix."""
    return rng.standard_normal(size=(m, r_dim))


def generate_q_blocks(m: int, block_sizes: Iterable[int]) -> list[np.ndarray]:
    """Create diagonal blocks for the covariance model."""
    q_blocks: list[np.ndarray] = []
    cursor = 0
    for size in block_sizes:
        q = np.zeros((m, m))
        idx = slice(cursor, cursor + size)
        q[idx, idx] = np.eye(size)
        q_blocks.append(q)
        cursor += size
    return q_blocks


def simulate_y(
    A: np.ndarray,
    b: np.ndarray,
    q_blocks: Sequence[np.ndarray],
    sigma_true: Sequence[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a single observation vector."""
    q = sum(s * qk for s, qk in zip(sigma_true, q_blocks))
    return A @ b + rng.multivariate_normal(np.zeros(A.shape[0]), q)


def run_statsmodels(
    A: np.ndarray, y: np.ndarray, q_blocks: Sequence[np.ndarray]
) -> tuple[np.ndarray, float, bool]:
    """Return REML estimates using :mod:`statsmodels`."""
    m, r_dim = A.shape
    df = pd.DataFrame(A, columns=[f"x{i}" for i in range(r_dim)])
    df["y"] = y
    vc: dict[str, str] = {}
    for k, Qk in enumerate(q_blocks):
        grp = (np.diag(Qk) > 0).astype(int)
        df[f"grp{k}"] = grp
        vc[f"sigma{k}"] = f"0 + grp{k}"
    model = MixedLM.from_formula(
        "y ~ " + " + ".join(df.columns[:r_dim]),
        groups=pd.Series(np.ones(m)),
        vc_formula=vc,
        data=df,
    )
    try:
        res = model.fit(reml=True, method="lbfgs", disp=False)
        sigmas = np.array([res.vcomp[k] for k in range(len(q_blocks))])
        converged = bool(res.converged)
    except Exception:
        sigmas = np.full(len(q_blocks), np.nan)
        converged = False
    return sigmas, float("nan"), converged


def run_estimators(
    A: np.ndarray, q_blocks: Sequence[np.ndarray], y: np.ndarray
) -> Dict[str, Dict[str, np.ndarray | float | bool]]:
    """Fit all estimators and return statistics for one observation."""
    estimators = {
        "helmert": HelmertVCE(A, q_blocks),
        "lsvce": LSVCE(A, q_blocks),
        "lsvce_plus": LSVCEPlus(A, q_blocks),
    }
    out: Dict[str, Dict[str, np.ndarray | float | bool]] = {}
    for name, est in estimators.items():
        est.fit(y)
        out[name] = {
            "sigma": est.sigma_,
            "cov_theo": est.cov_theo_,
            "chi2": est.chi2_,
            "n_iter": float(est.n_iter_),
            "converged": bool(est.converged_),
        }
    sig, n_iter, conv = run_statsmodels(A, y, q_blocks)
    out["mixedlm"] = {
        "sigma": sig,
        "cov_theo": np.full((len(q_blocks), len(q_blocks)), np.nan),
        "chi2": np.nan,
        "n_iter": n_iter,
        "converged": conv,
    }
    return out


def monte_carlo(scn: Scenario) -> Dict[str, Dict[str, np.ndarray]]:
    """Run repeated simulations and collect estimator statistics."""
    rng = np.random.default_rng(scn.seed)
    A = generate_design_matrix(scn.m, scn.r_dim, rng)
    b = rng.standard_normal(scn.r_dim)
    q_blocks = generate_q_blocks(scn.m, scn.block_sizes)
    results: Dict[str, Dict[str, list]] = {
        name: {"sigma": [], "cov_theo": [], "chi2": [], "n_iter": [], "converged": []}
        for name in ("helmert", "lsvce", "lsvce_plus", "mixedlm")
    }
    for _ in range(scn.n_trials):
        y = simulate_y(A, b, q_blocks, scn.sigma_true, rng)
        for name, vals in run_estimators(A, q_blocks, y).items():
            for metric, val in vals.items():
                results[name][metric].append(val)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, data in results.items():
        out[name] = {
            "sigma": np.vstack(data["sigma"]),
            "cov_theo": np.stack(data["cov_theo"]),
            "chi2": np.array(data["chi2"], float),
            "n_iter": np.array(data["n_iter"], float),
            "converged": np.array(data["converged"], bool),
        }
    return out


def evaluate(
    results: Dict[str, Dict[str, np.ndarray]],
    sigma_true: Sequence[float],
    m: int,
    r_dim: int,
) -> Dict[str, Metrics]:
    """Compute statistical summaries from Monte Carlo results."""
    true = np.asarray(sigma_true, float)
    summary: Dict[str, Metrics] = {}
    dof = m - r_dim
    for name, data in results.items():
        sigmas = data["sigma"]
        cov_theo = data["cov_theo"]
        chi2_vals = data["chi2"]
        n_iter = data["n_iter"]
        converged = data["converged"]

        mean_est = sigmas.mean(axis=0)
        bias = mean_est - true
        emp_cov = np.cov(sigmas.T, ddof=1)
        var = np.diag(emp_cov)
        rmse = np.sqrt(bias**2 + var)
        cov_mean = cov_theo.mean(axis=0)
        var_ratio = var / np.diag(cov_mean)
        ci = 1.96 * np.sqrt(np.diag(cov_mean))
        cover = ((true >= mean_est - ci) & (true <= mean_est + ci)).mean(axis=0)
        chi2_p = float(1.0 - scipy.stats.chi2.cdf(chi2_vals.mean(), dof))
        iter_mean = float(n_iter.mean())
        fail_rate = float(1.0 - converged.mean())
        summary[name] = Metrics(
            bias=bias,
            variance=var,
            rmse=rmse,
            var_ratio=var_ratio,
            cover95=cover,
            chi2_p=chi2_p,
            iter_mean=iter_mean,
            fail_rate=fail_rate,
        )
    return summary


def plot_cov_ratio(
    results: Dict[str, Dict[str, np.ndarray]],
    name: str,
    title: str = "",
) -> Any:
    """Return a heatmap of empirical/theoretical covariance ratio."""
    import plotly.express as px

    sigmas = results[name]["sigma"]
    cov_theo_mean = results[name]["cov_theo"].mean(axis=0)
    emp_cov = np.cov(sigmas.T, ddof=1)
    ratio = emp_cov / cov_theo_mean
    fig = px.imshow(ratio, color_continuous_scale="Viridis", text_auto=True)
    fig.update_layout(title=title or f"Covariance ratio {name}")
    return fig
