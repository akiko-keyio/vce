# vce/benchmark.py ─ Stress-benchmark for HelmertVCE & LSVCE
# =========================================================
"""
Run a battery of Monte-Carlo scenarios (defined on-the-fly) with
multiprocessing support and generate:
  • CSV summary   (metrics per estimator × scenario)
  • HTML report   (interactive Plotly figs)

Usage
-----
python -m vce.benchmark --trial-base 50 -p 8 \
       --outfile results.csv --report report.html
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from vce.simulation import Scenario, monte_carlo

# ------------------------------------------------------------------ #
# 1. Scenario factory
# ------------------------------------------------------------------ #
def make_scenarios(trial_base: int) -> List[Scenario]:
    """Return a diverse list of Scenario objects (≥ real-world coverage)."""
    m_default, r_default = 60, 4

    scns: List[Scenario] = [
        Scenario("1. Base", m_default, r_default, (5., 2., 1.),
                 n_trials=trial_base, seed=1),
        Scenario("2. VarRatio≫1", m_default, r_default, (100., 1., .1),
                 n_trials=trial_base, seed=2),
        Scenario("3. NearZeroComp", m_default, r_default, (5., 2., 1e-6),
                 n_trials=trial_base, seed=3),
        Scenario("4. OverlapQ", m_default, r_default, (5., 2., 1.),
                 n_trials=trial_base, q_structure="overlapping", seed=4),
        Scenario("5. IllCondA", m_default, r_default, (5., 2., 1.),
                 n_trials=trial_base, A_cond=1e3, seed=5),
        Scenario("6. Outliers", m_default, r_default, (5., 2., 1.),
                 n_trials=trial_base,
                 outlier_params={"fraction": .05, "magnitude": 8.}, seed=6),
        Scenario("7. WithQ0", m_default, r_default, (5., 2., 1.),
                 n_trials=trial_base, q0_factor=.5, seed=7),
    ]

    # ─ m-scaling --------------------------------------------------------
    for m_val in range(10, 110, 10):
        scns.append(
            Scenario(f"8. Scale_m={m_val}", m_val, 5,
                     (5., 2., 1.), n_trials=trial_base, seed=100 + m_val)
        )
    # ─ redundancy-scaling ----------------------------------------------
    m_fixed = 90
    for r_val in (5, 45, 80):
        dof = m_fixed - r_val
        scns.append(
            Scenario(f"9. DOF={dof}", m_fixed, r_val,
                     (5., 2., 1.), n_trials=trial_base, seed=200 + r_val)
        )
    return scns


# ------------------------------------------------------------------ #
# 2.   Metric calculation
# ------------------------------------------------------------------ #
def calc_metrics(res: Dict[str, Dict[str, np.ndarray]],
                 scn: Scenario) -> Dict[str, Dict[str, Any]]:
    """Return per-estimator statistics w.r.t. *true* σ."""
    import scipy.stats  # local to keep global import footprint small

    σ_true = np.asarray(scn.sigma_true, float)
    p      = len(σ_true)
    dof    = scn.m - scn.r_dim

    # handy containers ---------------------------------------------------
    nan_vec = np.full(p, np.nan)
    nan_metrics: Dict[str, Any] = {
        "bias": nan_vec, "median_bias": nan_vec, "sd": nan_vec,
        "var_ratio": nan_vec, "cover95": nan_vec,
        "chi2_p": np.nan, "iter_mean": np.nan, "iter_median": np.nan,
        "iter_max": np.nan,
    }

    out: Dict[str, Dict[str, Any]] = {}
    for est, dat in res.items():
        σ_hat   = dat["sigma"]
        conv    = dat["converged"]
        neg_var = np.mean((σ_hat < -1e-9).any(axis=1)) if σ_hat.size else 0.0

        ok = conv & np.isfinite(σ_hat).all(axis=1)
        if not ok.any():
            out[est] = {**nan_metrics,
                        "fail_rate": 1.0, "neg_var_rate": neg_var}
            continue

        σ_ok       = σ_hat[ok]
        cov_theo   = dat["cov_theo"][ok]
        χ2_vals    = dat["chi2"][ok]
        n_iter     = dat["n_iter"][ok]

        mean_est   = σ_ok.mean(axis=0)
        med_est    = np.median(σ_ok, axis=0)
        var        = np.diag(np.cov(σ_ok.T, ddof=1)) if σ_ok.shape[0] > 1 \
                     else np.zeros_like(σ_true)
        sd         = np.sqrt(var)

        # ─ variance ratio ----------------------------------------------
        diag_cov   = np.diagonal(cov_theo, axis1=1, axis2=2)  # (N,p)
        cov_mean   = diag_cov.mean(axis=0)
        var_ratio  = var / (cov_mean + 1e-12)

        # ─ coverage -----------------------------------------------------
        ci_hw      = 1.96 * np.sqrt(np.clip(diag_cov, 0.0, None))  # safe sqrt
        cover      = ((σ_true >= (σ_ok - ci_hw)) &
                      (σ_true <= (σ_ok + ci_hw))).mean(axis=0)

        χ2_p       = (1.0 - scipy.stats.chi2.cdf(χ2_vals.mean(), dof)
                      if χ2_vals.size and dof > 0 else np.nan)

        out[est] = {
            "bias": mean_est - σ_true,
            "median_bias": med_est - σ_true,
            "sd": sd,
            "var_ratio": var_ratio,
            "cover95": cover,
            "chi2_p": χ2_p,
            "neg_var_rate": neg_var,
            "fail_rate": 1.0 - ok.mean(),
            "iter_mean": n_iter.mean(),
            "iter_median": np.median(n_iter),
            "iter_max": n_iter.max(),
        }
    return out


# ------------------------------------------------------------------ #
# 3.   Worker (runs inside multiprocessing Pool)
# ------------------------------------------------------------------ #
def scenario_worker(scn: Scenario) -> pd.DataFrame:
    t0     = time.time()
    res    = monte_carlo(scn)
    mets   = calc_metrics(res, scn)
    rows   = []
    base   = asdict(scn)
    base.pop("sigma_true")
    base["runtime_s"] = time.time() - t0

    for est, m in mets.items():
        rec = {"estimator": est, **base}
        # vector metrics → separate cols (bias1, bias2, …)
        for key, val in m.items():
            if isinstance(val, np.ndarray):
                for i, x in enumerate(val, 1):
                    rec[f"{key}{i}"] = float(x)
            else:
                rec[key] = float(val)
        for i, σ in enumerate(scn.sigma_true, 1):
            rec[f"true_sigma{i}"] = σ
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# 4.   Visualisation helper
# ------------------------------------------------------------------ #
def analyze_and_visualize(df: pd.DataFrame, report: Path) -> None:
    """Generate an interactive HTML summary (Plotly)."""
    figs: List[go.Figure] = []
    est_colors = {"helmert": "royalblue", "lsvce": "firebrick"}

    # -------- helper: derive p_max & col groups ------------------------
    p_max = max(int(c.split("sd")[-1]) for c in df.columns if c.startswith("sd"))
    comp_labels = [f"σ{i}" for i in range(1, p_max + 1)]

    # -------- Fig 1: Precision ----------------------------------------
    melt_sd = df.melt(id_vars=["name", "estimator"],
                      value_vars=[f"sd{i}" for i in range(1, p_max + 1)],
                      var_name="comp", value_name="sd")
    melt_sd["comp"] = melt_sd["comp"].str.replace("sd", "σ")
    fig1 = go.Figure()
    for est in est_colors:
        d = melt_sd[melt_sd["estimator"] == est]
        fig1.add_trace(go.Box(y=d["sd"], x=d["comp"],
                              name=est, marker_color=est_colors[est],
                              boxpoints="all", jitter=0.3))
    fig1.update_layout(title="Figure 1 – Empirical Std. Dev. of σ̂",
                       yaxis_title="Std-Dev")
    figs.append(fig1)

    # -------- Fig 2: Variance ratio -----------------------------------
    melt_vr = df.melt(id_vars=["name", "estimator"],
                      value_vars=[f"var_ratio{i}" for i in range(1, p_max + 1)],
                      var_name="comp", value_name="vr")
    melt_vr["comp"] = melt_vr["comp"].str.replace("var_ratio", "σ")
    fig2 = go.Figure()
    for est in est_colors:
        d = melt_vr[melt_vr["estimator"] == est]
        fig2.add_trace(go.Box(y=d["vr"], x=d["comp"], name=est,
                              marker_color=est_colors[est],
                              boxpoints="all", jitter=0.3))
    fig2.add_hline(y=1.0, line_dash="dash",
                   line_color="black",
                   annotation_text="Ideal = 1")
    fig2.update_layout(title="Figure 2 – Empirical / Theoretical Variance",
                       yaxis_type="log", yaxis_title="Ratio")
    figs.append(fig2)

    # -------- Fig 3: Failure rate -------------------------------------
    df_fr = df.copy()
    df_fr["fail_pct"] = df_fr["fail_rate"] * 100
    fig3 = go.Figure()
    for est in est_colors:
        d = df_fr[df_fr["estimator"] == est]
        fig3.add_trace(go.Bar(name=est, x=d["name"], y=d["fail_pct"],
                              marker_color=est_colors[est]))
    fig3.update_layout(title="Figure 3 – Failure Rate (%)",
                       yaxis_title="% failures", barmode="group")
    figs.append(fig3)

    # -------- HTML export ---------------------------------------------
    with open(report, "w", encoding="utf-8") as f:
        f.write("<html><head><title>VCE Benchmark</title></head>"
                "<body style='font-family: sans-serif'>")
        f.write("<h1>VCE Estimator Benchmark Report</h1>")
        for i, fig in enumerate(figs, 1):
            f.write(f"<h2>Figure {i}</h2>")
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</body></html>")
    print(f"📊 Report saved → {report.absolute()}")


# ------------------------------------------------------------------ #
# 5.   CLI entry-point
# ------------------------------------------------------------------ #
def _main() -> None:
    ap = argparse.ArgumentParser(description="Monte-Carlo benchmark for HelmertVCE & LSVCE")
    ap.add_argument("--trial-base", type=int, default=1000,
                    help="Base number of trials per scenario")
    ap.add_argument("-p", "--processes", type=int,
                    default=max(1, mp.cpu_count() - 1),
                    help="Parallel processes (default: CPU-1)")
    ap.add_argument("--outfile", type=Path, default=Path("vce_benchmark.csv"),
                    help="CSV output path")
    ap.add_argument("--report", type=Path, default=Path("vce_benchmark.html"),
                    help="HTML report path")
    args = ap.parse_args()

    scns = make_scenarios(args.trial_base)
    print(f"🚀 Running {len(scns)} scenarios × {args.trial_base} trials …")

    with mp.Pool(args.processes) as pool:
        df_list = list(tqdm(pool.imap_unordered(scenario_worker, scns),
                            total=len(scns), desc="Scenarios"))

    summary = pd.concat(df_list, ignore_index=True)
    summary.to_csv(args.outfile, index=False)
    print(f"✅ CSV summary → {args.outfile.absolute()}")

    analyze_and_visualize(summary, args.report)

    # --- console glance -----------------------------------------------
    p_cols = [c for c in summary.columns if c.startswith("sd")]
    view   = summary.pivot_table(index="name", columns="estimator",
                                 values=p_cols + ["var_ratio1", "fail_rate"])
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\nQuick view (Std-Dev & Var-ratio, first comp + failure rate):")
        print(view[[("sd1",  "helmert"), ("sd1",  "lsvce"),
                    ("var_ratio1", "helmert"), ("var_ratio1", "lsvce"),
                    ("fail_rate",  "helmert"), ("fail_rate",  "lsvce")]].round(3))


if __name__ == "__main__":
    _main()
