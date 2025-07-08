#!/usr/bin/env python3
"""benchmark.py — extended Monte‑Carlo benchmark with **rich metrics**

* 4 lightweight scenarios: m ∈ {12, 30, 60, 120}
* Per‑scenario trials scaled so total observations stay balanced.
* Computes expanded Metrics (bias, **sd**, var, rmse, **mse**, var‑ratio,
  cover95, χ² p, mean/median/max iterations, fail‑rate).
* Outputs **one CSV row = one estimator × scenario**, plus a Markdown table
  on stdout for a quick glance.

Typical usage
-------------
```bash
python benchmark.py              # quick run (trial‑base 75 × 4 scenarios)
python benchmark.py -p 6 --trial-base 150  # deeper stats, 6 CPU cores
```
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import numpy as np
from tqdm import tqdm

from vce.simulation import Scenario, monte_carlo

# ---------------------------------------------------------------------------
# Scenario generator ---------------------------------------------------------


def make_scenarios(trial_base: int) -> List[Scenario]:
    """Return a list of Scenario objects with balanced trial counts."""
    m_list = [12, 30, 60, 90]
    out: List[Scenario] = []
    for m in m_list:
        r_dim = max(2, round(m / 3 - 20))
        block = (m // 3, m // 3, m - 2 * (m // 3))
        trials = trial_base
        out.append(
            Scenario(
                m=m,
                r_dim=r_dim,
                block_sizes=block,
                sigma_true=(5.0, 2.0, 1),
                n_trials=trials,
                seed=m,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Metric helpers -------------------------------------------------------------


def calc_metrics(
    res: Dict[str, Dict[str, np.ndarray]], true_sigma: List[float], m: int, r_dim: int
) -> Dict[str, Dict[str, Any]]:
    """Compute extended metrics for each estimator, tolerate NaNs."""
    import scipy.stats

    true = np.asarray(true_sigma, float)
    dof = m - r_dim
    out: Dict[str, Dict[str, Any]] = {}
    for est, data in res.items():
        sig = data["sigma"]
        conv = data["converged"]
        ok = conv & np.isfinite(sig).all(axis=1)
        if not ok.any():
            nan3 = [np.nan, np.nan, np.nan]
            out[est] = dict(
                bias=nan3,
                sd=nan3,
                variance=nan3,
                rmse=nan3,
                mse=nan3,
                var_ratio=nan3,
                cover95=nan3,
                chi2_p=np.nan,
                iter_mean=np.nan,
                iter_median=np.nan,
                iter_max=np.nan,
                fail_rate=1.0,
            )
            continue
        sig_ok = sig[ok]
        cov_theo_ok = data["cov_theo"][ok]
        chi2_vals = data["chi2"][ok]
        n_iter = data["n_iter"][ok]

        mean_est = sig_ok.mean(axis=0)
        bias = mean_est - true
        emp_cov = np.cov(sig_ok.T, ddof=1)
        var = np.diag(emp_cov)
        sd = np.sqrt(var)
        rmse = np.sqrt(bias**2 + var)
        mse = bias**2 + var

        if cov_theo_ok.size:
            cov_mean = cov_theo_ok.mean(axis=0)
            var_ratio = var / np.diag(cov_mean)
            ci = 1.96 * np.sqrt(np.diag(cov_mean))
            cover = (true >= mean_est - ci) & (true <= mean_est + ci)
        else:
            var_ratio = cover = np.full_like(true, np.nan)

        chi2_p = float(1.0 - scipy.stats.chi2.cdf(chi2_vals.mean(), dof))
        out[est] = dict(
            bias=bias.tolist(),
            sd=sd.tolist(),
            variance=var.tolist(),
            rmse=rmse.tolist(),
            mse=mse.tolist(),
            var_ratio=var_ratio.tolist(),
            cover95=cover.tolist(),
            chi2_p=chi2_p,
            iter_mean=float(n_iter.mean()),
            iter_median=float(np.median(n_iter)),
            iter_max=float(n_iter.max()),
            fail_rate=float(1.0 - ok.mean()),
        )
    return out


# ---------------------------------------------------------------------------
# Worker ---------------------------------------------------------------------


def scenario_worker(scn: Scenario) -> Dict[str, Any]:
    t0 = time.time()
    res = monte_carlo(scn)
    met = calc_metrics(res, list(scn.sigma_true), scn.m, scn.r_dim)
    dt = time.time() - t0

    info: Dict[str, Any] = {
        "m": scn.m,
        "r_dim": scn.r_dim,
        "block": scn.block_sizes,
        "runtime": dt,
    }
    info.update(met)
    return info


# ---------------------------------------------------------------------------
# CSV helpers ----------------------------------------------------------------



def _csv_header() -> List[str]:
    cols = [

        "estimator",
        "m",
        "r_dim",
        "block1",
        "block2",
        "block3",
    ]

    for pfx in (
        "bias",
        "sd",
        "variance",
        "rmse",
        "mse",
        "var_ratio",
        "cover95",
    ):
        cols.extend(f"{pfx}{i}" for i in range(1, 4))
    cols.extend(
        [
            "chi2_p",
            "iter_mean",
            "iter_median",
            "iter_max",
            "fail_rate",
            "scenario_runtime",
        ]
    )
    return cols



def metrics_to_row(est: str, scn: Dict[str, Any]) -> List[Any]:
    """Flatten one estimator's metrics into a CSV row, robust to scalars."""
    b1, b2, b3 = scn["block"]
    m = scn[est]
    row: List[Any] = [est, scn["m"], scn["r_dim"], b1, b2, b3]
    for key in ("bias", "sd", "variance", "rmse", "mse", "var_ratio", "cover95"):
        v = m[key]
        # ensure iterable
        if isinstance(v, (list, tuple)):
            row.extend(v)
        else:
            row.append(v)
    row.extend(
        [
            m["chi2_p"],
            m["iter_mean"],
            m["iter_median"],
            m["iter_max"],
            m["fail_rate"],
            scn["runtime"],
        ]
    )
    return row



def scenarios_to_df(datas: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [
        metrics_to_row(est, d)
        for d in datas
        for est in ("helmert", "lsvce", "lsvce_plus")
    ]
    return pd.DataFrame(rows, columns=_csv_header())


# ---------------------------------------------------------------------------
# Pretty table ---------------------------------------------------------------



def print_table(datas: List[Dict[str, Any]]) -> None:
    df = scenarios_to_df(datas)
    table = pd.DataFrame(
        {
            "est": df["estimator"],
            "m": df["m"],
            "σ1 (bias±sd)": [
                f"{b:+.2f} ± {s:.2f}" for b, s in zip(df["bias1"], df["sd1"])
            ],
            "σ2": [f"{b:+.2f} ± {s:.2f}" for b, s in zip(df["bias2"], df["sd2"])],
            "σ3": [f"{b:+.2f} ± {s:.2f}" for b, s in zip(df["bias3"], df["sd3"])],
            "χ² p": df["chi2_p"].map(lambda x: f"{x:.2f}"),
            "fail": (df["fail_rate"] * 100).map(lambda x: f"{x:.1f}%"),
            "iter¯": df["iter_mean"].map(lambda x: f"{x:.1f}"),
        }
    )
    try:
        print(table.to_markdown(index=False))
    except Exception:
        print(table.to_string(index=False))


# ---------------------------------------------------------------------------
# Main -----------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trial-base",
        type=int,
        default=10000,
        help="Base trial count for smallest m (default 75)",
    )
    ap.add_argument(
        "-p",
        "--processes",
        type=int,
        default=mp.cpu_count(),
        help="Worker processes (default=min(4,cpu))",
    )
    ap.add_argument("--outfile", type=Path, default="vce_lsummary.csv")
    args = ap.parse_args()

    scens = make_scenarios(args.trial_base)
    with mp.Pool(args.processes) as pool:
        data = list(tqdm(pool.imap_unordered(scenario_worker, scens), total=len(scens)))

    scenarios_to_df(data).to_csv(args.outfile, index=False)

    print("Saved summary to", args.outfile)

    print_table(data)
