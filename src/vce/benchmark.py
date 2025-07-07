#!/usr/bin/env python3
"""benchmark.py — concise Monte‑Carlo benchmark with *summary metrics*

Runs a reduced‑size Monte‑Carlo experiment for several (m, r_dim) scenarios,
computes the full `Metrics` suite (bias, variance, χ² p‑value …) for every
estimator, and saves a single CSV whose **one row = one estimator × scenario**.

Key changes v.s. the old per‑trial logger
----------------------------------------
* **Much lighter workload**
  * scenarios: m ∈ [30, 60, 120]  (max matrix 120×120)
  * default ``--trial-base 50``        → ≤ 7 500 total trials
* **Summary instead of raw trials** – easier to read; one glance shows which
  estimator is best.
* Pure *map‑reduce* layout – no shared queue needed; keeps Windows compatibility
  without the previous Manager/Queue plumbing.

Usage examples
--------------
```bash
python benchmark.py                            # default lightweight run
python benchmark.py --trial-base 200 -p 6      # deeper stats, 6 processes
```
The script prints a Markdown‑ready table to stdout **and** writes a CSV
(`vce_summary.csv` by default).
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

from vce.simulation import (
    Scenario,
    monte_carlo,
    evaluate,
)

# ---------------------------------------------------------------------------
# Scenario generator (lightweight) ------------------------------------------

def make_scenarios(trial_base: int) -> List[Scenario]:
    """Return a small list of Scenario objects with scaled trial counts."""
    m_list = [30, 60, 120]              # max 120×120 ⇒ ultra‑fast
    scenarios: List[Scenario] = []
    for m in m_list:
        r_dim = max(2, round(m / 30))
        block = (m // 3, m // 3, m - 2 * (m // 3))
        trials = trial_base * (120 // m)  # keep ~constant total obs per m
        scenarios.append(
            Scenario(
                m=m,
                r_dim=r_dim,
                block_sizes=block,
                sigma_true=(5.0, 2.0, 1.0),
                n_trials=trials,
                seed=m,  # deterministic but distinct
            )
        )
    return scenarios

# ---------------------------------------------------------------------------
# Worker – run one Scenario, summarise Metrics ------------------------------

def scenario_worker(scn: Scenario) -> Dict[str, Any]:
    """Run monte‑carlo + evaluation for one Scenario; return serialisable dict."""
    results = monte_carlo(scn)
    metrics = evaluate(results, scn.sigma_true, scn.m, scn.r_dim)
    out: Dict[str, Any] = {
        "m": scn.m,
        "r_dim": scn.r_dim,
        "block": scn.block_sizes,
    }
    # flatten every Metrics dataclass into separate keys
    for est_name, met in metrics.items():
        d = asdict(met)
        # numpy arrays → python lists for csv/json friendliness
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        out[est_name] = d
    return out

# ---------------------------------------------------------------------------
# CSV helpers ---------------------------------------------------------------

METRIC_FIELDS = [
    "bias", "variance", "rmse", "var_ratio", "cover95",
    "chi2_p", "iter_mean", "fail_rate",
]

CSV_HEADER: List[str] = (
    ["estimator", "m", "r_dim", "block1", "block2", "block3"] +
    [f"{fld}{i+1}" for fld in ["bias", "var", "rmse", "ratio", "cov"] for i in range(3)] +
    ["chi2_p", "iter_mean", "fail_rate"]
)

def metrics_to_row(est_name: str, scenario_info: Dict[str, Any]) -> List[Any]:
    """Flatten Metrics dict of a single estimator into a CSV row."""
    m = scenario_info["m"]
    r_dim = scenario_info["r_dim"]
    b1, b2, b3 = scenario_info["block"]
    met = scenario_info[est_name]
    row = [est_name, m, r_dim, b1, b2, b3]
    # order: bias/var/rmse/ratio/cover (×3 each)
    row.extend(sum([met["bias"], met["variance"], met["rmse"], met["var_ratio"], met["cover95"]], []))
    row.extend([met["chi2_p"], met["iter_mean"], met["fail_rate"]])
    return row

# ---------------------------------------------------------------------------
# Pretty print helper -------------------------------------------------------

def print_table(scenarios_data: List[Dict[str, Any]]):
    """Dump a concise Markdown table to stdout."""
    from tabulate import tabulate  # lightweight dep (ships with tqdm env)

    rows = []
    for scn in scenarios_data:
        for est in ("helmert", "lsvce", "lsvce_plus"):
            met = scn[est]
            rows.append([
                est,
                scn["m"],
                f"{met['bias'][0]:+.2f} ± {np.sqrt(met['variance'][0]):.2f}",
                f"{met['bias'][1]:+.2f} ± {np.sqrt(met['variance'][1]):.2f}",
                f"{met['bias'][2]:+.2f} ± {np.sqrt(met['variance'][2]):.2f}",
                f"{met['chi2_p']:.2f}",
                f"{met['fail_rate']*100:.1f}%",
            ])
    print(tabulate(rows, headers=[
        "est", "m", "σ1 (bias±sd)", "σ2", "σ3", "χ² p", "fail"
    ], tablefmt="github"))

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("VCE Monte‑Carlo benchmark (summary)")
    parser.add_argument("--trial-base", type=int, default=300,
                        help="Base #trials for smallest scenario (default=50)")
    parser.add_argument("-p", "--processes", type=int,
                        default=mp.cpu_count(),
                        help="Worker processes (default=min(4,cpu))")
    parser.add_argument("--outfile", type=Path, default=Path("vce_summary.csv"))
    args = parser.parse_args()

    scenarios = make_scenarios(args.trial_base)

    with mp.Pool(args.processes) as pool:
        data = list(tqdm(pool.imap_unordered(scenario_worker, scenarios),
                         total=len(scenarios)))

    # write summary CSV ------------------------------------------------------
    with args.outfile.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for scn in data:
            for est in ("helmert", "lsvce", "lsvce_plus"):
                w.writerow(metrics_to_row(est, scn))
    print("Saved summary to", args.outfile)

    # print nice table -------------------------------------------------------
    print_table(data)
