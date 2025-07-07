#!/usr/bin/env python3
"""Lightweight benchmark runner for VCE Monte‑Carlo tests.

Compared with the previous *benchmark.py*, this script **trades breadth for
speed** so that a full run finishes in ≲ 3 min on a laptop (≈ 8 cores).

Main simplifications
--------------------
* **Scenario grid** reduced to 2×2×2 = 8 combos:
  m ∈ {120, 240}, r ∈ {4, 6}, block pattern ∈ {"333", "336"}.
  (Blocks "246" & SNR scaling removed.)
* **Trials per scenario** default 300 — adequate to see bias & variance order.
* **Parallel limit** automatically caps at min(4, CPU) unless user overrides.
* Added **--quick** preset: runs *only 120×4* with 100 trials (~5 s) for CI.

Usage examples
--------------
    # Normal lightweight run (~ 1 min)
    python -m vce.benchmark_light

    # Quick sanity check (single scenario) – finishes in seconds
    python -m vce.benchmark_light --quick

    # Custom trials & workers
    python -m vce.benchmark_light --trials 500 --processes 6
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from itertools import product
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from vce.simulation import Scenario, monte_carlo, evaluate

# ---------------------------------------------------------------------------
# Scenario factory -----------------------------------------------------------


def make_scenarios(n_trials: int, seeds: List[int], quick: bool) -> List[Scenario]:
    if quick:
        return [
            Scenario(
                m=120,
                r_dim=4,
                block_sizes=(40, 40, 40),
                sigma_true=(5.0, 2.0, 1.0),
                n_trials=n_trials,
                seed=seeds[0],
            )
        ]

    m_choices = [120, 240]
    r_choices = [4, 6]
    block_patterns = {"333": (40, 40, 40), "336": (30, 30, 60)}
    sigma_true = np.array([5.0, 2.0, 1.0])

    scenarios: List[Scenario] = []
    for (m, r_dim, pattern_key, seed) in product(
        m_choices, r_choices, block_patterns, seeds
    ):
        scenarios.append(
            Scenario(
                m=m,
                r_dim=r_dim,
                block_sizes=block_patterns[pattern_key],
                sigma_true=sigma_true,
                n_trials=n_trials,
                seed=seed,
            )
        )
    return scenarios

# ---------------------------------------------------------------------------
# Worker ---------------------------------------------------------------------


def run_one(scn: Scenario):
    res = monte_carlo(scn)
    metrics = evaluate(res, scn.sigma_true, scn.m, scn.r_dim)
    return scn, metrics

# ---------------------------------------------------------------------------
# Main -----------------------------------------------------------------------


def main():
    default_proc = min(4, mp.cpu_count())
    parser = argparse.ArgumentParser(description="Run lightweight VCE benchmarks")
    parser.add_argument("--trials", type=int, default=300, help="Monte-Carlo trials per scenario")
    parser.add_argument("--processes", type=int, default=default_proc, help="Parallel processes")
    parser.add_argument("--quick", action="store_true", help="Single‑scenario quick run (100 trials)")
    parser.add_argument("--outfile", type=Path, default=Path("vce_benchmark_light.csv"), help="CSV output path")
    args = parser.parse_args()

    if args.quick:
        args.trials = min(args.trials, 100)

    seeds = list(range(100, 100 + args.processes))
    scenarios = make_scenarios(args.trials, seeds, args.quick)

    with mp.Pool(processes=args.processes) as pool:
        results = list(tqdm(pool.imap_unordered(run_one, scenarios), total=len(scenarios)))

    with args.outfile.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "estimator",
                "m",
                "r_dim",
                "blocks",
                "trials",
                "iter_mean",
                "fail_rate",
                "bias1",
                "bias2",
                "bias3",
                "rmse1",
                "rmse2",
                "rmse3",
                "chi2_p",
            ]
        )
        for scn, metrics in results:
            for est_name, m in metrics.items():
                writer.writerow(
                    [
                        est_name,
                        scn.m,
                        scn.r_dim,
                        ",".join(map(str, scn.block_sizes)),
                        scn.n_trials,
                        f"{m.iter_mean:.2f}",
                        f"{m.fail_rate:.3f}",
                        *[f"{x:.4f}" for x in m.bias],
                        *[f"{x:.4f}" for x in m.rmse],
                        f"{m.chi2_p:.4f}",
                    ]
                )
    print(f"Done. Results -> {args.outfile}")


if __name__ == "__main__":
    main()
