#!/usr/bin/env python3
"""Top‑level benchmark runner for large‑scale VCE Monte‑Carlo tests.

Usage (default params):
    python -m vce.benchmark

Optional CLI flags (see `--help`):
    --trials 2000               # number of MC trials per scenario
    --processes 8               # parallel workers
    --outfile results.csv       # where to save aggregated metrics

The script constructs a *grid* of test scenarios that exercise different
matrix sizes, signal‑to‑noise ratios, and covariance block patterns.  It then
uses multiprocessing to run them in parallel, leveraging `vce.simulation`.
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


def make_scenarios(n_trials: int, seeds: List[int]) -> List[Scenario]:
    """Build a grid of scenarios with varying dimensions & variances."""
    m_choices = [120, 240]            # number of observations
    r_choices = [4, 6]                # functional params
    block_patterns = {
        "333": (40, 40, 40),         # 3 equal blocks
        "336": (30, 30, 60),
        "246": (20, 40, 60),
    }
    base_sigma = np.array([5.0, 2.0, 1.0])
    scale = [0.5, 1.0, 2.0]           # SNR multipliers

    scenarios: List[Scenario] = []
    for (m, r_dim, pattern_key, s, seed) in product(
        m_choices, r_choices, block_patterns, scale, seeds
    ):
        block_sizes = block_patterns[pattern_key]
        sigma_true = base_sigma * s
        scenarios.append(
            Scenario(
                m=m,
                r_dim=r_dim,
                block_sizes=block_sizes,
                sigma_true=sigma_true,
                n_trials=n_trials,
                seed=seed,
            )
        )
    return scenarios

# ---------------------------------------------------------------------------
# Worker ---------------------------------------------------------------------


def run_one(scn: Scenario):
    """Run MC + evaluation for a single scenario (multiprocessing safe)."""
    res = monte_carlo(scn)
    metrics = evaluate(res, scn.sigma_true, scn.m, scn.r_dim)
    return scn, metrics

# ---------------------------------------------------------------------------
# Main -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run large‑scale VCE benchmarks")
    parser.add_argument("--trials", type=int, default=1000, help="Monte‑Carlo trials per scenario")
    parser.add_argument("--processes", type=int, default=mp.cpu_count(), help="Parallel processes")
    parser.add_argument("--outfile", type=Path, default=Path("vce_benchmark.csv"), help="CSV output path")
    args = parser.parse_args()

    seeds = list(range(100, 100 + args.processes))  # distinct seeds per scenario row
    scenarios = make_scenarios(args.trials, seeds)

    with mp.Pool(processes=args.processes) as pool:
        results = list(tqdm(pool.imap_unordered(run_one, scenarios), total=len(scenarios)))

    # ---------------------------------------------------------------------
    # Flatten & write CSV --------------------------------------------------
    # ---------------------------------------------------------------------
    header_written = False
    with args.outfile.open("w", newline="") as f:
        writer = csv.writer(f)

        for scn, metrics in results:
            for est_name, m in metrics.items():
                row = [
                    est_name,
                    scn.m,
                    scn.r_dim,
                    ",".join(map(str, scn.block_sizes)),
                    ",".join(map("{:.3f}".format, scn.sigma_true)),
                    scn.n_trials,
                    m.iter_mean,
                    m.fail_rate,
                    *m.bias,
                    *m.rmse,
                    *m.var_ratio,
                    *m.cover95,
                    m.chi2_p,
                ]
                if not header_written:
                    writer.writerow(
                        [
                            "estimator",
                            "m",
                            "r_dim",
                            "blocks",
                            "sigma_true",
                            "trials",
                            "iter_mean",
                            "fail_rate",
                            "bias1",
                            "bias2",
                            "bias3",
                            "rmse1",
                            "rmse2",
                            "rmse3",
                            "var_ratio1",
                            "var_ratio2",
                            "var_ratio3",
                            "cover1",
                            "cover2",
                            "cover3",
                            "chi2_p",
                        ]
                    )
                    header_written = True
                writer.writerow(row)
    print(f"Benchmark complete. Results saved to {args.outfile}")


if __name__ == "__main__":
    main()
