#!/usr/bin/env python3
"""
benchmark.py — minimal yet correct Monte-Carlo benchmark for VCE estimators

* 四个场景 m ∈ {12, 30, 60, 120}
* 试次数 trial_base × (120 // m) —— 总样本量大致均衡
* 扩展 Metrics: bias / sd / variance / rmse / mse / var_ratio /
               cover95 / χ²-p / mean-median-max iter / fail_rate
* 利用 pandas.DataFrame 直接写 CSV ＆打印 Markdown
"""
from __future__ import annotations

import argparse, multiprocessing as mp, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np, pandas as pd
from tqdm import tqdm

from vce.simulation import Scenario, monte_carlo   # ← 复用原工具

# --------------------------------------------------------------------------- #
# 1. 生成场景                                                                 #
# --------------------------------------------------------------------------- #
def make_scenarios(trial_base: int) -> List[Scenario]:
    m_list = [12, 30, 60]
    scns: List[Scenario] = []
    for m in m_list:
        r_dim      = max(2, m // 30)           # 随 m 线性增长
        trials     = trial_base * (120 // m)   # 保证总样本均衡
        block      = (m // 3, m // 3, m - 2*(m // 3))
        scns.append(
            Scenario(
                m=m, r_dim=r_dim, block_sizes=block,
                sigma_true=(5., 2., 1.),
                n_trials=trials, seed=m,
            )
        )
    return scns

# --------------------------------------------------------------------------- #
# 2. trial-→metric 计算（修正 cover95 计算）                                  #
# --------------------------------------------------------------------------- #
def calc_metrics(res: Dict[str, Dict[str, np.ndarray]],
                 true_sigma: List[float], m: int, r_dim: int
) -> Dict[str, Dict[str, Any]]:
    import scipy.stats

    true = np.asarray(true_sigma, float)
    dof  = m - r_dim
    out: Dict[str, Dict[str, Any]] = {}

    for est, dat in res.items():
        sig       = dat["sigma"]
        conv      = dat["converged"]
        ok_mask   = conv & np.isfinite(sig).all(axis=1)
        if not ok_mask.any():                             # 全部失败
            out[est] = {"fail_rate": 1.0}
            continue

        sig_ok       = sig[ok_mask]
        cov_theo_ok  = dat["cov_theo"][ok_mask]
        chi2_vals    = dat["chi2"][ok_mask]
        n_iter       = dat["n_iter"][ok_mask]

        mean_est = sig_ok.mean(axis=0)
        bias     = mean_est - true
        emp_cov  = np.cov(sig_ok.T, ddof=1)
        var      = np.diag(emp_cov)
        sd       = np.sqrt(var)
        rmse     = np.sqrt(bias**2 + var)
        mse      = bias**2 + var

        cov_mean  = cov_theo_ok.mean(axis=0)
        var_ratio = var / np.diag(cov_mean)
        ci_half   = 1.96 * np.sqrt(np.diag(cov_mean))
        cover95   = ((sig_ok >= (true - ci_half)) &
                     (sig_ok <= (true + ci_half))).mean(axis=0)

        chi2_p = 1.0 - scipy.stats.chi2.cdf(chi2_vals.mean(), dof)

        out[est] = {
            "bias":       bias,
            "sd":         sd,
            "variance":   var,
            "rmse":       rmse,
            "mse":        mse,
            "var_ratio":  var_ratio,
            "cover95":    cover95,
            "chi2_p":     chi2_p,
            "iter_mean":    n_iter.mean(),
            "iter_median":  np.median(n_iter),
            "iter_max":     n_iter.max(),
            "fail_rate":  1.0 - ok_mask.mean(),
        }
    return out

# --------------------------------------------------------------------------- #
# 3. worker —— 每个 Scenario → DataFrame 行                                   #
# --------------------------------------------------------------------------- #
def scenario_worker(scn: Scenario) -> pd.DataFrame:
    t0   = time.time()
    res  = monte_carlo(scn)
    mets = calc_metrics(res, list(scn.sigma_true), scn.m, scn.r_dim)
    rows = []
    for est, m in mets.items():
        base = dict(
            estimator = est,
            m = scn.m,
            r_dim = scn.r_dim,
            block1 = scn.block_sizes[0],
            block2 = scn.block_sizes[1],
            block3 = scn.block_sizes[2],
            scenario_runtime = time.time() - t0,
        )
        # 把 numpy 数组/标量统一展平成一行
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                for i, x in enumerate(v, 1):
                    base[f"{k}{i}"] = float(x)
            else:
                base[k] = float(v)
        rows.append(base)
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# 4. main                                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-base", type=int, default=75)
    ap.add_argument("-p", "--processes", type=int,
                    default=min(4, mp.cpu_count()))
    ap.add_argument("--outfile", type=Path, default="vce_summary.csv")
    args = ap.parse_args()

    scns = make_scenarios(args.trial_base)
    with mp.Pool(args.processes) as pool:
        df_list = list(tqdm(pool.imap_unordered(scenario_worker, scns),
                            total=len(scns)))
    summary = pd.concat(df_list, ignore_index=True)
    summary.to_csv(args.outfile, index=False)
    print("✅  saved", args.outfile)

    # 终端快速浏览（bias±sd + χ²p + fail% + iter¯）
    view = summary[[
        "estimator","m",
        "bias1","sd1","bias2","sd2","bias3","sd3",
        "chi2_p","fail_rate","iter_mean"
    ]].copy()
    view["σ1"] = view.apply(lambda r: f"{r.bias1:+.2f}±{r.sd1:.2f}", axis=1)
    view["σ2"] = view.apply(lambda r: f"{r.bias2:+.2f}±{r.sd2:.2f}", axis=1)
    view["σ3"] = view.apply(lambda r: f"{r.bias3:+.2f}±{r.sd3:.2f}", axis=1)
    view["fail%"] = (view.fail_rate*100).round(1).astype(str)+"%"
    view["iter¯"] = view.iter_mean.round(1)
    print(view[["estimator","m","σ1","σ2","σ3","chi2_p","fail%","iter¯"]])
