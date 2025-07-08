import numpy as np
from vce.simulation import Scenario, monte_carlo, evaluate


def test_monte_carlo_shapes() -> None:
    scn = Scenario(
        m=60, r_dim=3, block_sizes=[20, 20, 20], sigma_true=[5.0, 2.0, 1.0], n_trials=5
    )
    results = monte_carlo(scn)
    for data in results.values():
        assert data["sigma"].shape == (scn.n_trials, len(scn.block_sizes))
        assert data["cov_theo"].shape == (
            scn.n_trials,
            len(scn.block_sizes),
            len(scn.block_sizes),
        )
        assert data["chi2"].shape == (scn.n_trials,)


def test_monte_carlo_bias_small() -> None:
    scn = Scenario(
        m=60,
        r_dim=3,
        block_sizes=[20, 20, 20],
        sigma_true=[4.0, 1.5, 0.5],
        n_trials=20,
        seed=123,
    )
    results = monte_carlo(scn)
    for method, data in results.items():
        if method == "mixedlm":
            continue
        mean_est = data["sigma"].mean(axis=0)
        assert np.allclose(mean_est, scn.sigma_true, rtol=0.2)


def test_evaluate_and_seed() -> None:
    scn = Scenario(
        m=30,
        r_dim=2,
        block_sizes=[15, 15],
        sigma_true=[1.0, 2.0],
        n_trials=3,
        seed=42,
    )
    results_a = monte_carlo(scn)
    metrics = evaluate(results_a, scn.sigma_true, scn.m, scn.r_dim)
    results_b = monte_carlo(scn)
    for key in results_a:
        for metric in results_a[key]:
            arr_a = results_a[key][metric]
            arr_b = results_b[key][metric]
            if np.isnan(arr_a).all() and np.isnan(arr_b).all():
                continue
            assert np.array_equal(arr_a, arr_b)
        assert metrics[key].bias.shape == (len(scn.block_sizes),)
