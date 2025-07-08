from vce.benchmark import (
    make_scenarios,
    metrics_to_row,
    scenario_worker,
    scenarios_to_df,
)
from vce.simulation import Scenario


def test_make_scenarios_balanced() -> None:
    scens = make_scenarios(3)
    assert len(scens) == 4
    for scn in scens:
        assert scn.n_trials == 3


def test_metrics_to_row_len() -> None:
    scn = Scenario(
        m=12, r_dim=3, block_sizes=[4, 4, 4], sigma_true=[1.0, 1.0, 1.0], n_trials=1
    )
    data = scenario_worker(scn)
    row = metrics_to_row("helmert", data)
    header_len = len(scenarios_to_df([data]).columns)
    assert len(row) == header_len


def test_scenarios_to_df() -> None:
    scn = Scenario(
        m=12, r_dim=3, block_sizes=[4, 4, 4], sigma_true=[1.0, 1.0, 1.0], n_trials=1
    )
    data = [scenario_worker(scn)]
    df = scenarios_to_df(data)
    assert len(df.columns) == len(metrics_to_row("helmert", data[0]))
    assert len(df) == 3


def test_scenario_worker_complex() -> None:
    scn = Scenario(
        m=30,
        r_dim=5,
        block_sizes=[10, 10, 10],
        sigma_true=[2.0, 1.0, 0.5],
        n_trials=2,
        seed=123,
    )
    out = scenario_worker(scn)
    assert set(out).issuperset({"helmert", "lsvce", "lsvce_plus"})
