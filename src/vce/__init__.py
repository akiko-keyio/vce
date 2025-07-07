"""Public API for the VCE package."""

from .core import HelmertVCE, LSVCE, LSVCEPlus
from .simulation import Scenario, monte_carlo, evaluate, plot_cov_ratio

__all__ = [
    "HelmertVCE",
    "LSVCE",
    "LSVCEPlus",
    "Scenario",
    "monte_carlo",
    "evaluate",
    "plot_cov_ratio",
]


def main() -> None:  # pragma: no cover
    print("Hello from vce!")
