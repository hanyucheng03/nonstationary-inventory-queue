from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
RESULTS_DIR = ROOT / "results"

PRIMITIVE_SCRIPTS = (
    "utah_summer_primitive_experiment.py",
    "utah_summer_primitive_experiment_lower_power.py",
    "utah_winter_primitive_experiment.py",
    "utah_winter_primitive_experiment_lower_power.py",
)

# Files used directly as manuscript figure panels / figures.
ARTICLE_FIGURES = (
    # Figures 1--4.
    RESULTS_DIR / "numerical_utah_summer_figures" / "experiment1_marginal_costs_5days_summer.png",
    RESULTS_DIR / "numerical_utah_summer_figures" / "optimal_vs_primitive_certificates_summer.png",
    RESULTS_DIR / "numerical_utah_summer_figures" / "experiment1_marginal_costs_5days_summer_lower_power.png",
    RESULTS_DIR / "numerical_utah_summer_figures" / "optimal_vs_primitive_certificates_summer_lower_power.png",
    RESULTS_DIR / "numerical_utah_winter_figures" / "experiment1_marginal_costs_5days_winter.png",
    RESULTS_DIR / "numerical_utah_winter_figures" / "optimal_vs_primitive_certificates_winter.png",
    RESULTS_DIR / "numerical_utah_winter_figures" / "experiment1_marginal_costs_5days_winter_lower_power.png",
    RESULTS_DIR / "numerical_utah_winter_figures" / "optimal_vs_primitive_certificates_winter_lower_power.png",
    # Figure 5.
    RESULTS_DIR / "robustness_check" / "preemptive.png",
    # Figure 6(a)--(b).
    RESULTS_DIR / "scv_results" / "ph_scv_optimal_cost.png",
    RESULTS_DIR / "scv_results" / "ph_lost_demand_dominates.png",
    # Figure 7(a)--(b) and Figure 8.
    RESULTS_DIR / "bimodal_approx_optimal_cost.png",
    RESULTS_DIR / "bimodal_approx_policy_cost.png",
    RESULTS_DIR / "bimodal_approx_computational_time.png",
)


def execute(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    """Run one experiment as a checked subprocess."""
    print("\nRunning:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def run_primitive(environment: dict[str, str]) -> None:
    """Reproduce the four peak/off-peak experiments (Figures 1--4)."""
    for script in PRIMITIVE_SCRIPTS:
        execute(
            [sys.executable, str(CODE_DIR / script)],
            cwd=RESULTS_DIR,
            environment=environment,
        )


def run_preemptive(environment: dict[str, str]) -> None:
    """Reproduce the phase-type threshold experiment (Figure 5)."""
    output_dir = RESULTS_DIR / "robustness_check"
    execute(
        [
            sys.executable,
            str(CODE_DIR / "robust_preemptive.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        environment=environment,
    )


def run_ph_scv(environment: dict[str, str]) -> None:
    """Reproduce the two PH-SCV cost experiments (Figure 6)."""
    output_dir = RESULTS_DIR / "scv_results"

    execute(
        [
            sys.executable,
            str(CODE_DIR / "ph_scv_objective.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        environment=environment,
    )

    execute(
        [sys.executable, str(CODE_DIR / "ph_scv_objective_counter.py")],
        cwd=ROOT,
        environment=environment,
    )


def run_ex_general_plotter() -> None:
    script = CODE_DIR / "ex_general_scv_compare.py"
    spec = importlib.util.spec_from_file_location("ex_general_scv_compare_runner", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.EXPONENTIAL_CSV = (
        RESULTS_DIR
        / "bimodal_exponential_common_grid"
        / "bimodal_exponential_common_comparison.csv"
    )
    module.COMPONENT_H2_CSV = (
        RESULTS_DIR
        / "bimodal_h2_erlang_common_grid"
        / "bimodal_h2_erlang_common_comparison.csv"
    )
    module.OUTPUT_DIR = RESULTS_DIR
    module.POLICY_COST_PATH = RESULTS_DIR / "bimodal_approx_policy_cost.png"
    module.OPTIMAL_COST_PATH = RESULTS_DIR / "bimodal_approx_optimal_cost.png"
    module.COMPUTATIONAL_TIME_PATH = RESULTS_DIR / "bimodal_approx_computational_time.png"

    print("\nRunning: code/ex_general_scv_compare.py (portable paths)", flush=True)
    module.main()


def run_bimodal(environment: dict[str, str]) -> None:
    """Reproduce the bimodal/PH/exponential comparison (Figures 7--8)."""
    component_dir = RESULTS_DIR / "bimodal_h2_erlang_common_grid"

    execute(
        [
            sys.executable,
            str(CODE_DIR / "ph_general_mean_compare.py"),
            "--output-dir",
            str(component_dir),
        ],
        cwd=ROOT,
        environment=environment,
    )

    run_ex_general_plotter()


def verify_article_figures() -> None:
    """Fail loudly if any manuscript figure file is missing."""
    missing = [path for path in ARTICLE_FIGURES if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        raise RuntimeError(
            "The run finished, but the following manuscript figure files are missing:\n"
            + formatted
        )

    print("\nAll manuscript figure files were generated:")
    for path in ARTICLE_FIGURES:
        print(f"  {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=("primitive", "preemptive", "ph-scv", "bimodal", "figures", "all"),
        default="all",
        help=(
            "Run only one experiment group. 'figures' regenerates the final "
            "Figure 7/8 plots from existing CSV files; 'all' reproduces every "
            "experiment used by the current manuscript (default: all)."
        ),
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["MPLBACKEND"] = "Agg"
    environment = os.environ.copy()

    if args.subset in {"primitive", "all"}:
        run_primitive(environment)

    if args.subset in {"preemptive", "all"}:
        run_preemptive(environment)

    if args.subset in {"ph-scv", "all"}:
        run_ph_scv(environment)

    if args.subset in {"bimodal", "all"}:
        run_bimodal(environment)

    if args.subset == "figures":
        run_ex_general_plotter()

    if args.subset == "all":
        verify_article_figures()


if __name__ == "__main__":
    main()
