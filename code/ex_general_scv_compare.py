from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parent.parent

EXPONENTIAL_CSV = (
    ROOT
    / "results"
    / "bimodal_exponential_common_grid"
    / "bimodal_exponential_common_comparison.csv"
)

COMPONENT_H2_CSV = Path(
    ROOT
    / "results"
    / "bimodal_h2_erlang_common_grid"
    / "bimodal_h2_erlang_common_comparison.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"

POLICY_COST_PATH = OUTPUT_DIR / "bimodal_approx_policy_cost.png"
OPTIMAL_COST_PATH = OUTPUT_DIR / "bimodal_approx_optimal_cost.png"
COMPUTATIONAL_TIME_PATH = OUTPUT_DIR / "bimodal_approx_computational_time.png"

def read_csv_as_dicts(path: Path) -> Dict[float, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find:\n{path}")

    data: Dict[float, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"No header found in {path}")
        if "scv" not in reader.fieldnames:
            raise RuntimeError(f"'scv' column not found in {path}")
        for row in reader:
            data[round(float(row["scv"]), 10)] = row

    if not data:
        raise RuntimeError(f"No data found in {path}")
    return data


def value_from_columns(row: dict, columns: Iterable[str], source: Path) -> float:
    """Read the first available column, with a useful schema error."""
    candidates = tuple(columns)
    for column in candidates:
        if column in row and row[column] not in (None, ""):
            return float(row[column])
    raise RuntimeError(
        f"None of the required columns {candidates} was found in {source}.\n"
        f"Available columns: {tuple(row.keys())}"
    )


def configure_plots() -> None:
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in fonts else "Nimbus Roman"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.serif": [family],
            "mathtext.fontset": "stix",
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "figure.dpi": 140,
            "savefig.dpi": 420,
        }
    )


def format_scv_axis(ax: plt.Axes, x: np.ndarray) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("SCV of bimodal distribution")
    ax.margins(y=0.08)
    ax.grid(alpha=0.2)

def main() -> None:
    exp_rows = read_csv_as_dicts(EXPONENTIAL_CSV)
    component_rows = read_csv_as_dicts(COMPONENT_H2_CSV)

    common_scvs = sorted(set(exp_rows) & set(component_rows))
    if not common_scvs:
        raise RuntimeError("The two CSV files have no common SCV values.")
    x = np.asarray(common_scvs, dtype=float)

    bimodal_cost_exp = np.asarray(
        [float(exp_rows[scv]["bimodal_optimal_expected_cost"]) for scv in common_scvs],
        dtype=float,
    )
    bimodal_cost_component = np.asarray(
        [float(component_rows[scv]["bimodal_optimal_expected_cost"]) for scv in common_scvs],
        dtype=float,
    )
    if not np.allclose(
        bimodal_cost_exp,
        bimodal_cost_component,
        atol=1e-8,
        rtol=1e-10,
    ):
        raise RuntimeError(
            "The bimodal optimal costs in the exponential and "
            "component-matched H2 CSV files do not match."
        )
    bimodal_cost = bimodal_cost_component

    exponential_policy_cost = np.asarray(
        [
            value_from_columns(
                exp_rows[scv],
                ("exponential_policy_bimodal_expected_cost",),
                EXPONENTIAL_CSV,
            )
            for scv in common_scvs
        ],
        dtype=float,
    )
    component_policy_cost = np.asarray(
        [
            value_from_columns(
                component_rows[scv],
                (
                    "surrogate_policy_bimodal_expected_cost",
                    "h2_policy_bimodal_expected_cost",
                ),
                COMPONENT_H2_CSV,
            )
            for scv in common_scvs
        ],
        dtype=float,
    )

    exponential_loss = 100.0 * (exponential_policy_cost - bimodal_cost) / bimodal_cost
    component_loss = 100.0 * (component_policy_cost - bimodal_cost) / bimodal_cost

    exponential_optimal_cost = np.asarray(
        [float(exp_rows[scv]["exponential_optimal_expected_cost"]) for scv in common_scvs],
        dtype=float,
    )
    component_optimal_cost = np.asarray(
        [
            value_from_columns(
                component_rows[scv],
                ("surrogate_optimal_expected_cost", "h2_optimal_expected_cost"),
                COMPONENT_H2_CSV,
            )
            for scv in common_scvs
        ],
        dtype=float,
    )

    bimodal_time = np.asarray(
        [float(component_rows[scv]["exact_elapsed_seconds"]) for scv in common_scvs],
        dtype=float,
    )
    exponential_time = np.asarray(
        [
            value_from_columns(
                exp_rows[scv],
                ("exponential_elapsed_seconds", "exp_elapsed_seconds"),
                EXPONENTIAL_CSV,
            )
            for scv in common_scvs
        ],
        dtype=float,
    )
    component_time = np.asarray(
        [
            value_from_columns(
                component_rows[scv],
                ("surrogate_elapsed_seconds", "h2_elapsed_seconds"),
                COMPONENT_H2_CSV,
            )
            for scv in common_scvs
        ],
        dtype=float,
    )

    print(
        "\nSCV | Bimodal policy cost | Exponential policy cost | "
        "Component H2 policy cost | Exp loss (%) | Component H2 loss (%)"
    )
    print("-" * 116)
    for values in zip(
        x,
        bimodal_cost,
        exponential_policy_cost,
        component_policy_cost,
        exponential_loss,
        component_loss,
    ):
        scv, bimodal_c, exp_c, component_c, exp_loss, component_l = values
        print(
            f"{scv:>3g} | {bimodal_c:>19.4f} | {exp_c:>23.4f} | "
            f"{component_c:>24.4f} | {exp_loss:>12.4f} | {component_l:>21.4f}"
        )

    print(
        "\nSCV | Bimodal optimal cost | Exponential optimal cost | "
        "Component H2 optimal cost"
    )
    print("-" * 91)
    for scv, bimodal_c, exp_c, component_c in zip(
        x,
        bimodal_cost,
        exponential_optimal_cost,
        component_optimal_cost,
    ):
        print(
            f"{scv:>3g} | {bimodal_c:>20.4f} | {exp_c:>24.4f} | "
            f"{component_c:>25.4f}"
        )

    print("\nSCV | Bimodal time | Exponential time | Component H2 time")
    print("-" * 65)
    for scv, exact_t, exp_t, component_t in zip(
        x,
        bimodal_time,
        exponential_time,
        component_time,
    ):
        print(
            f"{scv:>3g} | {exact_t:>12.4f} | {exp_t:>16.4f} | "
            f"{component_t:>17.4f}"
        )

    configure_plots()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(10, 6.1))
    ax1.plot(
        x,
        exponential_policy_cost,
        color="#2474b5",
        marker="s",
        markersize=8,
        linewidth=2.1,
        label="Exponential",
    )
    ax1.plot(
        x,
        component_policy_cost,
        color="#d95f02",
        marker="D",
        markersize=7.5,
        linewidth=2.1,
        label=r"$H_2$ (component-matched)",
    )
    ax1.plot(
        x,
        bimodal_cost,
        color="#222222",
        marker="o",
        markersize=8,
        linewidth=2.1,
        label="Bimodal",
    )
    format_scv_axis(ax1, x)
    ax1.set_ylabel("Expected cost under bimodal (USD)")
    ax1.legend(frameon=True)
    fig1.tight_layout()
    fig1.savefig(POLICY_COST_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6.1))
    ax2.plot(
        x,
        exponential_optimal_cost,
        color="#2474b5",
        marker="s",
        markersize=8,
        linewidth=2.1,
        label="Exponential",
    )
    ax2.plot(
        x,
        component_optimal_cost,
        color="#d95f02",
        marker="D",
        markersize=7.5,
        linewidth=2.1,
        label=r"$H_2$ (component-matched)",
    )
    ax2.plot(
        x,
        bimodal_cost,
        color="#222222",
        marker="o",
        markersize=8,
        linewidth=2.1,
        label="Bimodal",
    )
    format_scv_axis(ax2, x)
    ax2.set_ylabel("Optimal expected cost (USD)")
    ax2.legend(frameon=True)
    fig2.tight_layout()
    fig2.savefig(OPTIMAL_COST_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6.1))
    ax3.plot(
        x,
        exponential_time,
        color="#2474b5",
        marker="s",
        markersize=8,
        linewidth=2.1,
        label="Exponential",
    )
    ax3.plot(
        x,
        component_time,
        color="#d95f02",
        marker="D",
        markersize=7.5,
        linewidth=2.1,
        label=r"$H_2$ (component-matched)",
    )
    ax3.plot(
        x,
        bimodal_time,
        color="#222222",
        marker="o",
        markersize=8,
        linewidth=2.1,
        label="Bimodal",
    )
    format_scv_axis(ax3, x)
    ax3.set_yscale("log")
    ax3.set_ylabel("Computational time (seconds)")
    ax3.legend(frameon=True)
    fig3.tight_layout()
    fig3.savefig(COMPUTATIONAL_TIME_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig3)

    print("\nFigures saved to:")
    print(POLICY_COST_PATH)
    print(OPTIMAL_COST_PATH)
    print(COMPUTATIONAL_TIME_PATH)


if __name__ == "__main__":
    main()
