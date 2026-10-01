from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


pc_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results/robustness_check"


@dataclass(frozen=True)
class pc_Config:
    horizon_hours: float = 120.0
    dt: float = 0.01
    n_batteries: int = 23
    n_chargers: int = 20
    lost_demand_penalty: float = 3.01
    charger_power_kw: float = 41.0
    high_electricity_price: float = 0.222755
    low_electricity_price: float = 0.067881
    projection_mass_tolerance: float = 1e-14
    projection_probability_cutoff: float = 0.0
    numerical_tolerance: float = 1e-11

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_hours / self.dt))

    @property
    def represented_horizon_hours(self) -> float:
        return self.n_steps * self.dt


@dataclass(frozen=True)
class pc_PhaseTypeModel:
    name: str
    initial_probabilities: np.ndarray
    phase_rates: np.ndarray
    next_phase: Tuple[int, ...]  # -1 denotes replenishment completion

    @property
    def n_phases(self) -> int:
        return len(self.phase_rates)

    @property
    def mean_service_hours(self) -> float:
        if np.count_nonzero(self.initial_probabilities) == 1:
            phase = int(np.argmax(self.initial_probabilities))
            mean = 0.0
            while phase >= 0:
                mean += 1.0 / float(self.phase_rates[phase])
                phase = self.next_phase[phase]
            return mean
        return float(np.sum(self.initial_probabilities / self.phase_rates))


@dataclass(frozen=True)
class pc_ModelData:
    states: np.ndarray
    totals: np.ndarray
    state_index: Dict[Tuple[int, ...], int]
    service_next: np.ndarray
    arrival_next: np.ndarray
    empty_state_id: int
    states_by_total: Tuple[np.ndarray, ...]


def pc_arrival_rate(t: float) -> float:
    return 5.68 if (t % 24.0) < 9.0 else 14.51


def pc_electricity_price(cfg: pc_Config, t: float) -> float:
    hour = t % 24.0
    if 15.0 <= hour < 20.0:
        return cfg.high_electricity_price
    return cfg.low_electricity_price


def pc_phase_type_models() -> Tuple[pc_PhaseTypeModel, ...]:
    erlang_rate = 3.0 / 0.78
    return (
        pc_PhaseTypeModel(
            name="Erlang-3",
            initial_probabilities=np.asarray([1.0, 0.0, 0.0]),
            phase_rates=np.asarray([erlang_rate, erlang_rate, erlang_rate]),
            next_phase=(1, 2, -1),
        ),
        pc_PhaseTypeModel(
            name="Hyperexponential",
            initial_probabilities=np.asarray([0.5, 0.5]),
            phase_rates=np.asarray([1.0 / 0.26, 1.0 / 1.30]),
            next_phase=(-1, -1),
        ),
    )


def pc_enumerate_compositions(
    total_limit: int, dimension: int
) -> List[Tuple[int, ...]]:
    result: List[Tuple[int, ...]] = []

    def append_suffix(
        prefix: Tuple[int, ...], remaining_dim: int, budget: int
    ) -> None:
        if remaining_dim == 1:
            for value in range(budget + 1):
                result.append(prefix + (value,))
            return
        for value in range(budget + 1):
            append_suffix(prefix + (value,), remaining_dim - 1, budget - value)

    append_suffix((), dimension, total_limit)
    return result


def pc_build_model_data(
    cfg: pc_Config, model: pc_PhaseTypeModel
) -> pc_ModelData:
    state_tuples = pc_enumerate_compositions(
        cfg.n_batteries, model.n_phases
    )
    state_index = {state: sid for sid, state in enumerate(state_tuples)}
    states = np.asarray(state_tuples, dtype=np.int16)
    totals = states.sum(axis=1).astype(np.int16)
    n_states = len(state_tuples)
    service_next = np.full(
        (n_states, model.n_phases), -1, dtype=np.int32
    )
    arrival_next = np.full(
        (n_states, model.n_phases), -1, dtype=np.int32
    )

    for sid, state in enumerate(state_tuples):
        for phase in range(model.n_phases):
            if state[phase] > 0:
                target = list(state)
                target[phase] -= 1
                following = model.next_phase[phase]
                if following >= 0:
                    target[following] += 1
                service_next[sid, phase] = state_index[tuple(target)]
            if totals[sid] < cfg.n_batteries:
                target = list(state)
                target[phase] += 1
                arrival_next[sid, phase] = state_index[tuple(target)]

    states_by_total = tuple(
        np.flatnonzero(totals == total)
        for total in range(cfg.n_batteries + 1)
    )
    return pc_ModelData(
        states=states,
        totals=totals,
        state_index=state_index,
        service_next=service_next,
        arrival_next=arrival_next,
        empty_state_id=state_index[(0,) * model.n_phases],
        states_by_total=states_by_total,
    )


def pc_check_time_step(
    cfg: pc_Config, models: Sequence[pc_PhaseTypeModel]
) -> None:
    max_phase_rate = max(
        float(model.phase_rates.max()) for model in models
    )
    max_total_rate = 14.51 + cfg.n_chargers * max_phase_rate
    stability_number = cfg.dt * max_total_rate
    if stability_number > 1.0 + cfg.numerical_tolerance:
        raise ValueError(
            "The time step is too large for the explicit CTMDP recursion: "
            f"dt*(lambda_max+K*rate_max)={stability_number:.6f}>1. "
            "Reduce --dt."
        )


def pc_optimal_phase_allocation(
    cfg: pc_Config,
    model: pc_PhaseTypeModel,
    data: pc_ModelData,
    value_next: np.ndarray,
    energy_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the phase allocation and minimized charging HJB term."""
    n_states = len(data.states)
    coefficients = np.full(
        (n_states, model.n_phases), np.inf, dtype=float
    )
    rows = np.arange(n_states)

    for phase, rate in enumerate(model.phase_rates):
        valid = data.states[:, phase] > 0
        target = data.service_next[valid, phase]
        coefficients[valid, phase] = energy_rate + float(rate) * (
            value_next[target] - value_next[valid]
        )

    order = np.argsort(coefficients, axis=1, kind="stable")
    remaining = np.full(n_states, cfg.n_chargers, dtype=np.int16)
    allocation = np.zeros_like(data.states, dtype=np.int8)

    for rank in range(model.n_phases):
        phase = order[:, rank]
        coefficient = coefficients[rows, phase]
        available = data.states[rows, phase]
        take = np.where(
            coefficient < -cfg.numerical_tolerance,
            np.minimum(available, remaining),
            0,
        ).astype(np.int16)
        allocation[rows, phase] = take.astype(np.int8)
        remaining -= take

    products = np.zeros_like(coefficients)
    np.multiply(
        allocation,
        coefficients,
        out=products,
        where=allocation > 0,
    )
    return allocation, products.sum(axis=1)


def pc_solve_ctmdp(
    cfg: pc_Config,
    model: pc_PhaseTypeModel,
    data: pc_ModelData,
) -> Mapping[str, object]:
    started = time.perf_counter()
    n_states = len(data.states)
    value_next = np.zeros(n_states, dtype=float)
    allocation = np.zeros(
        (cfg.n_steps, n_states, model.n_phases), dtype=np.int8
    )
    full = data.totals == cfg.n_batteries

    for step in range(cfg.n_steps - 1, -1, -1):
        t = step * cfg.dt
        lam = pc_arrival_rate(t)
        energy_rate = cfg.charger_power_kw * pc_electricity_price(cfg, t)

        arrival_term = np.zeros(n_states, dtype=float)
        not_full = ~full
        for phase, alpha in enumerate(model.initial_probabilities):
            if alpha <= 0.0:
                continue
            target = data.arrival_next[not_full, phase]
            arrival_term[not_full] += lam * float(alpha) * (
                value_next[target] - value_next[not_full]
            )
        arrival_term[full] = cfg.lost_demand_penalty * lam

        action, charging_term = pc_optimal_phase_allocation(
            cfg, model, data, value_next, energy_rate
        )
        allocation[step] = action
        value_next = value_next + cfg.dt * (
            arrival_term + charging_term
        )

    probability, observed, active, objective = pc_forward_projection(
        cfg, model, data, allocation
    )
    threshold = pc_threshold_from_activation(cfg, active)
    return {
        "allocation": allocation,
        "total_u": allocation.sum(axis=2).astype(np.int8),
        "initial_value": value_next[data.empty_state_id],
        "forward_expected_cost": objective,
        "projected_positive_charging_probability": probability,
        "projected_observed": observed,
        "projected_active": active,
        "projected_threshold": threshold,
        "elapsed_seconds": time.perf_counter() - started,
    }


def pc_forward_projection(
    cfg: pc_Config,
    model: pc_PhaseTypeModel,
    data: pc_ModelData,
    allocation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Propagate state mass and compute q_n(i) and 1{q_n(i)>cutoff}."""
    n_states = len(data.states)
    mass = np.zeros(n_states, dtype=float)
    mass[data.empty_state_id] = 1.0
    probability = np.full(
        (cfg.n_steps, cfg.n_batteries + 1), np.nan, dtype=float
    )
    observed = np.zeros_like(probability, dtype=bool)
    active = np.zeros_like(probability, dtype=bool)
    objective = 0.0
    not_full = data.totals < cfg.n_batteries

    for step in range(cfg.n_steps):
        t = step * cfg.dt
        lam = pc_arrival_rate(t)
        action = allocation[step].astype(np.int16)
        total_u = action.sum(axis=1)

        for total, ids in enumerate(data.states_by_total):
            denominator = float(mass[ids].sum())
            if denominator <= cfg.projection_mass_tolerance:
                continue
            numerator = float(mass[ids][total_u[ids] > 0].sum())
            q_value = numerator / denominator
            probability[step, total] = q_value
            observed[step, total] = True
            active[step, total] = (
                q_value > cfg.projection_probability_cutoff
            )

        objective += cfg.dt * (
            cfg.charger_power_kw
            * pc_electricity_price(cfg, t)
            * float(mass @ total_u)
            + cfg.lost_demand_penalty
            * lam
            * float(mass[~not_full].sum())
        )

        service_rate = np.sum(
            action * model.phase_rates[None, :], axis=1
        )
        outgoing_rate = service_rate + lam * not_full.astype(float)
        self_probability = 1.0 - cfg.dt * outgoing_rate
        if float(self_probability.min()) < -cfg.numerical_tolerance:
            raise RuntimeError("Negative CTMDP self-transition probability.")
        nxt = mass * np.maximum(self_probability, 0.0)

        for phase, alpha in enumerate(model.initial_probabilities):
            if alpha <= 0.0:
                continue
            target = data.arrival_next[not_full, phase]
            weights = mass[not_full] * cfg.dt * lam * float(alpha)
            nxt += np.bincount(
                target, weights=weights, minlength=n_states
            )

        for phase, rate in enumerate(model.phase_rates):
            valid = action[:, phase] > 0
            if not np.any(valid):
                continue
            target = data.service_next[valid, phase]
            weights = (
                mass[valid]
                * cfg.dt
                * action[valid, phase]
                * float(rate)
            )
            nxt += np.bincount(
                target, weights=weights, minlength=n_states
            )

        total_mass = float(nxt.sum())
        if abs(total_mass - 1.0) > 1e-9:
            raise RuntimeError(
                f"CTMDP probability mass error at step {step}: "
                f"{total_mass:.12f}."
            )
        mass = nxt / total_mass

    return probability, observed, active, objective


def pc_threshold_from_activation(
    cfg: pc_Config, active: np.ndarray
) -> np.ndarray:
    threshold = np.full(
        cfg.n_steps, cfg.n_batteries + 1, dtype=np.int16
    )
    for step, row in enumerate(active):
        ids = np.flatnonzero(row)
        if ids.size:
            threshold[step] = int(ids[0])
    return threshold


def pc_activation_holes(
    active: np.ndarray, observed: np.ndarray
) -> int:
    """Count inactive observed cells above the first active cell."""
    holes = 0
    for active_row, observed_row in zip(active, observed):
        ids = np.flatnonzero(active_row)
        if ids.size:
            start = int(ids[0])
            holes += int(
                np.sum(observed_row[start:] & ~active_row[start:])
            )
    return holes


def pc_configure_plots() -> None:
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = (
        "Times New Roman" if "Times New Roman" in fonts else "Nimbus Roman"
    )
    plt.rcParams.update(
        {
            "font.family": family,
            "font.serif": [family],
            "mathtext.fontset": "stix",
            "font.size": 28,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 22,
            "figure.dpi": 140,
            "savefig.dpi": 420,
        }
    )


def pc_draw_panel(
    ax: plt.Axes,
    cfg: pc_Config,
    result: Mapping[str, object],
    distribution_name: str,
) -> None:
    active = np.asarray(result["projected_active"], dtype=bool)
    threshold = np.asarray(result["projected_threshold"], dtype=float)

    time_edges = (
        np.arange(cfg.n_steps + 1, dtype=float) * cfg.dt / 24.0
    )
    state_edges = np.arange(cfg.n_batteries + 2, dtype=float)

    cells = np.ma.masked_where(~active.T, np.ones_like(active.T, dtype=float))
    cmap = ListedColormap([(1.0, 0.0, 0.0, 0.18)])
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    ax.pcolormesh(
        time_edges,
        state_edges,
        cells,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        edgecolors="none",
        zorder=1,
    )

    threshold_step = np.append(threshold, threshold[-1])
    ax.step(
        time_edges,
        threshold_step,
        where="post",
        color="red",
        linewidth=2.4,
        zorder=3,
    )

    handles = [
        Patch(
            facecolor="red",
            alpha=0.18,
            edgecolor="none",
            label=(
                distribution_name
                + r": $u_n^{\mathrm{agg}}(i)>0$"
            ),
        ),
        Line2D(
            [],
            [],
            color="red",
            linewidth=2.4,
            label=(
                distribution_name
                + r": $i_n^{\mathrm{proj}}$"
            ),
        ),
    ]
    
    ax.legend(handles=handles, loc="upper right", frameon=True)
    ax.set_ylabel(r"State $i$")
    ax.set_ylim(0.0, cfg.n_batteries + 1.25)
    ax.set_yticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlim(0.0, cfg.represented_horizon_hours / 24.0)
    ax.grid(alpha=0.15)


def pc_plot_results(
    cfg: pc_Config,
    results: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> Path:
    pc_configure_plots()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    fig, axes = plt.subplots(
        len(names), 1, figsize=(11.5, 10.5), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        pc_draw_panel(ax, cfg, results[name], name)
    axes[-1].set_xlabel("Time (days)")
    axes[-1].set_xticks(
        np.arange(
            0,
            math.floor(cfg.represented_horizon_hours / 24.0) + 1,
        )
    )
    fig.tight_layout()
    path = output_dir / "preemptive.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def pc_write_summary(
    models: Sequence[pc_PhaseTypeModel],
    data_by_name: Mapping[str, pc_ModelData],
    results: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> Path:
    path = output_dir / "preemptive_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "distribution",
                "mean_service_hours",
                "phases",
                "ctmdp_states",
                "projected_q_positive_holes",
                "initial_value",
                "forward_expected_cost",
                "elapsed_seconds",
            ]
        )
        for model in models:
            result = results[model.name]
            data = data_by_name[model.name]
            holes = pc_activation_holes(
                np.asarray(result["projected_active"]),
                np.asarray(result["projected_observed"]),
            )
            writer.writerow(
                [
                    model.name,
                    f"{model.mean_service_hours:.6f}",
                    model.n_phases,
                    len(data.states),
                    holes,
                    f"{float(result['initial_value']):.10f}",
                    f"{float(result['forward_expected_cost']):.10f}",
                    f"{float(result['elapsed_seconds']):.6f}",
                ]
            )
    return path


def pc_write_npz(
    cfg: pc_Config,
    models: Sequence[pc_PhaseTypeModel],
    data_by_name: Mapping[str, pc_ModelData],
    results: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> Path:
    payload: Dict[str, np.ndarray] = {
        "times_hours": np.arange(cfg.n_steps + 1, dtype=float) * cfg.dt,
        "q_cutoff": np.asarray(cfg.projection_probability_cutoff),
    }
    for model in models:
        key = model.name.lower().replace("-", "_")
        data = data_by_name[model.name]
        result = results[model.name]
        payload[f"{key}_states"] = data.states
        payload[f"{key}_phase_allocation"] = np.asarray(
            result["allocation"]
        )
        payload[f"{key}_total_u"] = np.asarray(result["total_u"])
        payload[f"{key}_projected_positive_charging_probability"] = (
            np.asarray(result["projected_positive_charging_probability"])
        )
        payload[f"{key}_projected_observed"] = np.asarray(
            result["projected_observed"]
        )
        payload[f"{key}_projected_active"] = np.asarray(
            result["projected_active"]
        )
        payload[f"{key}_projected_threshold"] = np.asarray(
            result["projected_threshold"]
        )
    path = output_dir / "preemptive_results.npz"
    np.savez_compressed(path, **payload)
    return path


def pc_run(
    cfg: pc_Config, output_dir: Path
) -> Dict[str, Mapping[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = pc_phase_type_models()
    pc_check_time_step(cfg, models)
    data_by_name: Dict[str, pc_ModelData] = {}
    results: Dict[str, Mapping[str, object]] = {}

    print(
        f"Preemptive phase-type CTMDP: N={cfg.n_batteries}, "
        f"K={cfg.n_chargers}, steps={cfg.n_steps}, "
        f"represented horizon={cfg.represented_horizon_hours:.2f} h, "
        f"active iff q>{cfg.projection_probability_cutoff:g}"
    )

    for model in models:
        data = pc_build_model_data(cfg, model)
        data_by_name[model.name] = data
        result = pc_solve_ctmdp(cfg, model, data)
        results[model.name] = result
        compact = {
            "distribution": model.name,
            "mean_service_hours": model.mean_service_hours,
            "phases": model.n_phases,
            "ctmdp_states": len(data.states),
            "projected_q_positive_holes": pc_activation_holes(
                np.asarray(result["projected_active"]),
                np.asarray(result["projected_observed"]),
            ),
            "initial_value": result["initial_value"],
            "forward_expected_cost": result["forward_expected_cost"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        print(json.dumps(compact, indent=2))

    figure = pc_plot_results(cfg, results, output_dir)
    summary = pc_write_summary(
        models, data_by_name, results, output_dir
    )
    archive = pc_write_npz(
        cfg, models, data_by_name, results, output_dir
    )
    print(figure)
    print(summary)
    print(archive)
    return results


def pc_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=pc_OUTPUT_DIR)
    parser.add_argument("--horizon", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--q-cutoff",
        type=float,
        default=0.0,
        help=(
            "Shade a cell iff q_n(i) is strictly greater than this value "
            "(default: 0)."
        ),
    )
    args, _unknown = parser.parse_known_args()
    if not 0.0 <= args.q_cutoff < 1.0:
        parser.error("--q-cutoff must satisfy 0 <= cutoff < 1.")
    return args


def main() -> None:
    args = pc_parse_args()
    cfg = pc_Config(
        horizon_hours=args.horizon,
        dt=args.dt,
        projection_probability_cutoff=args.q_cutoff,
    )
    pc_run(cfg, args.output_dir)


if __name__ == "__main__":
    main()
