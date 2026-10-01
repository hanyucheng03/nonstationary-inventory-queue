from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import NullFormatter

try:
    import numba
    from numba import prange

    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False
    prange = range


DISTRIBUTIONS = [
    "Erlang-5",
    "Erlang-3",
    "Erlang-2",
    "Exponential",
    "Hyper-2",
    "Hyper-3",
    "Hyper-5",
    "Hyper-10",
]

MAX_STATES = 100_000_000

REQUESTED_DT = 0.01
CTMDP_STABILITY_MARGIN = 0.95

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results/scv_results"

NUMBA_THREADS = None

LOW_ARRIVAL_RATE = 5.68
HIGH_ARRIVAL_RATE = 14.51


@dataclass(frozen=True)
class Config:
    horizon_hours: float = 120.0
    dt: float = REQUESTED_DT
    n_batteries: int = 23
    n_chargers: int = 20
    mean_replenishment_hours: float = 0.78
    lost_demand_penalty: float = 3.01
    charger_power_kw: float = 41.0
    high_electricity_price: float = 0.222755
    low_electricity_price: float = 0.067881
    numerical_tolerance: float = 1e-11

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_hours / self.dt))

    @property
    def represented_horizon_hours(self) -> float:
        return self.n_steps * self.dt


@dataclass(frozen=True)
class PhaseTypeModel:
    name: str
    family: str
    initial_probabilities: np.ndarray
    phase_rates: np.ndarray
    next_phase: Tuple[int, ...]  # -1 denotes replenishment completion
    target_scv: float
    specification: str

    @property
    def n_phases(self) -> int:
        return len(self.phase_rates)

    @property
    def mean_hours(self) -> float:
        if self.family in {"Erlang", "Exponential"}:
            return self.n_phases / float(self.phase_rates[0])
        return float(
            np.sum(self.initial_probabilities / self.phase_rates)
        )

    @property
    def variance_hours2(self) -> float:
        if self.family in {"Erlang", "Exponential"}:
            return self.n_phases / float(self.phase_rates[0]) ** 2
        second_moment = float(
            np.sum(
                self.initial_probabilities
                * 2.0
                / self.phase_rates**2
            )
        )
        return second_moment - self.mean_hours**2

    @property
    def actual_scv(self) -> float:
        return self.variance_hours2 / self.mean_hours**2


@dataclass(frozen=True)
class ModelData:
    states: np.ndarray
    totals: np.ndarray
    arrival_next: np.ndarray
    service_next: np.ndarray
    empty_state_id: int


@dataclass(frozen=True)
class Result:
    distribution: str
    family: str
    specification: str
    target_scv: float
    actual_scv: float
    mean_hours: float
    variance_hours2: float
    phases: int
    states: int
    time_step: float
    time_steps: int
    optimal_charging_cost: float
    optimal_lost_demand_cost: float
    optimal_expected_cost: float
    relative_to_exponential_percent: float
    charging_relative_to_exponential_percent: float
    lost_relative_to_exponential_percent: float
    exponential_policy_expected_cost: float
    misspecification_loss_absolute: float
    misspecification_loss_percent: float
    elapsed_seconds: float


def arrival_rate(t: float) -> float:
    return LOW_ARRIVAL_RATE if (t % 24.0) < 9.0 else HIGH_ARRIVAL_RATE


def electricity_price(cfg: Config, t: float) -> float:
    hour = t % 24.0
    if 15.0 <= hour < 20.0:
        return cfg.high_electricity_price
    return cfg.low_electricity_price


def build_time_profiles(
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.arange(cfg.n_steps, dtype=np.float64) * cfg.dt
    hours = np.mod(times, 24.0)

    arrival_rates = np.where(
        hours < 9.0,
        LOW_ARRIVAL_RATE,
        HIGH_ARRIVAL_RATE,
    ).astype(np.float64)

    electricity_prices = np.where(
        (hours >= 15.0) & (hours < 20.0),
        cfg.high_electricity_price,
        cfg.low_electricity_price,
    ).astype(np.float64)

    operating_rates = (
        cfg.charger_power_kw * electricity_prices
    ).astype(np.float64)

    return arrival_rates, operating_rates


def make_erlang(mean: float, order: int) -> PhaseTypeModel:
    if order < 1:
        raise ValueError("An Erlang order must be positive.")
    rate = order / mean
    alpha = np.zeros(order)
    alpha[0] = 1.0
    if order == 1:
        return PhaseTypeModel(
            name="Exponential",
            family="Exponential",
            initial_probabilities=alpha,
            phase_rates=np.asarray([rate]),
            next_phase=(-1,),
            target_scv=1.0,
            specification=f"rate={rate:.10g}",
        )
    return PhaseTypeModel(
        name=f"Erlang-{order}",
        family="Erlang",
        initial_probabilities=alpha,
        phase_rates=np.full(order, rate),
        next_phase=tuple(range(1, order)) + (-1,),
        target_scv=1.0 / order,
        specification=f"{order} sequential phases; rate={rate:.10g}",
    )


def make_hyperexponential(
    mean: float, target_scv: float
) -> PhaseTypeModel:
    if target_scv <= 1.0:
        raise ValueError("Hyper-c requires c>1.")
    root = math.sqrt((target_scv - 1.0) / (target_scv + 1.0))
    p = 0.5 * (1.0 + root)
    probabilities = np.asarray([p, 1.0 - p])
    rates = np.asarray([2.0 * p / mean, 2.0 * (1.0 - p) / mean])
    return PhaseTypeModel(
        name=f"Hyper-{target_scv:g}",
        family="Hyperexponential",
        initial_probabilities=probabilities,
        phase_rates=rates,
        next_phase=(-1, -1),
        target_scv=target_scv,
        specification=(
            f"p={p:.10g}; 1-p={1.0-p:.10g}; "
            f"rates=({rates[0]:.10g},{rates[1]:.10g})"
        ),
    )


def parse_distribution(name: str, mean: float) -> PhaseTypeModel:
    cleaned = name.strip()
    lower = cleaned.lower()
    if lower == "exponential":
        return make_erlang(mean, 1)
    if lower.startswith("erlang-"):
        try:
            order = int(cleaned.split("-", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid distribution name: {name}") from exc
        return make_erlang(mean, order)
    if lower.startswith("hyper-"):
        try:
            target_scv = float(cleaned.split("-", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid distribution name: {name}") from exc
        return make_hyperexponential(mean, target_scv)
    raise ValueError(
        f"Unknown distribution '{name}'. Use Erlang-k, Exponential, or Hyper-c."
    )


def build_models(
    cfg: Config,
    names: Sequence[str],
) -> Tuple[PhaseTypeModel, ...]:
    if not names:
        raise ValueError("The distribution list cannot be empty.")
    models = tuple(
        parse_distribution(name, cfg.mean_replenishment_hours)
        for name in names
    )
    normalized_names = [model.name.lower() for model in models]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("The distribution list contains duplicates.")
    for model in models:
        if not np.isclose(
            model.mean_hours,
            cfg.mean_replenishment_hours,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Mean check failed for {model.name}.")
        if not np.isclose(
            model.actual_scv,
            model.target_scv,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"SCV check failed for {model.name}.")
    return models


def choose_common_time_step(
    cfg: Config,
    models: Sequence[PhaseTypeModel],
) -> Config:
    max_exit_rate = max(
        float(model.phase_rates.max())
        for model in models
    )

    max_total_rate = (
        max(LOW_ARRIVAL_RATE, HIGH_ARRIVAL_RATE)
        + cfg.n_chargers * max_exit_rate
    )

    subdivisions = max(
        1,
        math.ceil(
            cfg.dt
            * max_total_rate
            / CTMDP_STABILITY_MARGIN
        ),
    )
    effective_dt = cfg.dt / subdivisions

    if subdivisions > 1:
        print(
            f"Refining the common time step from {cfg.dt:g} to "
            f"{effective_dt:g} ({subdivisions} substeps) to satisfy the "
            "explicit CTMDP stability condition."
        )

    return replace(cfg, dt=effective_dt)


def number_of_states(
    n_batteries: int,
    n_phases: int,
) -> int:
    return math.comb(
        n_batteries + n_phases,
        n_phases,
    )


def enumerate_compositions(
    total_limit: int,
    dimension: int,
) -> List[Tuple[int, ...]]:
    result: List[Tuple[int, ...]] = []

    def append_suffix(
        prefix: Tuple[int, ...],
        remaining_dim: int,
        budget: int,
    ) -> None:
        if remaining_dim == 1:
            for value in range(budget + 1):
                result.append(prefix + (value,))
            return
        for value in range(budget + 1):
            append_suffix(
                prefix + (value,),
                remaining_dim - 1,
                budget - value,
            )

    append_suffix((), dimension, total_limit)
    return result


def build_model_data(
    cfg: Config,
    model: PhaseTypeModel,
) -> ModelData:
    predicted_states = number_of_states(
        cfg.n_batteries,
        model.n_phases,
    )
    if predicted_states > MAX_STATES:
        raise RuntimeError(
            f"{model.name} requires {predicted_states:,} exact "
            f"phase-composition states for N={cfg.n_batteries}, exceeding "
            f"MAX_STATES={MAX_STATES:,}. Remove/comment out this entry, "
            "reduce N, or deliberately raise MAX_STATES."
        )

    state_tuples = enumerate_compositions(
        cfg.n_batteries,
        model.n_phases,
    )
    state_index: Dict[Tuple[int, ...], int] = {
        state: sid
        for sid, state in enumerate(state_tuples)
    }

    states = np.asarray(
        state_tuples,
        dtype=np.int16,
    )
    totals = states.sum(
        axis=1,
    ).astype(np.int16)

    n_states = len(states)

    arrival_next = np.full(
        (n_states, model.n_phases),
        -1,
        dtype=np.int32,
    )
    service_next = np.full(
        (n_states, model.n_phases),
        -1,
        dtype=np.int32,
    )

    for sid, state_tuple in enumerate(state_tuples):
        state = list(state_tuple)

        for phase in range(model.n_phases):
            if totals[sid] < cfg.n_batteries:
                target = state.copy()
                target[phase] += 1
                arrival_next[sid, phase] = state_index[
                    tuple(target)
                ]

            if state[phase] > 0:
                target = state.copy()
                target[phase] -= 1

                successor = model.next_phase[phase]
                if successor >= 0:
                    target[successor] += 1

                service_next[sid, phase] = state_index[
                    tuple(target)
                ]

    empty_state_id = state_index[
        (0,) * model.n_phases
    ]

    return ModelData(
        states=states,
        totals=totals,
        arrival_next=arrival_next,
        service_next=service_next,
        empty_state_id=empty_state_id,
    )


def minimize_charging_term(
    cfg: Config,
    model: PhaseTypeModel,
    data: ModelData,
    value_next: np.ndarray,
    operating_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_states = len(data.states)

    coefficients = np.full(
        (n_states, model.n_phases),
        np.inf,
        dtype=float,
    )

    for phase, rate in enumerate(model.phase_rates):
        valid = data.states[:, phase] > 0
        target = data.service_next[
            valid,
            phase,
        ]
        coefficients[valid, phase] = (
            operating_rate
            + float(rate)
            * (
                value_next[target]
                - value_next[valid]
            )
        )

    order = np.argsort(
        coefficients,
        axis=1,
        kind="stable",
    )

    remaining = np.full(
        n_states,
        cfg.n_chargers,
        dtype=np.int16,
    )
    allocation = np.zeros_like(
        data.states,
        dtype=np.int16,
    )
    minimized_term = np.zeros(
        n_states,
        dtype=float,
    )

    rows = np.arange(n_states)

    for rank in range(model.n_phases):
        phase = order[:, rank]
        coefficient = coefficients[
            rows,
            phase,
        ]
        available = data.states[
            rows,
            phase,
        ]

        take = np.where(
            coefficient < -cfg.numerical_tolerance,
            np.minimum(
                available,
                remaining,
            ),
            0,
        ).astype(np.int16)

        minimized_term += (
            take
            * np.where(
                np.isfinite(coefficient),
                coefficient,
                0.0,
            )
        )

        allocation[
            rows,
            phase,
        ] = take

        remaining -= take

    return allocation, minimized_term


def transition_generator_term(
    cfg: Config,
    model: PhaseTypeModel,
    data: ModelData,
    continuation_next: np.ndarray,
    lam: float,
    phase_allocation: np.ndarray,
) -> np.ndarray:
    n_states = len(data.states)
    current = continuation_next

    term = np.zeros(
        n_states,
        dtype=float,
    )

    not_full = (
        data.totals
        < cfg.n_batteries
    )

    for phase, alpha in enumerate(
        model.initial_probabilities
    ):
        if alpha <= 0.0:
            continue

        target = data.arrival_next[
            not_full,
            phase,
        ]

        term[not_full] += (
            lam
            * float(alpha)
            * (
                current[target]
                - current[not_full]
            )
        )

    for phase, rate in enumerate(
        model.phase_rates
    ):
        valid = (
            phase_allocation[:, phase]
            > 0.0
        )

        if not np.any(valid):
            continue

        target = data.service_next[
            valid,
            phase,
        ]

        term[valid] += (
            phase_allocation[
                valid,
                phase,
            ]
            * float(rate)
            * (
                current[target]
                - current[valid]
            )
        )

    return term

def solve_optimal_value_numpy(
    cfg: Config,
    model: PhaseTypeModel,
    data: ModelData,
) -> Tuple[float, float, float, int, float]:
    started = time.perf_counter()

    n_states = len(data.states)

    value_next = np.zeros(
        n_states,
        dtype=float,
    )
    charging_next = np.zeros(
        n_states,
        dtype=float,
    )

    full = (
        data.totals
        == cfg.n_batteries
    )
    not_full = ~full

    for step in range(
        cfg.n_steps - 1,
        -1,
        -1,
    ):
        t = step * cfg.dt
        lam = arrival_rate(t)

        operating_rate = (
            cfg.charger_power_kw
            * electricity_price(
                cfg,
                t,
            )
        )

        arrival_term = np.zeros(
            n_states,
            dtype=float,
        )

        for phase, alpha in enumerate(
            model.initial_probabilities
        ):
            if alpha <= 0.0:
                continue

            target = data.arrival_next[
                not_full,
                phase,
            ]

            arrival_term[not_full] += (
                lam
                * float(alpha)
                * (
                    value_next[target]
                    - value_next[not_full]
                )
            )

        arrival_term[full] = (
            cfg.lost_demand_penalty
            * lam
        )

        allocation, charging_term = (
            minimize_charging_term(
                cfg,
                model,
                data,
                value_next,
                operating_rate,
            )
        )

        value_next = (
            value_next
            + cfg.dt
            * (
                arrival_term
                + charging_term
            )
        )

        total_u = allocation.sum(axis=1)

        charging_generator = (
            transition_generator_term(
                cfg,
                model,
                data,
                charging_next,
                lam,
                allocation,
            )
        )

        charging_next = (
            charging_next
            + cfg.dt
            * (
                operating_rate
                * total_u
                + charging_generator
            )
        )

    optimal_value = float(
        value_next[
            data.empty_state_id
        ]
    )
    charging_value = float(
        charging_next[
            data.empty_state_id
        ]
    )
    lost_value = (
        optimal_value
        - charging_value
    )

    return (
        optimal_value,
        charging_value,
        lost_value,
        n_states,
        time.perf_counter() - started,
    )


def evaluate_exponential_policy_numpy(
    cfg: Config,
    model: PhaseTypeModel,
    data: ModelData,
    exponential_policy: np.ndarray,
) -> float:
    n_states = len(data.states)

    value_next = np.zeros(
        n_states,
        dtype=float,
    )

    full = (
        data.totals
        == cfg.n_batteries
    )

    positive_total = (
        data.totals > 0
    )

    phase_shares = np.zeros_like(
        data.states,
        dtype=float,
    )

    phase_shares[positive_total] = (
        data.states[positive_total]
        / data.totals[
            positive_total,
            None,
        ]
    )

    for step in range(
        cfg.n_steps - 1,
        -1,
        -1,
    ):
        t = step * cfg.dt
        lam = arrival_rate(t)

        operating_rate = (
            cfg.charger_power_kw
            * electricity_price(
                cfg,
                t,
            )
        )

        total_u = exponential_policy[
            step,
            data.totals,
        ].astype(float)

        phase_allocation = (
            phase_shares
            * total_u[:, None]
        )

        generator = (
            transition_generator_term(
                cfg,
                model,
                data,
                value_next,
                lam,
                phase_allocation,
            )
        )

        immediate = (
            operating_rate
            * total_u
            + cfg.lost_demand_penalty
            * lam
            * full.astype(float)
        )

        value_next = (
            value_next
            + cfg.dt
            * (
                immediate
                + generator
            )
        )

    return float(
        value_next[
            data.empty_state_id
        ]
    )

if NUMBA_AVAILABLE:

    @numba.njit(
        cache=True,
        parallel=True,
        fastmath=False,
    )
    def _solve_optimal_value_numba(
        states,
        totals,
        arrival_next,
        service_next,
        initial_probabilities,
        phase_rates,
        arrival_rates,
        operating_rates,
        n_batteries,
        n_chargers,
        lost_demand_penalty,
        dt,
        numerical_tolerance,
        empty_state_id,
    ):
        n_states = states.shape[0]
        n_phases = states.shape[1]
        n_steps = arrival_rates.shape[0]

        value_old = np.zeros(
            n_states,
            dtype=np.float64,
        )
        value_new = np.empty(
            n_states,
            dtype=np.float64,
        )

        charging_old = np.zeros(
            n_states,
            dtype=np.float64,
        )
        charging_new = np.empty(
            n_states,
            dtype=np.float64,
        )

        for step in range(
            n_steps - 1,
            -1,
            -1,
        ):
            lam = arrival_rates[step]
            operating_rate = (
                operating_rates[step]
            )

            for sid in prange(n_states):
                total = totals[sid]

                value_here = value_old[sid]
                charging_here = (
                    charging_old[sid]
                )

                arrival_value_term = 0.0
                arrival_charging_term = 0.0

                if total == n_batteries:
                    arrival_value_term = (
                        lost_demand_penalty
                        * lam
                    )
                else:
                    for phase in range(
                        n_phases
                    ):
                        alpha = (
                            initial_probabilities[
                                phase
                            ]
                        )

                        if alpha <= 0.0:
                            continue

                        target = arrival_next[
                            sid,
                            phase,
                        ]

                        arrival_value_term += (
                            lam
                            * alpha
                            * (
                                value_old[target]
                                - value_here
                            )
                        )

                        arrival_charging_term += (
                            lam
                            * alpha
                            * (
                                charging_old[
                                    target
                                ]
                                - charging_here
                            )
                        )
                remaining = n_chargers
                minimized_term = 0.0
                charging_service_term = 0.0
                total_u = 0

                used_mask = 0

                while remaining > 0:
                    best_phase = -1
                    best_coefficient = 1.0e300
                    best_target = -1

                    for phase in range(
                        n_phases
                    ):
                        if (
                            used_mask
                            & (1 << phase)
                        ) != 0:
                            continue

                        available = states[
                            sid,
                            phase,
                        ]

                        if available <= 0:
                            continue

                        target = service_next[
                            sid,
                            phase,
                        ]

                        coefficient = (
                            operating_rate
                            + phase_rates[
                                phase
                            ]
                            * (
                                value_old[target]
                                - value_here
                            )
                        )

                        if (
                            coefficient
                            < best_coefficient
                        ):
                            best_coefficient = (
                                coefficient
                            )
                            best_phase = phase
                            best_target = target

                    if best_phase < 0:
                        break

                    used_mask |= (
                        1 << best_phase
                    )

                    if (
                        best_coefficient
                        >= -numerical_tolerance
                    ):
                        break

                    available = states[
                        sid,
                        best_phase,
                    ]

                    if available < remaining:
                        take = available
                    else:
                        take = remaining

                    minimized_term += (
                        take
                        * best_coefficient
                    )

                    charging_service_term += (
                        take
                        * phase_rates[
                            best_phase
                        ]
                        * (
                            charging_old[
                                best_target
                            ]
                            - charging_here
                        )
                    )

                    total_u += take
                    remaining -= take


                value_new[sid] = (
                    value_here
                    + dt
                    * (
                        arrival_value_term
                        + minimized_term
                    )
                )

                charging_new[sid] = (
                    charging_here
                    + dt
                    * (
                        operating_rate
                        * total_u
                        + arrival_charging_term
                        + charging_service_term
                    )
                )

            tmp = value_old
            value_old = value_new
            value_new = tmp

            tmp = charging_old
            charging_old = charging_new
            charging_new = tmp

        return (
            value_old[empty_state_id],
            charging_old[empty_state_id],
        )


    @numba.njit(
        cache=True,
        parallel=True,
        fastmath=False,
    )
    def _evaluate_exponential_policy_numba(
        states,
        totals,
        arrival_next,
        service_next,
        initial_probabilities,
        phase_rates,
        exponential_policy,
        arrival_rates,
        operating_rates,
        n_batteries,
        lost_demand_penalty,
        dt,
        empty_state_id,
    ):
        n_states = states.shape[0]
        n_phases = states.shape[1]
        n_steps = arrival_rates.shape[0]

        value_old = np.zeros(
            n_states,
            dtype=np.float64,
        )
        value_new = np.empty(
            n_states,
            dtype=np.float64,
        )

        for step in range(
            n_steps - 1,
            -1,
            -1,
        ):
            lam = arrival_rates[step]
            operating_rate = (
                operating_rates[step]
            )

            for sid in prange(n_states):
                total = totals[sid]
                value_here = value_old[sid]

                total_u = float(
                    exponential_policy[
                        step,
                        total,
                    ]
                )

                generator = 0.0

                if total < n_batteries:
                    for phase in range(
                        n_phases
                    ):
                        alpha = (
                            initial_probabilities[
                                phase
                            ]
                        )

                        if alpha <= 0.0:
                            continue

                        target = arrival_next[
                            sid,
                            phase,
                        ]

                        generator += (
                            lam
                            * alpha
                            * (
                                value_old[target]
                                - value_here
                            )
                        )

                if (
                    total > 0
                    and total_u > 0.0
                ):
                    inverse_total = (
                        1.0 / total
                    )

                    for phase in range(
                        n_phases
                    ):
                        count = states[
                            sid,
                            phase,
                        ]

                        if count <= 0:
                            continue

                        phase_u = (
                            total_u
                            * count
                            * inverse_total
                        )

                        target = service_next[
                            sid,
                            phase,
                        ]

                        generator += (
                            phase_u
                            * phase_rates[
                                phase
                            ]
                            * (
                                value_old[target]
                                - value_here
                            )
                        )

                immediate = (
                    operating_rate
                    * total_u
                )

                if total == n_batteries:
                    immediate += (
                        lost_demand_penalty
                        * lam
                    )

                value_new[sid] = (
                    value_here
                    + dt
                    * (
                        immediate
                        + generator
                    )
                )

            tmp = value_old
            value_old = value_new
            value_new = tmp

        return value_old[
            empty_state_id
        ]


def solve_optimal_value(
    cfg: Config,
    model: PhaseTypeModel,
    data: ModelData,
) -> Tuple[float, float, float, int, float]:

    if not NUMBA_AVAILABLE:
        return solve_optimal_value_numpy(
            cfg,
            model,
            data,
        )

    arrival_rates, operating_rates = (
        build_time_profiles(cfg)
    )

    started = time.perf_counter()

    optimal_value, charging_value = (
        _solve_optimal_value_numba(
            data.states,
            data.totals,
            data.arrival_next,
            data.service_next,
            model.initial_probabilities.astype(
                np.float64,
                copy=False,
            ),
            model.phase_rates.astype(
                np.float64,
                copy=False,
            ),
            arrival_rates,
            operating_rates,
            cfg.n_batteries,
            cfg.n_chargers,
            cfg.lost_demand_penalty,
            cfg.dt,
            cfg.numerical_tolerance,
            data.empty_state_id,
        )
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    lost_value = (
        optimal_value
        - charging_value
    )

    return (
        float(optimal_value),
        float(charging_value),
        float(lost_value),
        len(data.states),
        elapsed,
    )


def evaluate_exponential_policy(
    cfg: Config,
    model: PhaseTypeModel,
    data: ModelData,
    exponential_policy: np.ndarray,
) -> float:

    if not NUMBA_AVAILABLE:
        return evaluate_exponential_policy_numpy(
            cfg,
            model,
            data,
            exponential_policy,
        )

    arrival_rates, operating_rates = (
        build_time_profiles(cfg)
    )

    value = (
        _evaluate_exponential_policy_numba(
            data.states,
            data.totals,
            data.arrival_next,
            data.service_next,
            model.initial_probabilities.astype(
                np.float64,
                copy=False,
            ),
            model.phase_rates.astype(
                np.float64,
                copy=False,
            ),
            exponential_policy,
            arrival_rates,
            operating_rates,
            cfg.n_batteries,
            cfg.lost_demand_penalty,
            cfg.dt,
            data.empty_state_id,
        )
    )

    return float(value)



def solve_exponential_benchmark(
    cfg: Config,
) -> Tuple[
    float,
    float,
    float,
    np.ndarray,
    float,
]:
    started = time.perf_counter()

    n_states = (
        cfg.n_batteries + 1
    )

    replenishment_rate = (
        1.0
        / cfg.mean_replenishment_hours
    )

    states = np.arange(
        n_states,
        dtype=np.int16,
    )

    maximum_action = np.minimum(
        states,
        cfg.n_chargers,
    ).astype(np.int16)

    policy = np.zeros(
        (
            cfg.n_steps,
            n_states,
        ),
        dtype=np.int16,
    )

    value_next = np.zeros(
        n_states,
        dtype=float,
    )
    charging_next = np.zeros(
        n_states,
        dtype=float,
    )

    for step in range(
        cfg.n_steps - 1,
        -1,
        -1,
    ):
        t = step * cfg.dt
        lam = arrival_rate(t)

        operating_rate = (
            cfg.charger_power_kw
            * electricity_price(
                cfg,
                t,
            )
        )

        arrival_term = np.empty(
            n_states,
            dtype=float,
        )

        arrival_term[:-1] = (
            lam
            * (
                value_next[1:]
                - value_next[:-1]
            )
        )

        arrival_term[-1] = (
            cfg.lost_demand_penalty
            * lam
        )

        coefficient = np.full(
            n_states,
            np.inf,
            dtype=float,
        )

        coefficient[1:] = (
            operating_rate
            + replenishment_rate
            * (
                value_next[:-1]
                - value_next[1:]
            )
        )

        action = np.where(
            coefficient
            < -cfg.numerical_tolerance,
            maximum_action,
            0,
        ).astype(np.int16)

        policy[step] = action

        charging_term = (
            action
            * np.where(
                np.isfinite(
                    coefficient
                ),
                coefficient,
                0.0,
            )
        )

        value_next = (
            value_next
            + cfg.dt
            * (
                arrival_term
                + charging_term
            )
        )

        charging_generator = np.zeros(
            n_states,
            dtype=float,
        )

        charging_generator[:-1] += (
            lam
            * (
                charging_next[1:]
                - charging_next[:-1]
            )
        )

        charging_generator[1:] += (
            action[1:]
            * replenishment_rate
            * (
                charging_next[:-1]
                - charging_next[1:]
            )
        )

        charging_next = (
            charging_next
            + cfg.dt
            * (
                operating_rate
                * action
                + charging_generator
            )
        )

    value = float(
        value_next[0]
    )
    charging = float(
        charging_next[0]
    )
    lost = value - charging

    return (
        value,
        charging,
        lost,
        policy,
        time.perf_counter() - started,
    )



def warm_up_numba() -> None:
    if not NUMBA_AVAILABLE:
        return

    if NUMBA_THREADS is not None:
        requested = int(NUMBA_THREADS)
        available = (
            numba.config.NUMBA_NUM_THREADS
        )
        numba.set_num_threads(
            min(
                requested,
                available,
            )
        )

    print(
        f"Numba available; "
        f"using {numba.get_num_threads()} thread(s)."
    )
    print(
        "Compiling Numba kernels..."
    )

    states = np.asarray(
        [
            [0],
            [1],
        ],
        dtype=np.int16,
    )

    totals = np.asarray(
        [0, 1],
        dtype=np.int16,
    )

    arrival_next = np.asarray(
        [
            [1],
            [-1],
        ],
        dtype=np.int32,
    )

    service_next = np.asarray(
        [
            [-1],
            [0],
        ],
        dtype=np.int32,
    )

    alpha = np.asarray(
        [1.0],
        dtype=np.float64,
    )

    rates = np.asarray(
        [1.0],
        dtype=np.float64,
    )

    arrivals = np.asarray(
        [1.0],
        dtype=np.float64,
    )

    operating = np.asarray(
        [1.0],
        dtype=np.float64,
    )

    _solve_optimal_value_numba(
        states,
        totals,
        arrival_next,
        service_next,
        alpha,
        rates,
        arrivals,
        operating,
        1,
        1,
        1.0,
        0.01,
        1e-11,
        0,
    )

    policy = np.zeros(
        (1, 2),
        dtype=np.int16,
    )

    _evaluate_exponential_policy_numba(
        states,
        totals,
        arrival_next,
        service_next,
        alpha,
        rates,
        policy,
        arrivals,
        operating,
        1,
        1.0,
        0.01,
        0,
    )

    print(
        "Numba kernels compiled."
    )


def configure_plots() -> None:
    fonts = {
        font.name
        for font in font_manager.fontManager.ttflist
    }

    family = (
        "Times New Roman"
        if "Times New Roman" in fonts
        else "Nimbus Roman"
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
            "legend.fontsize": 24,
            "figure.dpi": 140,
            "savefig.dpi": 420,
        }
    )


def scv_tick_labels(
    results: Sequence[Result],
) -> List[str]:
    labels: List[str] = []

    for result in results:
        if result.family == "Erlang":
            labels.append(
                rf"$1/{result.phases}$"
            )
        else:
            labels.append(
                f"{result.target_scv:g}"
            )

    return labels


def format_scv_axis(
    ax: plt.Axes,
    ordered: Sequence[Result],
    show_xlabel: bool,
) -> None:
    x = np.asarray(
        [
            result.actual_scv
            for result in ordered
        ]
    )

    ax.axvline(
        1.0,
        color="gray",
        linestyle=":",
        linewidth=1.4,
    )

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        scv_tick_labels(ordered)
    )
    ax.xaxis.set_minor_formatter(
        NullFormatter()
    )

    if show_xlabel:
        ax.set_xlabel(
            "SCV of replenishment time"
        )

    ax.margins(y=0.08)
    ax.grid(alpha=0.2)


def plot_results(
    results: Sequence[Result],
    output_dir: Path,
) -> Tuple[Path, Path]:

    configure_plots()

    ordered = sorted(
        results,
        key=lambda result: result.actual_scv,
    )

    x = np.asarray(
        [
            result.actual_scv
            for result in ordered
        ]
    )

    component_series = {
        "Total cost": (
            np.asarray(
                [
                    r.optimal_expected_cost
                    for r in ordered
                ]
            ),
            "#222222",
            "o",
        ),
        "Charging cost": (
            np.asarray(
                [
                    r.optimal_charging_cost
                    for r in ordered
                ]
            ),
            "#2474b5",
            "s",
        ),
        "Lost-demand cost": (
            np.asarray(
                [
                    r.optimal_lost_demand_cost
                    for r in ordered
                ]
            ),
            "#c53b3b",
            "^",
        ),
    }

    misspecification_percent = (
        np.asarray(
            [
                r.misspecification_loss_percent
                for r in ordered
            ]
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cost_fig, cost_ax = plt.subplots(
        figsize=(10, 6.1)
    )

    for (
        label,
        (
            absolute,
            color,
            marker,
        ),
    ) in component_series.items():

        cost_ax.plot(
            x,
            absolute,
            color=color,
            marker=marker,
            markersize=7.5,
            linewidth=2.0,
            label=label,
        )

    format_scv_axis(
        cost_ax,
        ordered,
        show_xlabel=True,
    )

    cost_ax.set_ylabel(
        "Optimal expected cost (USD)"
    )

    cost_ax.legend(
        loc="center left",
        bbox_to_anchor=(0.02, 0.3),
        frameon=True,
    )

    cost_fig.tight_layout()

    cost_path = (
        output_dir
        / "ph_scv_optimal_cost.png"
    )

    cost_fig.savefig(
        cost_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(cost_fig)

    loss_fig, loss_ax = plt.subplots(
        figsize=(10, 6.1)
    )

    loss_ax.plot(
        x,
        misspecification_percent,
        color="#6a3d9a",
        marker="o",
        markersize=8,
        linewidth=2.1,
    )

    format_scv_axis(
        loss_ax,
        ordered,
        show_xlabel=True,
    )

    loss_ax.axhline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=1.4,
    )

    loss_ax.set_ylabel(
        "Relative optimality loss (%)"
    )

    loss_fig.tight_layout()

    loss_path = (
        output_dir
        / "ph_scv_policy_loss.png"
    )

    loss_fig.savefig(
        loss_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(loss_fig)

    return cost_path, loss_path


def write_csv(
    results: Sequence[Result],
    output_dir: Path,
) -> Path:
    path = (
        output_dir
        / "ph_scv_objective.csv"
    )

    fieldnames = list(
        Result.__dataclass_fields__.keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in sorted(
            results,
            key=lambda item: item.actual_scv,
        ):
            writer.writerow(
                asdict(result)
            )

    return path


def write_npz(
    results: Sequence[Result],
    output_dir: Path,
) -> Path:
    ordered = sorted(
        results,
        key=lambda result: result.actual_scv,
    )

    path = (
        output_dir
        / "ph_scv_objective.npz"
    )

    np.savez_compressed(
        path,
        distributions=np.asarray(
            [
                result.distribution
                for result in ordered
            ]
        ),
        families=np.asarray(
            [
                result.family
                for result in ordered
            ]
        ),
        scv=np.asarray(
            [
                result.actual_scv
                for result in ordered
            ]
        ),
        mean_hours=np.asarray(
            [
                result.mean_hours
                for result in ordered
            ]
        ),
        optimal_charging_cost=np.asarray(
            [
                result.optimal_charging_cost
                for result in ordered
            ]
        ),
        optimal_lost_demand_cost=np.asarray(
            [
                result.optimal_lost_demand_cost
                for result in ordered
            ]
        ),
        optimal_expected_cost=np.asarray(
            [
                result.optimal_expected_cost
                for result in ordered
            ]
        ),
        relative_to_exponential_percent=np.asarray(
            [
                result.relative_to_exponential_percent
                for result in ordered
            ]
        ),
        charging_relative_to_exponential_percent=np.asarray(
            [
                result.charging_relative_to_exponential_percent
                for result in ordered
            ]
        ),
        lost_relative_to_exponential_percent=np.asarray(
            [
                result.lost_relative_to_exponential_percent
                for result in ordered
            ]
        ),
        exponential_policy_expected_cost=np.asarray(
            [
                result.exponential_policy_expected_cost
                for result in ordered
            ]
        ),
        misspecification_loss_absolute=np.asarray(
            [
                result.misspecification_loss_absolute
                for result in ordered
            ]
        ),
        misspecification_loss_percent=np.asarray(
            [
                result.misspecification_loss_percent
                for result in ordered
            ]
        ),
    )

    return path


def parse_distribution_argument(
    raw: str,
) -> Tuple[str, ...]:
    values = tuple(
        value.strip()
        for value in raw.split(",")
        if value.strip()
    )

    if not values:
        raise argparse.ArgumentTypeError(
            "The distribution argument cannot be empty."
        )

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--horizon",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=REQUESTED_DT,
    )

    parser.add_argument(
        "--distributions",
        type=parse_distribution_argument,
        default=None,
        help=(
            "Optional comma-separated override, for example "
            "Erlang-3,Erlang-2,Exponential,Hyper-2."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    warm_up_numba()

    selected_names = (
        tuple(DISTRIBUTIONS)
        if args.distributions is None
        else args.distributions
    )

    requested_cfg = Config(
        horizon_hours=args.horizon,
        dt=args.dt,
    )

    models = build_models(
        requested_cfg,
        selected_names,
    )

    cfg = choose_common_time_step(
        requested_cfg,
        models,
    )

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Exact PH-CTMDP SCV experiment: "
        f"N={cfg.n_batteries}, "
        f"K={cfg.n_chargers}, "
        f"horizon={cfg.represented_horizon_hours:.2f} h, "
        f"dt={cfg.dt:g}, "
        f"mean={cfg.mean_replenishment_hours:g} h"
    )

    print(
        "Selected distributions: "
        + ", ".join(selected_names)
    )

    if NUMBA_AVAILABLE:
        print(
            "PH backward recursions: Numba parallel implementation."
        )
    else:
        print(
            "PH backward recursions: NumPy fallback implementation."
        )

    (
        exponential_value,
        exponential_charging,
        exponential_lost,
        exponential_policy,
        exponential_elapsed,
    ) = solve_exponential_benchmark(cfg)

    print(
        json.dumps(
            {
                "exponential_benchmark_value":
                    exponential_value,
                "exponential_benchmark_charging_cost":
                    exponential_charging,
                "exponential_benchmark_lost_demand_cost":
                    exponential_lost,
                "benchmark_elapsed_seconds":
                    exponential_elapsed,
                "cross_evaluation_allocation": (
                    "uniform random allocation among depleted units when i>K"
                ),
            },
            indent=2,
        )
    )

    temporary: List[
        Tuple[
            PhaseTypeModel,
            float,
            float,
            float,
            float,
            int,
            float,
        ]
    ] = []

    for model in sorted(
        models,
        key=lambda item: item.actual_scv,
    ):
        predicted_states = number_of_states(
            cfg.n_batteries,
            model.n_phases,
        )

        print(
            f"Solving {model.name}: "
            f"SCV={model.actual_scv:g}, "
            f"phases={model.n_phases}, "
            f"states={predicted_states:,}"
        )

        build_started = (
            time.perf_counter()
        )

        data = build_model_data(
            cfg,
            model,
        )

        build_elapsed = (
            time.perf_counter()
            - build_started
        )

        (
            value,
            charging_cost,
            lost_cost,
            n_states,
            solve_elapsed,
        ) = solve_optimal_value(
            cfg,
            model,
            data,
        )

        exp_policy_value = (
            evaluate_exponential_policy(
                cfg,
                model,
                data,
                exponential_policy,
            )
        )

        elapsed = (
            build_elapsed
            + solve_elapsed
        )

        temporary.append(
            (
                model,
                value,
                charging_cost,
                lost_cost,
                exp_policy_value,
                n_states,
                elapsed,
            )
        )

        print(
            json.dumps(
                {
                    "distribution":
                        model.name,
                    "scv":
                        model.actual_scv,
                    "states":
                        n_states,
                    "optimal_charging_cost":
                        charging_cost,
                    "optimal_lost_demand_cost":
                        lost_cost,
                    "optimal_expected_cost":
                        value,
                    "exponential_policy_expected_cost":
                        exp_policy_value,
                    "misspecification_loss_absolute":
                        exp_policy_value - value,
                    "state_build_elapsed_seconds":
                        build_elapsed,
                    "optimal_solve_elapsed_seconds":
                        solve_elapsed,
                    "elapsed_seconds":
                        elapsed,
                },
                indent=2,
            )
        )

    def relative_change(
        value: float,
        benchmark: float,
    ) -> float:
        if (
            abs(benchmark)
            <= cfg.numerical_tolerance
        ):
            if (
                abs(value)
                <= cfg.numerical_tolerance
            ):
                return 0.0
            return math.nan

        return (
            100.0
            * (
                value
                - benchmark
            )
            / benchmark
        )

    def nonnegative_loss(
        policy_value: float,
        optimum: float,
    ) -> float:
        loss = (
            policy_value
            - optimum
        )

        scale = max(
            1.0,
            abs(policy_value),
            abs(optimum),
        )

        if loss < -1e-7 * scale:
            raise RuntimeError(
                "The evaluated exponential policy is cheaper than the "
                f"computed optimum by {-loss:.6g}; check the recursion."
            )

        return (
            0.0
            if abs(loss)
            <= 1e-10 * scale
            else loss
        )

    results = [
        Result(
            distribution=model.name,
            family=model.family,
            specification=model.specification,
            target_scv=model.target_scv,
            actual_scv=model.actual_scv,
            mean_hours=model.mean_hours,
            variance_hours2=model.variance_hours2,
            phases=model.n_phases,
            states=n_states,
            time_step=cfg.dt,
            time_steps=cfg.n_steps,
            optimal_charging_cost=charging_cost,
            optimal_lost_demand_cost=lost_cost,
            optimal_expected_cost=value,
            relative_to_exponential_percent=(
                relative_change(
                    value,
                    exponential_value,
                )
            ),
            charging_relative_to_exponential_percent=(
                relative_change(
                    charging_cost,
                    exponential_charging,
                )
            ),
            lost_relative_to_exponential_percent=(
                relative_change(
                    lost_cost,
                    exponential_lost,
                )
            ),
            exponential_policy_expected_cost=(
                exp_policy_value
            ),
            misspecification_loss_absolute=(
                nonnegative_loss(
                    exp_policy_value,
                    value,
                )
            ),
            misspecification_loss_percent=(
                100.0
                * nonnegative_loss(
                    exp_policy_value,
                    value,
                )
                / value
                if abs(value)
                > cfg.numerical_tolerance
                else math.nan
            ),
            elapsed_seconds=elapsed,
        )
        for (
            model,
            value,
            charging_cost,
            lost_cost,
            exp_policy_value,
            n_states,
            elapsed,
        ) in temporary
    ]

    figures = plot_results(
        results,
        output_dir,
    )

    table = write_csv(
        results,
        output_dir,
    )

    archive = write_npz(
        results,
        output_dir,
    )

    for figure in figures:
        print(figure)

    print(table)
    print(archive)


if __name__ == "__main__":
    main()
