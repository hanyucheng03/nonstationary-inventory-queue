from __future__ import annotations



import argparse

import csv

import itertools

import json

import math

import time

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Dict, Iterable, List, Sequence, Tuple



import matplotlib.pyplot as plt

import numpy as np

from matplotlib import font_manager

from matplotlib.ticker import NullFormatter

from scipy import sparse

from scipy.special import gammainc

from scipy.stats import binom, poisson





TARGET_SCVS = tuple(float(x) for x in [2, 4, 6, 8, 10])



ERLANG_BRANCH_ORDER = 2



BASE_BIMODAL_UNIT_HOURS = 0.39



GRID_REFINEMENT = 2

COMMON_DT_HOURS = BASE_BIMODAL_UNIT_HOURS / GRID_REFINEMENT



MAX_EXACT_STATES = 10_000_000_000



OUTPUT_DIR = (

    Path(__file__).resolve().parent.parent

    / "results/bimodal_h2_erlang_common_grid"

)





@dataclass(frozen=True)

class Config:

    n_batteries: int = 4

    n_chargers: int = 3

    horizon_hours: float = 120.0

    mean_replenishment_hours: float = 0.78

    lost_demand_penalty: float = 3.01

    charger_power_kw: float = 41.0

    high_electricity_price: float = 0.222755

    low_electricity_price: float = 0.067881

    dt: float = COMMON_DT_HOURS

    erlang_order: int = ERLANG_BRANCH_ORDER

    numerical_tolerance: float = 1e-11



    @property

    def arrival_scale(self) -> float:

        return self.n_batteries / 23.0



    @property

    def steps(self) -> int:

        base_steps = int(round(self.horizon_hours / BASE_BIMODAL_UNIT_HOURS))

        return base_steps * GRID_REFINEMENT



    @property

    def represented_horizon_hours(self) -> float:

        return self.steps * self.dt



    @property

    def n_phases(self) -> int:

        return 2 * self.erlang_order





@dataclass(frozen=True)

class BimodalModel:

    target_scv: float

    low_quanta: int

    high_quanta: int

    p_low: float

    p_high: float

    mean_hours: float

    variance_hours2: float

    actual_scv: float



    @property

    def specification(self) -> str:

        return (

            f"P(S={self.low_quanta}*delta)={self.p_low:.10g}; "

            f"P(S={self.high_quanta}*delta)={self.p_high:.10g}; "

            f"mean={self.mean_hours:.10g}; SCV={self.actual_scv:.10g}"

        )





@dataclass(frozen=True)

class ComponentErlangModel:

    target_scv: float

    order: int

    p_fast: float

    p_slow: float

    rate_fast: float

    rate_slow: float

    mean_fast: float

    mean_slow: float

    mean_hours: float

    variance_hours2: float

    actual_scv: float



    @property

    def probabilities(self) -> np.ndarray:

        return np.asarray([self.p_fast, self.p_slow], dtype=float)



    @property

    def rates(self) -> np.ndarray:

        return np.asarray([self.rate_fast, self.rate_slow], dtype=float)



    @property

    def specification(self) -> str:

        return (

            f"branches=(Erlang-{self.order}, Erlang-{self.order}); "

            f"probabilities=({self.p_fast:.10g},{self.p_slow:.10g}); "

            f"rates=({self.rate_fast:.10g},{self.rate_slow:.10g}); "

            f"branch means=({self.mean_fast:.10g},{self.mean_slow:.10g}); "

            f"mean={self.mean_hours:.10g}; SCV={self.actual_scv:.10g}"

        )





@dataclass(frozen=True)

class BimodalData:

    short_residuals: np.ndarray

    short_counts: np.ndarray

    long_residuals: np.ndarray

    long_counts: np.ndarray

    totals: np.ndarray

    state_index: Dict[Tuple[int, ...], int]

    empty_state_id: int

    action_masks: Tuple[Tuple[int, ...], ...]

    action_total: np.ndarray

    service_successor: np.ndarray

    arrival_matrices: Tuple[sparse.csr_matrix, sparse.csr_matrix]

    expected_lost: Tuple[np.ndarray, np.ndarray]





@dataclass(frozen=True)

class PHData:

    states: np.ndarray

    totals: np.ndarray

    state_index: Dict[Tuple[int, ...], int]

    empty_state_id: int

    actions: np.ndarray

    service_matrices: Tuple[sparse.csr_matrix, ...]

    expected_service_charge_hours: np.ndarray

    arrival_matrices: Tuple[sparse.csr_matrix, sparse.csr_matrix]

    expected_lost: Tuple[np.ndarray, np.ndarray]





@dataclass(frozen=True)

class Result:

    scv: float

    n_batteries: int

    n_chargers: int

    common_dt_hours: float

    time_steps: int

    erlang_branch_order: int

    bimodal_low_quanta: int

    bimodal_high_quanta: int

    bimodal_low_hours: float

    bimodal_high_hours: float

    bimodal_p_low: float

    bimodal_p_high: float

    surrogate_p_fast: float

    surrogate_p_slow: float

    surrogate_rate_fast: float

    surrogate_rate_slow: float

    surrogate_actual_scv: float

    exact_states: int

    surrogate_states: int

    bimodal_optimal_charging_cost: float

    bimodal_optimal_lost_demand_cost: float

    bimodal_optimal_expected_cost: float

    surrogate_optimal_charging_cost: float

    surrogate_optimal_lost_demand_cost: float

    surrogate_optimal_expected_cost: float

    surrogate_policy_bimodal_expected_cost: float

    surrogate_policy_loss_absolute: float

    surrogate_policy_loss_percent: float

    surrogate_vs_bimodal_optimal_cost_gap_percent: float

    exact_elapsed_seconds: float

    surrogate_elapsed_seconds: float

    surrogate_policy_eval_elapsed_seconds: float



@dataclass(frozen=True)
class ExponentialData:
    states: np.ndarray
    service_matrices: Tuple[sparse.csr_matrix, ...]
    expected_service_charge_hours: np.ndarray
    arrival_matrices: Tuple[sparse.csr_matrix, sparse.csr_matrix]
    expected_lost: Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ExponentialResult:
    scv: float
    n_batteries: int
    n_chargers: int
    common_dt_hours: float
    time_steps: int
    bimodal_low_quanta: int
    bimodal_high_quanta: int
    bimodal_low_hours: float
    bimodal_high_hours: float
    bimodal_p_low: float
    bimodal_p_high: float
    exact_states: int
    exponential_states: int
    exponential_mean_hours: float
    exponential_rate: float
    bimodal_optimal_charging_cost: float
    bimodal_optimal_lost_demand_cost: float
    bimodal_optimal_expected_cost: float
    exponential_optimal_charging_cost: float
    exponential_optimal_lost_demand_cost: float
    exponential_optimal_expected_cost: float
    exponential_policy_bimodal_expected_cost: float
    exponential_policy_loss_absolute: float
    exponential_policy_loss_percent: float
    exponential_vs_bimodal_optimal_cost_gap_percent: float
    exact_elapsed_seconds: float
    exponential_elapsed_seconds: float
    exponential_policy_eval_elapsed_seconds: float


def arrival_rate(cfg: Config, t: float) -> float:

    baseline = 5.68 if (t % 24.0) < 9.0 else 14.51

    return cfg.arrival_scale * baseline





def electricity_price(cfg: Config, t: float) -> float:

    hour = t % 24.0

    if 15.0 <= hour < 20.0:

        return cfg.high_electricity_price

    return cfg.low_electricity_price





def number_of_typed_states(N: int, short_levels: int, long_levels: int) -> int:

    total = 0

    for s in range(N + 1):

        short_states = math.comb(short_levels + s - 1, s)

        for ell in range(N - s + 1):

            long_states = math.comb(long_levels + ell - 1, ell)

            total += short_states * long_states

    return total





def weak_compositions_at_most(total: int, parts: int) -> Iterable[Tuple[int, ...]]:

    """All nonnegative integer tuples of length parts with sum <= total."""

    for values in itertools.product(range(total + 1), repeat=parts):

        if sum(values) <= total:

            yield tuple(int(x) for x in values)





def weak_compositions_equal(total: int, parts: int) -> Iterable[Tuple[int, ...]]:

    """All nonnegative integer tuples of length parts with sum exactly total."""

    if parts == 1:

        yield (total,)

        return

    for first in range(total + 1):

        for tail in weak_compositions_equal(total - first, parts - 1):

            yield (first,) + tail





def multinomial_probability(counts: Sequence[int], probabilities: Sequence[float]) -> float:

    n = int(sum(counts))

    coefficient = math.factorial(n)

    for count in counts:

        coefficient //= math.factorial(int(count))

    out = float(coefficient)

    for count, probability in zip(counts, probabilities):

        if count:

            out *= probability ** int(count)

    return out





def make_bimodal(cfg: Config, target_scv: float) -> BimodalModel:

    if target_scv < 1.0:

        raise ValueError("This construction requires SCV >= 1.")



    high_multiplier_float = 4.0 * target_scv + 2.0

    high_multiplier = int(round(high_multiplier_float))

    if not np.isclose(high_multiplier_float, high_multiplier, atol=1e-12):

        raise ValueError("Current construction requires 4*SCV+2 to be integer.")



    low_hours = BASE_BIMODAL_UNIT_HOURS

    high_hours = high_multiplier * BASE_BIMODAL_UNIT_HOURS

    low_quanta_float = low_hours / cfg.dt

    high_quanta_float = high_hours / cfg.dt

    low = int(round(low_quanta_float))

    high = int(round(high_quanta_float))



    if not np.isclose(low_quanta_float, low, atol=1e-12):

        raise RuntimeError("Short service time is not aligned with the common grid.")

    if not np.isclose(high_quanta_float, high, atol=1e-12):

        raise RuntimeError("Long service time is not aligned with the common grid.")



    p_high = 1.0 / (high_multiplier - 1.0)

    p_low = 1.0 - p_high

    mean = p_low * low_hours + p_high * high_hours

    variance = p_low * (low_hours - mean) ** 2 + p_high * (high_hours - mean) ** 2

    actual_scv = variance / mean**2



    if not np.isclose(mean, cfg.mean_replenishment_hours, atol=1e-12):

        raise RuntimeError("Bimodal mean check failed.")

    if not np.isclose(actual_scv, target_scv, atol=1e-12):

        raise RuntimeError("Bimodal SCV check failed.")



    return BimodalModel(

        target_scv=target_scv,

        low_quanta=low,

        high_quanta=high,

        p_low=p_low,

        p_high=p_high,

        mean_hours=mean,

        variance_hours2=variance,

        actual_scv=actual_scv,

    )





def make_h2_erlang(cfg: Config, bimodal: BimodalModel) -> ComponentErlangModel:

    """Same branch probabilities/means as bimodal; Erlang-k within branches."""

    k = cfg.erlang_order

    if k < 1:

        raise ValueError("Erlang branch order must be positive.")



    mean_fast = bimodal.low_quanta * cfg.dt

    mean_slow = bimodal.high_quanta * cfg.dt

    p_fast = bimodal.p_low

    p_slow = bimodal.p_high

    rate_fast = k / mean_fast

    rate_slow = k / mean_slow



    mean = p_fast * mean_fast + p_slow * mean_slow

    second = (1.0 + 1.0 / k) * (

        p_fast * mean_fast**2 + p_slow * mean_slow**2

    )

    variance = second - mean**2

    actual_scv = variance / mean**2



    expected_scv = bimodal.actual_scv + (bimodal.actual_scv + 1.0) / k



    if not np.isclose(mean, bimodal.mean_hours, atol=1e-12):

        raise RuntimeError("Component-Erlang mean check failed.")

    if not np.isclose(actual_scv, expected_scv, atol=1e-10):

        raise RuntimeError(

            "Component-Erlang SCV identity check failed: "

            f"computed={actual_scv}, expected={expected_scv}."

        )



    return ComponentErlangModel(

        target_scv=bimodal.actual_scv,

        order=k,

        p_fast=p_fast,

        p_slow=p_slow,

        rate_fast=rate_fast,

        rate_slow=rate_slow,

        mean_fast=mean_fast,

        mean_slow=mean_slow,

        mean_hours=mean,

        variance_hours2=variance,

        actual_scv=actual_scv,

    )





def admitted_arrival_distribution(capacity: int, mean_arrivals: float) -> List[Tuple[int, float]]:

    if capacity == 0:

        return [(0, 1.0)]

    out = [(n, float(poisson.pmf(n, mean_arrivals))) for n in range(capacity)]

    out.append((capacity, float(poisson.sf(capacity - 1, mean_arrivals))))

    return out





def expected_lost_arrivals(capacity: int, mean_arrivals: float) -> float:

    if capacity == 0:

        return mean_arrivals

    return (

        mean_arrivals * float(poisson.sf(capacity - 1, mean_arrivals))

        - capacity * float(poisson.sf(capacity, mean_arrivals))

    )





def make_bimodal_state_key(

    short_values: Sequence[int], long_values: Sequence[int], N: int

) -> Tuple[int, ...]:

    short_ordered = tuple(sorted(int(x) for x in short_values if x > 0))

    long_ordered = tuple(sorted(int(x) for x in long_values if x > 0))

    if len(short_ordered) + len(long_ordered) > N:

        raise ValueError("Too many depleted batteries.")

    return (

        short_ordered + (0,) * (N - len(short_ordered))

        + long_ordered + (0,) * (N - len(long_ordered))

    )





def enumerate_bimodal_states(N: int, short_levels: int, long_levels: int):

    predicted = number_of_typed_states(N, short_levels, long_levels)

    short_residuals = np.zeros((predicted, N), dtype=np.int16)

    short_counts = np.empty(predicted, dtype=np.int8)

    long_residuals = np.zeros((predicted, N), dtype=np.int16)

    long_counts = np.empty(predicted, dtype=np.int8)

    state_index: Dict[Tuple[int, ...], int] = {}



    sid = 0

    for s in range(N + 1):

        for short_tuple in itertools.combinations_with_replacement(range(1, short_levels + 1), s):

            for ell in range(N - s + 1):

                for long_tuple in itertools.combinations_with_replacement(range(1, long_levels + 1), ell):

                    key = make_bimodal_state_key(short_tuple, long_tuple, N)

                    state_index[key] = sid

                    short_counts[sid] = s

                    long_counts[sid] = ell

                    if s:

                        short_residuals[sid, :s] = short_tuple

                    if ell:

                        long_residuals[sid, :ell] = long_tuple

                    sid += 1



    if sid != predicted:

        raise RuntimeError("Bimodal state count mismatch.")

    return short_residuals, short_counts, long_residuals, long_counts, state_index





def exact_action_templates(N: int, K: int):

    masks: List[Tuple[int, ...]] = []

    totals: List[int] = []

    for u in range(K + 1):

        for positions in itertools.combinations(range(N), u):

            masks.append(tuple(positions))

            totals.append(u)

    return tuple(masks), np.asarray(totals, dtype=np.int8)





def deterministic_bimodal_service_successor(

    cfg: Config,

    short_values: np.ndarray,

    short_count: int,

    long_values: np.ndarray,

    long_count: int,

    selected_positions: Tuple[int, ...],

    state_index: Dict[Tuple[int, ...], int],

) -> int:

    total_count = short_count + long_count

    if any(pos >= total_count for pos in selected_positions):

        return -1

    target_short = np.array(short_values[:short_count], copy=True)

    target_long = np.array(long_values[:long_count], copy=True)

    for pos in selected_positions:

        if pos < short_count:

            target_short[pos] -= 1

        else:

            target_long[pos - short_count] -= 1

    target_short = target_short[target_short > 0]

    target_long = target_long[target_long > 0]

    return state_index[make_bimodal_state_key(target_short, target_long, cfg.n_batteries)]





def build_bimodal_exact_service_successor(

    cfg: Config,

    short_residuals: np.ndarray,

    short_counts: np.ndarray,

    long_residuals: np.ndarray,

    long_counts: np.ndarray,

    state_index: Dict[Tuple[int, ...], int],

):

    masks, totals = exact_action_templates(cfg.n_batteries, cfg.n_chargers)

    successor = np.full((len(masks), len(short_counts)), -1, dtype=np.int32)

    for aid, selected_positions in enumerate(masks):

        for sid in range(len(short_counts)):

            successor[aid, sid] = deterministic_bimodal_service_successor(

                cfg,

                short_residuals[sid],

                int(short_counts[sid]),

                long_residuals[sid],

                int(long_counts[sid]),

                selected_positions,

                state_index,

            )

    return masks, totals, successor





def build_bimodal_arrival_matrix(

    cfg: Config,

    model: BimodalModel,

    short_residuals: np.ndarray,

    short_counts: np.ndarray,

    long_residuals: np.ndarray,

    long_counts: np.ndarray,

    state_index: Dict[Tuple[int, ...], int],

    lam: float,

):

    n_states = len(short_counts)

    mean_arrivals = lam * cfg.dt

    rows: List[int] = []

    cols: List[int] = []

    vals: List[float] = []

    expected_lost = np.zeros(n_states, dtype=float)



    for sid in range(n_states):

        s = int(short_counts[sid])

        ell = int(long_counts[sid])

        capacity = cfg.n_batteries - s - ell

        expected_lost[sid] = expected_lost_arrivals(capacity, mean_arrivals)

        accumulator: Dict[int, float] = {}



        for admitted, p_admitted in admitted_arrival_distribution(capacity, mean_arrivals):

            if p_admitted <= 0.0:

                continue

            for new_long in range(admitted + 1):

                new_short = admitted - new_long

                probability = p_admitted * float(binom.pmf(new_long, admitted, model.p_high))

                if probability <= 0.0:

                    continue

                target_short = list(map(int, short_residuals[sid, :s]))

                target_long = list(map(int, long_residuals[sid, :ell]))

                target_short.extend([model.low_quanta] * new_short)

                target_long.extend([model.high_quanta] * new_long)

                tid = state_index[make_bimodal_state_key(target_short, target_long, cfg.n_batteries)]

                accumulator[tid] = accumulator.get(tid, 0.0) + probability



        if not np.isclose(sum(accumulator.values()), 1.0, atol=2e-11):

            raise RuntimeError(f"Bimodal arrival row {sid} does not sum to 1.")

        for tid, probability in accumulator.items():

            rows.append(sid)

            cols.append(tid)

            vals.append(probability)



    return (

        sparse.csr_matrix((vals, (rows, cols)), shape=(n_states, n_states)),

        expected_lost,

    )





def build_bimodal_data(cfg: Config, model: BimodalModel) -> BimodalData:

    predicted = number_of_typed_states(cfg.n_batteries, model.low_quanta, model.high_quanta)

    if predicted > MAX_EXACT_STATES:

        raise RuntimeError(

            f"SCV={model.actual_scv:g} requires {predicted:,} exact states; "

            f"guard={MAX_EXACT_STATES:,}."

        )

    print(f"  bimodal exact states: {predicted:,}")



    short_residuals, short_counts, long_residuals, long_counts, state_index = (

        enumerate_bimodal_states(cfg.n_batteries, model.low_quanta, model.high_quanta)

    )

    totals = (short_counts + long_counts).astype(np.int8)

    empty_state_id = state_index[make_bimodal_state_key((), (), cfg.n_batteries)]

    masks, action_total, service_successor = build_bimodal_exact_service_successor(

        cfg, short_residuals, short_counts, long_residuals, long_counts, state_index

    )



    lam_low = arrival_rate(cfg, 0.0)

    lam_high = arrival_rate(cfg, 9.0)

    arrival_low, lost_low = build_bimodal_arrival_matrix(

        cfg, model, short_residuals, short_counts, long_residuals, long_counts, state_index, lam_low

    )

    arrival_high, lost_high = build_bimodal_arrival_matrix(

        cfg, model, short_residuals, short_counts, long_residuals, long_counts, state_index, lam_high

    )



    return BimodalData(

        short_residuals=short_residuals,

        short_counts=short_counts,

        long_residuals=long_residuals,

        long_counts=long_counts,

        totals=totals,

        state_index=state_index,

        empty_state_id=empty_state_id,

        action_masks=masks,

        action_total=action_total,

        service_successor=service_successor,

        arrival_matrices=(arrival_low, arrival_high),

        expected_lost=(lost_low, lost_high),

    )





def solve_bimodal_exact(cfg: Config, data: BimodalData):

    started = time.perf_counter()

    n_states = len(data.short_counts)

    value_next = np.zeros(n_states, dtype=float)

    charging_next = np.zeros(n_states, dtype=float)

    lam_low = arrival_rate(cfg, 0.0)



    for step in range(cfg.steps - 1, -1, -1):

        t = step * cfg.dt

        regime = 0 if np.isclose(arrival_rate(cfg, t), lam_low) else 1

        A = data.arrival_matrices[regime]

        after_arrival_value = A.dot(value_next)

        after_arrival_charging = A.dot(charging_next)

        post_service_value = (

            cfg.lost_demand_penalty * data.expected_lost[regime] + after_arrival_value

        )

        price_rate = cfg.charger_power_kw * electricity_price(cfg, t)

        best = np.full(n_states, np.inf, dtype=float)

        best_action = np.full(n_states, -1, dtype=np.int16)



        for aid in range(data.service_successor.shape[0]):

            successor = data.service_successor[aid]

            feasible = successor >= 0

            if not np.any(feasible):

                continue

            q = np.full(n_states, np.inf, dtype=float)

            q[feasible] = (

                price_rate * int(data.action_total[aid]) * cfg.dt

                + post_service_value[successor[feasible]]

            )

            improve = q < best - cfg.numerical_tolerance

            best[improve] = q[improve]

            best_action[improve] = aid



        if np.any(best_action < 0):

            raise RuntimeError("Bimodal model has a state without a feasible action.")



        charging_current = np.empty(n_states, dtype=float)

        for aid in range(data.service_successor.shape[0]):

            chosen = best_action == aid

            if not np.any(chosen):

                continue

            successor = data.service_successor[aid, chosen]

            charging_current[chosen] = (

                price_rate * int(data.action_total[aid]) * cfg.dt

                + after_arrival_charging[successor]

            )

        value_next = best

        charging_next = charging_current



    value = float(value_next[data.empty_state_id])

    charging = float(charging_next[data.empty_state_id])

    return value, charging, value - charging, n_states, time.perf_counter() - started





def enumerate_ph_states(N: int, n_phases: int):

    states = list(weak_compositions_at_most(N, n_phases))

    state_index = {state: sid for sid, state in enumerate(states)}

    return np.asarray(states, dtype=np.int16), state_index





def ph_action_templates(K: int, n_phases: int) -> np.ndarray:

    return np.asarray(list(weak_compositions_at_most(K, n_phases)), dtype=np.int8)





def selected_group_outcomes(

    state: Tuple[int, ...],

    branch: int,

    phase: int,

    selected: int,

    order: int,

    rate: float,

    dt: float,

) -> Dict[Tuple[int, ...], float]:

    """Transition distribution contributed by one selected phase group."""

    if selected == 0:

        return {state: 1.0}



    remaining = order - phase

    z = rate * dt

    probabilities = [float(poisson.pmf(j, z)) for j in range(remaining)]

    probabilities.append(float(poisson.sf(remaining - 1, z)))



    source = branch * order + phase

    output: Dict[Tuple[int, ...], float] = {}

    for counts in weak_compositions_equal(selected, remaining + 1):

        probability = multinomial_probability(counts, probabilities)

        if probability <= 0.0:

            continue

        target = list(state)

        target[source] -= selected

        for advance in range(remaining):

            target[source + advance] += counts[advance]

        key = tuple(target)

        output[key] = output.get(key, 0.0) + probability

    return output





def ph_service_row_distribution(

    cfg: Config,

    model: ComponentErlangModel,

    state: Tuple[int, ...],

    action: Tuple[int, ...],

) -> Dict[Tuple[int, ...], float]:

    distribution: Dict[Tuple[int, ...], float] = {state: 1.0}

    rates = (model.rate_fast, model.rate_slow)



    for branch in range(2):

        for phase in range(model.order):

            selected = int(action[branch * model.order + phase])

            if selected == 0:

                continue

            updated: Dict[Tuple[int, ...], float] = {}

            for current_state, current_probability in distribution.items():

                local = selected_group_outcomes(

                    current_state,

                    branch,

                    phase,

                    selected,

                    model.order,

                    rates[branch],

                    cfg.dt,

                )

                for target, conditional_probability in local.items():

                    updated[target] = updated.get(target, 0.0) + (

                        current_probability * conditional_probability

                    )

            distribution = updated

    return distribution





def build_ph_service_matrix(

    cfg: Config,

    model: ComponentErlangModel,

    states: np.ndarray,

    state_index: Dict[Tuple[int, ...], int],

    action: Tuple[int, ...],

) -> sparse.csr_matrix:

    n_states = len(states)

    rows: List[int] = []

    cols: List[int] = []

    vals: List[float] = []



    for sid, state_array in enumerate(states):

        state = tuple(int(x) for x in state_array)

        if any(action[j] > state[j] for j in range(len(action))):

            continue

        distribution = ph_service_row_distribution(cfg, model, state, action)

        if not np.isclose(sum(distribution.values()), 1.0, atol=2e-11):

            raise RuntimeError(f"PH service row {sid} does not sum to 1.")

        for target, probability in distribution.items():

            rows.append(sid)

            cols.append(state_index[target])

            vals.append(probability)



    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_states, n_states))





def expected_erlang_busy_time(remaining_phases: int, rate: float, dt: float) -> float:

    """E[min(Erlang(remaining_phases, rate), dt)]."""

    z = rate * dt

    return sum(float(gammainc(m + 1, z)) for m in range(remaining_phases)) / rate





def expected_ph_busy_hours_per_action(

    cfg: Config, model: ComponentErlangModel, action: Tuple[int, ...]

) -> float:

    total = 0.0

    rates = (model.rate_fast, model.rate_slow)

    for branch in range(2):

        for phase in range(model.order):

            selected = int(action[branch * model.order + phase])

            remaining = model.order - phase

            total += selected * expected_erlang_busy_time(remaining, rates[branch], cfg.dt)

    return total





def build_ph_arrival_matrix(

    cfg: Config,

    model: ComponentErlangModel,

    states: np.ndarray,

    state_index: Dict[Tuple[int, ...], int],

    lam: float,

):

    n_states = len(states)

    mean_arrivals = lam * cfg.dt

    rows: List[int] = []

    cols: List[int] = []

    vals: List[float] = []

    expected_lost = np.zeros(n_states, dtype=float)



    for sid, state_array in enumerate(states):

        state = tuple(int(x) for x in state_array)

        capacity = cfg.n_batteries - sum(state)

        expected_lost[sid] = expected_lost_arrivals(capacity, mean_arrivals)

        accumulator: Dict[int, float] = {}

        for admitted, p_admitted in admitted_arrival_distribution(capacity, mean_arrivals):

            if p_admitted <= 0.0:

                continue

            for new_fast in range(admitted + 1):

                new_slow = admitted - new_fast

                probability = p_admitted * float(

                    binom.pmf(new_fast, admitted, model.p_fast)

                )

                if probability <= 0.0:

                    continue

                target = list(state)

                target[0] += new_fast

                target[model.order] += new_slow

                tid = state_index[tuple(target)]

                accumulator[tid] = accumulator.get(tid, 0.0) + probability

        if not np.isclose(sum(accumulator.values()), 1.0, atol=2e-11):

            raise RuntimeError(f"PH arrival row {sid} does not sum to 1.")

        for tid, probability in accumulator.items():

            rows.append(sid)

            cols.append(tid)

            vals.append(probability)



    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_states, n_states)), expected_lost





def build_ph_data(cfg: Config, model: ComponentErlangModel) -> PHData:

    states, state_index = enumerate_ph_states(cfg.n_batteries, cfg.n_phases)

    totals = states.sum(axis=1).astype(np.int8)

    actions = ph_action_templates(cfg.n_chargers, cfg.n_phases)

    service_matrices: List[sparse.csr_matrix] = []

    expected_charge_hours: List[float] = []



    for action_array in actions:

        action = tuple(int(x) for x in action_array)

        service_matrices.append(

            build_ph_service_matrix(cfg, model, states, state_index, action)

        )

        expected_charge_hours.append(

            expected_ph_busy_hours_per_action(cfg, model, action)

        )



    lam_low = arrival_rate(cfg, 0.0)

    lam_high = arrival_rate(cfg, 9.0)

    arrival_low, lost_low = build_ph_arrival_matrix(cfg, model, states, state_index, lam_low)

    arrival_high, lost_high = build_ph_arrival_matrix(cfg, model, states, state_index, lam_high)



    return PHData(

        states=states,

        totals=totals,

        state_index=state_index,

        empty_state_id=state_index[(0,) * cfg.n_phases],

        actions=actions,

        service_matrices=tuple(service_matrices),

        expected_service_charge_hours=np.asarray(expected_charge_hours, dtype=float),

        arrival_matrices=(arrival_low, arrival_high),

        expected_lost=(lost_low, lost_high),

    )





def solve_ph_common_grid(cfg: Config, data: PHData):

    started = time.perf_counter()

    n_states = len(data.states)

    value_next = np.zeros(n_states, dtype=float)

    charging_next = np.zeros(n_states, dtype=float)

    policy = np.zeros((cfg.steps, n_states, cfg.n_phases), dtype=np.int8)

    lam_low = arrival_rate(cfg, 0.0)



    for step in range(cfg.steps - 1, -1, -1):

        t = step * cfg.dt

        regime = 0 if np.isclose(arrival_rate(cfg, t), lam_low) else 1

        A = data.arrival_matrices[regime]

        after_arrival_value = A.dot(value_next)

        after_arrival_charging = A.dot(charging_next)

        post_service_value = (

            cfg.lost_demand_penalty * data.expected_lost[regime] + after_arrival_value

        )

        price_rate = cfg.charger_power_kw * electricity_price(cfg, t)

        best = np.full(n_states, np.inf, dtype=float)

        best_action_id = np.full(n_states, -1, dtype=np.int16)



        for aid, action_array in enumerate(data.actions):

            feasible = np.all(data.states >= action_array, axis=1)

            if not np.any(feasible):

                continue

            expected_post = data.service_matrices[aid].dot(post_service_value)

            q = price_rate * data.expected_service_charge_hours[aid] + expected_post

            improve = feasible & (q < best - cfg.numerical_tolerance)

            best[improve] = q[improve]

            best_action_id[improve] = aid



        if np.any(best_action_id < 0):

            raise RuntimeError("PH-DTMDP has a state without a feasible action.")



        charging_current = np.empty(n_states, dtype=float)

        for aid, action_array in enumerate(data.actions):

            chosen = best_action_id == aid

            if not np.any(chosen):

                continue

            continuation = data.service_matrices[aid].dot(after_arrival_charging)

            charging_current[chosen] = (

                price_rate * data.expected_service_charge_hours[aid]

                + continuation[chosen]

            )

            policy[step, chosen, :] = action_array

        value_next = best

        charging_next = charging_current



    value = float(value_next[data.empty_state_id])

    charging = float(charging_next[data.empty_state_id])

    return value, charging, value - charging, n_states, policy, time.perf_counter() - started







def residual_to_phase(residual: int, total_quanta: int, order: int) -> int:

    """Map deterministic progress to the corresponding Erlang stage."""

    completed = total_quanta - residual

    phase = (completed * order) // total_quanta

    return min(order - 1, max(0, int(phase)))





def mapped_phase_positions(

    cfg: Config,

    model: BimodalModel,

    short_values: np.ndarray,

    short_count: int,

    long_values: np.ndarray,

    long_count: int,

) -> Tuple[Tuple[Tuple[str, int], ...], ...]:

    groups: List[List[Tuple[str, int]]] = [[] for _ in range(cfg.n_phases)]

    for pos in range(short_count):

        phase = residual_to_phase(int(short_values[pos]), model.low_quanta, cfg.erlang_order)

        groups[phase].append(("short", pos))

    for pos in range(long_count):

        phase = residual_to_phase(int(long_values[pos]), model.high_quanta, cfg.erlang_order)

        groups[cfg.erlang_order + phase].append(("long", pos))

    return tuple(tuple(group) for group in groups)





def mapped_ph_state_ids(

    cfg: Config, model: BimodalModel, bimodal_data: BimodalData, ph_data: PHData

) -> np.ndarray:

    ids = np.empty(len(bimodal_data.short_counts), dtype=np.int32)

    for sid in range(len(ids)):

        groups = mapped_phase_positions(

            cfg,

            model,

            bimodal_data.short_residuals[sid],

            int(bimodal_data.short_counts[sid]),

            bimodal_data.long_residuals[sid],

            int(bimodal_data.long_counts[sid]),

        )

        ids[sid] = ph_data.state_index[tuple(len(group) for group in groups)]

    return ids





def build_bimodal_randomized_service_matrix(

    cfg: Config,

    model: BimodalModel,

    data: BimodalData,

    action: Tuple[int, ...],

) -> sparse.csr_matrix:

    """Apply a phase-count action without using within-phase residual detail."""

    n_states = len(data.short_counts)

    rows: List[int] = []

    cols: List[int] = []

    vals: List[float] = []



    for sid in range(n_states):

        s = int(data.short_counts[sid])

        ell = int(data.long_counts[sid])

        groups = mapped_phase_positions(

            cfg,

            model,

            data.short_residuals[sid],

            s,

            data.long_residuals[sid],

            ell,

        )

        if any(action[g] > len(groups[g]) for g in range(cfg.n_phases)):

            continue



        choices_by_group = [

            list(itertools.combinations(groups[g], int(action[g])))

            for g in range(cfg.n_phases)

        ]

        denominator = math.prod(len(choices) for choices in choices_by_group)

        accumulator: Dict[int, int] = {}



        for chosen_groups in itertools.product(*choices_by_group):

            target_short = np.array(data.short_residuals[sid, :s], copy=True)

            target_long = np.array(data.long_residuals[sid, :ell], copy=True)

            for chosen_group in chosen_groups:

                for kind, pos in chosen_group:

                    if kind == "short":

                        target_short[pos] -= 1

                    else:

                        target_long[pos] -= 1

            target_short = target_short[target_short > 0]

            target_long = target_long[target_long > 0]

            key = make_bimodal_state_key(target_short, target_long, cfg.n_batteries)

            tid = data.state_index[key]

            accumulator[tid] = accumulator.get(tid, 0) + 1



        for tid, count in accumulator.items():

            rows.append(sid)

            cols.append(tid)

            vals.append(count / denominator)



    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_states, n_states))





def evaluate_ph_policy_in_bimodal(

    cfg: Config,

    model: BimodalModel,

    ph_data: PHData,

    ph_policy: np.ndarray,

    bimodal_data: BimodalData,

):

    started = time.perf_counter()

    n_states = len(bimodal_data.short_counts)

    value_next = np.zeros(n_states, dtype=float)

    ph_state_ids = mapped_ph_state_ids(cfg, model, bimodal_data, ph_data)

    lam_low = arrival_rate(cfg, 0.0)



    used_actions = np.unique(ph_policy.reshape(-1, cfg.n_phases), axis=0)

    service_matrices = {

        tuple(int(x) for x in action): build_bimodal_randomized_service_matrix(

            cfg, model, bimodal_data, tuple(int(x) for x in action)

        )

        for action in used_actions

    }



    for step in range(cfg.steps - 1, -1, -1):

        t = step * cfg.dt

        regime = 0 if np.isclose(arrival_rate(cfg, t), lam_low) else 1

        A = bimodal_data.arrival_matrices[regime]

        after_arrival_value = A.dot(value_next)

        post_service_value = (

            cfg.lost_demand_penalty * bimodal_data.expected_lost[regime]

            + after_arrival_value

        )

        phase_actions = ph_policy[step, ph_state_ids, :]

        price_rate = cfg.charger_power_kw * electricity_price(cfg, t)

        value_current = np.empty(n_states, dtype=float)



        for action_array in np.unique(phase_actions, axis=0):

            action = tuple(int(x) for x in action_array)

            mask = np.all(phase_actions == action_array, axis=1)

            expected_post = service_matrices[action].dot(post_service_value)

            immediate_charge = price_rate * sum(action) * cfg.dt

            value_current[mask] = immediate_charge + expected_post[mask]

        value_next = value_current



    return float(value_next[bimodal_data.empty_state_id]), time.perf_counter() - started





def build_exponential_data(cfg: Config) -> ExponentialData:
    """Build the mean-matched exponential approximation on the common grid."""
    N = cfg.n_batteries
    K = cfg.n_chargers
    states = np.arange(N + 1, dtype=np.int16)

    rate = 1.0 / cfg.mean_replenishment_hours
    completion_probability = 1.0 - math.exp(-rate * cfg.dt)
    expected_busy_one = (1.0 - math.exp(-rate * cfg.dt)) / rate

    service_matrices: List[sparse.csr_matrix] = []
    expected_service_charge_hours = np.zeros(K + 1, dtype=float)

    for u in range(K + 1):
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        expected_service_charge_hours[u] = u * expected_busy_one

        for i in range(N + 1):
            if u > i:
                continue
            for completed in range(u + 1):
                probability = float(
                    binom.pmf(completed, u, completion_probability)
                )
                if probability <= 0.0:
                    continue
                rows.append(i)
                cols.append(i - completed)
                vals.append(probability)

        service_matrices.append(
            sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(N + 1, N + 1),
            )
        )

    arrival_matrices: List[sparse.csr_matrix] = []
    expected_lost: List[np.ndarray] = []

    for lam in (arrival_rate(cfg, 0.0), arrival_rate(cfg, 9.0)):
        mean_arrivals = lam * cfg.dt
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        lost = np.zeros(N + 1, dtype=float)

        for i in range(N + 1):
            capacity = N - i
            lost[i] = expected_lost_arrivals(capacity, mean_arrivals)
            for admitted, probability in admitted_arrival_distribution(
                capacity, mean_arrivals
            ):
                rows.append(i)
                cols.append(i + admitted)
                vals.append(probability)

        arrival_matrices.append(
            sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(N + 1, N + 1),
            )
        )
        expected_lost.append(lost)

    return ExponentialData(
        states=states,
        service_matrices=tuple(service_matrices),
        expected_service_charge_hours=expected_service_charge_hours,
        arrival_matrices=(arrival_matrices[0], arrival_matrices[1]),
        expected_lost=(expected_lost[0], expected_lost[1]),
    )


def solve_exponential_common_grid(cfg: Config, data: ExponentialData):
    """Solve the mean-matched exponential DTMDP on the common grid."""
    started = time.perf_counter()
    N = cfg.n_batteries
    K = cfg.n_chargers

    value_next = np.zeros(N + 1, dtype=float)
    charging_next = np.zeros(N + 1, dtype=float)
    policy = np.zeros((cfg.steps, N + 1), dtype=np.int8)
    lam_low = arrival_rate(cfg, 0.0)

    for step in range(cfg.steps - 1, -1, -1):
        t = step * cfg.dt
        regime = 0 if np.isclose(arrival_rate(cfg, t), lam_low) else 1
        A = data.arrival_matrices[regime]

        after_arrival_value = A.dot(value_next)
        after_arrival_charging = A.dot(charging_next)
        post_service_value = (
            cfg.lost_demand_penalty * data.expected_lost[regime]
            + after_arrival_value
        )

        price_rate = cfg.charger_power_kw * electricity_price(cfg, t)
        best = np.full(N + 1, np.inf, dtype=float)
        best_action = np.full(N + 1, -1, dtype=np.int8)

        for u in range(K + 1):
            feasible = data.states >= u
            expected_post = data.service_matrices[u].dot(post_service_value)
            q = (
                price_rate * data.expected_service_charge_hours[u]
                + expected_post
            )
            improve = feasible & (q < best - cfg.numerical_tolerance)
            best[improve] = q[improve]
            best_action[improve] = u

        if np.any(best_action < 0):
            raise RuntimeError(
                "Exponential model has a state without a feasible action."
            )

        charging_current = np.empty(N + 1, dtype=float)
        for u in range(K + 1):
            chosen = best_action == u
            if not np.any(chosen):
                continue
            continuation = data.service_matrices[u].dot(
                after_arrival_charging
            )
            charging_current[chosen] = (
                price_rate * data.expected_service_charge_hours[u]
                + continuation[chosen]
            )
            policy[step, chosen] = u

        value_next = best
        charging_next = charging_current

    value = float(value_next[0])
    charging = float(charging_next[0])
    return (
        value,
        charging,
        value - charging,
        N + 1,
        policy,
        time.perf_counter() - started,
    )


def build_bimodal_uniform_selection_matrix(
    data: BimodalData,
    u: int,
) -> sparse.csr_matrix:
    """
    True-bimodal service transition when an aggregate policy activates u
    batteries but has no residual-time information. The u batteries are
    selected uniformly from all depleted batteries in the state.
    """
    n_states = len(data.totals)
    action_ids = np.flatnonzero(data.action_total == u)
    if len(action_ids) == 0:
        raise RuntimeError(f"No exact-action templates found for u={u}.")

    totals = data.totals.astype(np.int16, copy=False)
    feasible_states = totals >= u
    denominators = np.zeros(n_states, dtype=float)

    if u == 0:
        denominators[feasible_states] = 1.0
    else:
        for total in np.unique(totals[feasible_states]):
            denominators[totals == total] = math.comb(int(total), u)

    rows_parts: List[np.ndarray] = []
    cols_parts: List[np.ndarray] = []
    vals_parts: List[np.ndarray] = []

    for aid in action_ids:
        successor = data.service_successor[int(aid)]
        valid = successor >= 0
        if not np.any(valid):
            continue
        row_ids = np.flatnonzero(valid).astype(np.int32, copy=False)
        rows_parts.append(row_ids)
        cols_parts.append(successor[valid].astype(np.int32, copy=False))
        vals_parts.append(1.0 / denominators[valid])

    if not rows_parts:
        return sparse.csr_matrix((n_states, n_states))

    matrix = sparse.csr_matrix(
        (
            np.concatenate(vals_parts),
            (np.concatenate(rows_parts), np.concatenate(cols_parts)),
        ),
        shape=(n_states, n_states),
    )
    matrix.sum_duplicates()

    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.allclose(
        row_sums[feasible_states],
        1.0,
        atol=2e-11,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"Uniform-selection service matrix for u={u} "
            "does not have unit row sums on feasible states."
        )

    return matrix


def evaluate_exponential_policy_in_bimodal(
    cfg: Config,
    exponential_policy: np.ndarray,
    bimodal_data: BimodalData,
):
    """
    Evaluate the exponential policy under the true bimodal model.

    Because the exponential approximation has no branch/residual-time
    information, any activated subset is chosen uniformly among the currently
    depleted batteries.
    """
    started = time.perf_counter()
    n_states = len(bimodal_data.totals)
    totals = bimodal_data.totals.astype(np.int16, copy=False)
    value_next = np.zeros(n_states, dtype=float)
    lam_low = arrival_rate(cfg, 0.0)

    used_actions = tuple(int(u) for u in np.unique(exponential_policy))
    service_matrices = {
        u: build_bimodal_uniform_selection_matrix(bimodal_data, u)
        for u in used_actions
    }

    for step in range(cfg.steps - 1, -1, -1):
        t = step * cfg.dt
        regime = 0 if np.isclose(arrival_rate(cfg, t), lam_low) else 1
        A = bimodal_data.arrival_matrices[regime]

        after_arrival_value = A.dot(value_next)
        post_service_value = (
            cfg.lost_demand_penalty * bimodal_data.expected_lost[regime]
            + after_arrival_value
        )

        state_actions = exponential_policy[step, totals]
        price_rate = cfg.charger_power_kw * electricity_price(cfg, t)
        value_current = np.empty(n_states, dtype=float)

        for u in used_actions:
            mask = state_actions == u
            if not np.any(mask):
                continue
            expected_post = service_matrices[u].dot(post_service_value)
            value_current[mask] = (
                price_rate * u * cfg.dt + expected_post[mask]
            )

        value_next = value_current

    return (
        float(value_next[bimodal_data.empty_state_id]),
        time.perf_counter() - started,
    )


EXPONENTIAL_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "results/bimodal_exponential_common_grid"
)
EXPONENTIAL_CSV_NAME = "bimodal_exponential_common_comparison.csv"
EXPONENTIAL_NPZ_NAME = "bimodal_exponential_common_comparison.npz"


def write_exponential_csv(
    results: Sequence[ExponentialResult],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / EXPONENTIAL_CSV_NAME
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(ExponentialResult.__dataclass_fields__.keys()),
        )
        writer.writeheader()
        for result in sorted(results, key=lambda r: r.scv):
            writer.writerow(asdict(result))
    return path


def write_exponential_npz(
    results: Sequence[ExponentialResult],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda r: r.scv)
    path = output_dir / EXPONENTIAL_NPZ_NAME
    np.savez_compressed(
        path,
        scv=np.asarray([r.scv for r in ordered], dtype=float),
        bimodal_optimal_expected_cost=np.asarray(
            [r.bimodal_optimal_expected_cost for r in ordered],
            dtype=float,
        ),
        exponential_optimal_expected_cost=np.asarray(
            [r.exponential_optimal_expected_cost for r in ordered],
            dtype=float,
        ),
        exponential_policy_bimodal_expected_cost=np.asarray(
            [r.exponential_policy_bimodal_expected_cost for r in ordered],
            dtype=float,
        ),
        exponential_policy_loss_percent=np.asarray(
            [r.exponential_policy_loss_percent for r in ordered],
            dtype=float,
        ),
    )
    return path


CSV_NAME = "bimodal_h2_erlang_common_comparison.csv"

NPZ_NAME = "bimodal_h2_erlang_common_comparison.npz"





def write_csv(results: Sequence[Result], output_dir: Path) -> Path:

    path = output_dir / CSV_NAME

    with path.open("w", newline="", encoding="utf-8") as handle:

        writer = csv.DictWriter(handle, fieldnames=list(Result.__dataclass_fields__.keys()))

        writer.writeheader()

        for result in sorted(results, key=lambda r: r.scv):

            writer.writerow(asdict(result))

    return path





def write_npz(results: Sequence[Result], output_dir: Path) -> Path:

    ordered = sorted(results, key=lambda r: r.scv)

    path = output_dir / NPZ_NAME

    np.savez_compressed(

        path,

        scv=np.asarray([r.scv for r in ordered], dtype=float),

        surrogate_actual_scv=np.asarray([r.surrogate_actual_scv for r in ordered], dtype=float),

        bimodal_optimal_expected_cost=np.asarray(

            [r.bimodal_optimal_expected_cost for r in ordered], dtype=float

        ),

        surrogate_optimal_expected_cost=np.asarray(

            [r.surrogate_optimal_expected_cost for r in ordered], dtype=float

        ),

        surrogate_policy_bimodal_expected_cost=np.asarray(

            [r.surrogate_policy_bimodal_expected_cost for r in ordered], dtype=float

        ),

        surrogate_policy_loss_percent=np.asarray(

            [r.surrogate_policy_loss_percent for r in ordered], dtype=float

        ),

    )

    return path





def parse_scv_argument(raw: str) -> Tuple[float, ...]:

    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())

    if not values:

        raise argparse.ArgumentTypeError("SCV list cannot be empty.")

    return values





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--N", type=int, default=4)

    parser.add_argument("--K", type=int, default=3)

    parser.add_argument("--scv", type=parse_scv_argument, default=TARGET_SCVS)

    parser.add_argument("--erlang-order", type=int, default=ERLANG_BRANCH_ORDER)

    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    parser.add_argument(

        "--plot-only",

        action="store_true",

        help="Regenerate figures from the existing CSV without recomputing the DTMDPs.",

    )

    return parser.parse_args()





def nonnegative_loss(policy_value: float, optimum: float) -> float:

    loss = policy_value - optimum

    scale = max(1.0, abs(policy_value), abs(optimum))

    if loss < -1e-7 * scale:

        raise RuntimeError(

            "Transferred component-Erlang policy is cheaper than the exact "

            "bimodal optimum. Check the implementation."

        )

    if abs(loss) <= 1e-10 * scale:

        return 0.0

    return loss







def configure_plots() -> None:

    fonts = {font.name for font in font_manager.fontManager.ttflist}

    family = "Times New Roman" if "Times New Roman" in fonts else "Nimbus Roman"

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





def format_scv_axis(ax: plt.Axes, x: np.ndarray) -> None:

    ax.set_xticks(x)

    ax.set_xticklabels([f"{v:g}" for v in x])

    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("SCV of bimodal replenishment time")

    ax.margins(y=0.08)

    ax.grid(alpha=0.2)





def make_plots(

    x: np.ndarray,

    bimodal_cost: np.ndarray,

    surrogate_cost: np.ndarray,

    transferred_policy_cost: np.ndarray,

    exact_time: np.ndarray,

    surrogate_time: np.ndarray,

    order: int,

    output_dir: Path,

):

    configure_plots()

    idx = np.argsort(x)

    x = x[idx]

    bimodal_cost = bimodal_cost[idx]

    surrogate_cost = surrogate_cost[idx]

    transferred_policy_cost = transferred_policy_cost[idx]

    exact_time = exact_time[idx]

    surrogate_time = surrogate_time[idx]

    output_dir.mkdir(parents=True, exist_ok=True)

    surrogate_label = f"Component Erlang-{order}"



    fig1, ax1 = plt.subplots(figsize=(10, 6.1))

    ax1.plot(x, bimodal_cost, color="#222222", marker="o", markersize=7.5,

             linewidth=2.0, label="Bimodal")

    ax1.plot(x, surrogate_cost, color="#2474b5", marker="s", markersize=7.5,

             linewidth=2.0, label=surrogate_label)

    format_scv_axis(ax1, x)

    ax1.set_ylabel("Optimal expected cost (USD)")

    ax1.legend(frameon=True)

    fig1.tight_layout()

    cost_path = output_dir / "bimodal_h2_erlang_optimal_cost.png"

    fig1.savefig(cost_path, bbox_inches="tight", facecolor="white")

    plt.close(fig1)



    fig2, ax2 = plt.subplots(figsize=(10, 6.1))

    ax2.plot(x, bimodal_cost, color="#222222", marker="o", markersize=7.5,

             linewidth=2.0, label="Bimodal-optimal policy")

    ax2.plot(x, transferred_policy_cost, color="#6a3d9a", marker="s", markersize=7.5,

             linewidth=2.0, label=f"Erlang-{order} policy")

    format_scv_axis(ax2, x)

    ax2.set_ylabel("Expected cost under bimodal model (USD)")

    ax2.legend(frameon=True)

    fig2.tight_layout()

    policy_path = output_dir / "bimodal_h2_erlang_policy_cost.png"

    fig2.savefig(policy_path, bbox_inches="tight", facecolor="white")

    plt.close(fig2)



    fig3, ax3 = plt.subplots(figsize=(10, 6.1))

    ax3.plot(x, exact_time, color="#222222", marker="o", markersize=7.5,

             linewidth=2.0, label="Bimodal")

    ax3.plot(x, surrogate_time, color="#2474b5", marker="s", markersize=7.5,

             linewidth=2.0, label=surrogate_label)

    format_scv_axis(ax3, x)

    ax3.set_yscale("log")

    ax3.set_ylabel("Computational time (seconds)")

    ax3.legend(frameon=True)

    fig3.tight_layout()

    time_path = output_dir / "bimodal_h2_erlang_computational_time.png"

    fig3.savefig(time_path, bbox_inches="tight", facecolor="white")

    plt.close(fig3)

    return cost_path, policy_path, time_path





def plot_results(results: Sequence[Result], output_dir: Path):

    ordered = sorted(results, key=lambda r: r.scv)

    orders = {r.erlang_branch_order for r in ordered}

    if len(orders) != 1:

        raise RuntimeError("A single plot cannot mix different Erlang orders.")

    return make_plots(

        x=np.asarray([r.scv for r in ordered], dtype=float),

        bimodal_cost=np.asarray([r.bimodal_optimal_expected_cost for r in ordered], dtype=float),

        surrogate_cost=np.asarray([r.surrogate_optimal_expected_cost for r in ordered], dtype=float),

        transferred_policy_cost=np.asarray(

            [r.surrogate_policy_bimodal_expected_cost for r in ordered], dtype=float

        ),

        exact_time=np.asarray([r.exact_elapsed_seconds for r in ordered], dtype=float),

        surrogate_time=np.asarray([r.surrogate_elapsed_seconds for r in ordered], dtype=float),

        order=next(iter(orders)),

        output_dir=output_dir,

    )





def plot_results_from_csv(csv_path: Path, output_dir: Path):

    rows: List[Dict[str, str]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:

        rows.extend(csv.DictReader(handle))

    if not rows:

        raise RuntimeError(f"No data found in {csv_path}")

    orders = {int(row["erlang_branch_order"]) for row in rows}

    if len(orders) != 1:

        raise RuntimeError("The CSV mixes different Erlang branch orders.")

    return make_plots(

        x=np.asarray([float(row["scv"]) for row in rows]),

        bimodal_cost=np.asarray([float(row["bimodal_optimal_expected_cost"]) for row in rows]),

        surrogate_cost=np.asarray([float(row["surrogate_optimal_expected_cost"]) for row in rows]),

        transferred_policy_cost=np.asarray(

            [float(row["surrogate_policy_bimodal_expected_cost"]) for row in rows]

        ),

        exact_time=np.asarray([float(row["exact_elapsed_seconds"]) for row in rows]),

        surrogate_time=np.asarray([float(row["surrogate_elapsed_seconds"]) for row in rows]),

        order=next(iter(orders)),

        output_dir=output_dir,

    )





def main() -> None:

    args = parse_args()

    if args.N < 1:

        raise ValueError("N must be positive.")

    if not 1 <= args.K <= args.N:

        raise ValueError("Need 1 <= K <= N.")

    if args.erlang_order < 1:

        raise ValueError("Erlang order must be positive.")



    cfg = Config(

        n_batteries=args.N,

        n_chargers=args.K,

        erlang_order=args.erlang_order,

    )

    output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)



    if args.plot_only:

        figures = plot_results_from_csv(output_dir / CSV_NAME, output_dir)

        print("\nFigures regenerated from existing CSV:")

        for figure in figures:

            print(figure)

        return



    exponential_output_dir = EXPONENTIAL_OUTPUT_DIR.resolve()
    exponential_output_dir.mkdir(parents=True, exist_ok=True)

    exponential_build_started = time.perf_counter()
    exponential_data = build_exponential_data(cfg)
    exponential_build_elapsed = time.perf_counter() - exponential_build_started
    (
        exponential_value,
        exponential_charging,
        exponential_lost,
        exponential_states,
        exponential_policy,
        exponential_solve_elapsed,
    ) = solve_exponential_common_grid(cfg, exponential_data)
    exponential_elapsed = (
        exponential_build_elapsed + exponential_solve_elapsed
    )

    print(

        "COMMON-GRID component-Erlang experiment\n"

        f"N={cfg.n_batteries}, K={cfg.n_chargers}, "

        f"branch order={cfg.erlang_order}, phases={cfg.n_phases}, "

        f"dt={cfg.dt:g} h, steps={cfg.steps}, "

        f"represented horizon={cfg.represented_horizon_hours:.2f} h"

    )

    print(

        "\nSurrogate construction:"

        "\n  same branch probabilities as bimodal"

        "\n  same conditional branch means as bimodal support points"

        f"\n  Erlang-{cfg.erlang_order} within each branch"

        "\n  same overall mean"

        "\n  SCV is not matched for any finite Erlang order"

    )



    print(
        "\nMean-matched exponential approximation:"
        f"\n  mean={cfg.mean_replenishment_hours:g} h"
        f"\n  rate={1.0 / cfg.mean_replenishment_hours:.12g} per hour"
        f"\n  optimal expected cost={exponential_value:.10f}"
        f"\n  aggregate states={exponential_states}"
    )

    results: List[Result] = []
    exponential_results: List[ExponentialResult] = []

    for target_scv in args.scv:

        print("\n" + "=" * 72)

        print(f"TRUE BIMODAL SCV={target_scv:g}")

        bimodal = make_bimodal(cfg, target_scv)

        surrogate = make_h2_erlang(cfg, bimodal)

        expected_surrogate_scv = (

            bimodal.actual_scv + (bimodal.actual_scv + 1.0) / cfg.erlang_order

        )

        print("\nBimodal:\n  " + bimodal.specification)

        print("\nComponent-Erlang surrogate:\n  " + surrogate.specification)

        print(f"\n  True bimodal SCV       = {bimodal.actual_scv:g}")

        print(f"  Surrogate SCV          = {surrogate.actual_scv:g}")

        print(f"  Identity check         = {expected_surrogate_scv:g}")

        print(f"  Bimodal/surrogate mean = {bimodal.mean_hours:g} h")



        build_started = time.perf_counter()

        bimodal_data = build_bimodal_data(cfg, bimodal)

        bimodal_build_elapsed = time.perf_counter() - build_started

        (

            bimodal_value,

            bimodal_charging,

            bimodal_lost,

            exact_states,

            bimodal_solve_elapsed,

        ) = solve_bimodal_exact(cfg, bimodal_data)

        exact_elapsed = bimodal_build_elapsed + bimodal_solve_elapsed



        ph_build_started = time.perf_counter()

        ph_data = build_ph_data(cfg, surrogate)

        ph_build_elapsed = time.perf_counter() - ph_build_started

        (

            surrogate_value,

            surrogate_charging,

            surrogate_lost,

            surrogate_states,

            surrogate_policy,

            surrogate_solve_elapsed,

        ) = solve_ph_common_grid(cfg, ph_data)

        surrogate_elapsed = ph_build_elapsed + surrogate_solve_elapsed



        transferred_value, policy_eval_elapsed = evaluate_ph_policy_in_bimodal(

            cfg, bimodal, ph_data, surrogate_policy, bimodal_data

        )

        loss_abs = nonnegative_loss(transferred_value, bimodal_value)

        loss_pct = 100.0 * loss_abs / bimodal_value

        optimal_gap_pct = 100.0 * (surrogate_value - bimodal_value) / bimodal_value

        (
            exponential_transferred_value,
            exponential_policy_eval_elapsed,
        ) = evaluate_exponential_policy_in_bimodal(
            cfg,
            exponential_policy,
            bimodal_data,
        )
        exponential_loss_abs = nonnegative_loss(
            exponential_transferred_value,
            bimodal_value,
        )
        exponential_loss_pct = (
            100.0 * exponential_loss_abs / bimodal_value
        )
        exponential_optimal_gap_pct = (
            100.0 * (exponential_value - bimodal_value) / bimodal_value
        )



        result = Result(

            scv=bimodal.actual_scv,

            n_batteries=cfg.n_batteries,

            n_chargers=cfg.n_chargers,

            common_dt_hours=cfg.dt,

            time_steps=cfg.steps,

            erlang_branch_order=cfg.erlang_order,

            bimodal_low_quanta=bimodal.low_quanta,

            bimodal_high_quanta=bimodal.high_quanta,

            bimodal_low_hours=bimodal.low_quanta * cfg.dt,

            bimodal_high_hours=bimodal.high_quanta * cfg.dt,

            bimodal_p_low=bimodal.p_low,

            bimodal_p_high=bimodal.p_high,

            surrogate_p_fast=surrogate.p_fast,

            surrogate_p_slow=surrogate.p_slow,

            surrogate_rate_fast=surrogate.rate_fast,

            surrogate_rate_slow=surrogate.rate_slow,

            surrogate_actual_scv=surrogate.actual_scv,

            exact_states=exact_states,

            surrogate_states=surrogate_states,

            bimodal_optimal_charging_cost=bimodal_charging,

            bimodal_optimal_lost_demand_cost=bimodal_lost,

            bimodal_optimal_expected_cost=bimodal_value,

            surrogate_optimal_charging_cost=surrogate_charging,

            surrogate_optimal_lost_demand_cost=surrogate_lost,

            surrogate_optimal_expected_cost=surrogate_value,

            surrogate_policy_bimodal_expected_cost=transferred_value,

            surrogate_policy_loss_absolute=loss_abs,

            surrogate_policy_loss_percent=loss_pct,

            surrogate_vs_bimodal_optimal_cost_gap_percent=optimal_gap_pct,

            exact_elapsed_seconds=exact_elapsed,

            surrogate_elapsed_seconds=surrogate_elapsed,

            surrogate_policy_eval_elapsed_seconds=policy_eval_elapsed,

        )

        exponential_result = ExponentialResult(
            scv=bimodal.actual_scv,
            n_batteries=cfg.n_batteries,
            n_chargers=cfg.n_chargers,
            common_dt_hours=cfg.dt,
            time_steps=cfg.steps,
            bimodal_low_quanta=bimodal.low_quanta,
            bimodal_high_quanta=bimodal.high_quanta,
            bimodal_low_hours=bimodal.low_quanta * cfg.dt,
            bimodal_high_hours=bimodal.high_quanta * cfg.dt,
            bimodal_p_low=bimodal.p_low,
            bimodal_p_high=bimodal.p_high,
            exact_states=exact_states,
            exponential_states=exponential_states,
            exponential_mean_hours=cfg.mean_replenishment_hours,
            exponential_rate=1.0 / cfg.mean_replenishment_hours,
            bimodal_optimal_charging_cost=bimodal_charging,
            bimodal_optimal_lost_demand_cost=bimodal_lost,
            bimodal_optimal_expected_cost=bimodal_value,
            exponential_optimal_charging_cost=exponential_charging,
            exponential_optimal_lost_demand_cost=exponential_lost,
            exponential_optimal_expected_cost=exponential_value,
            exponential_policy_bimodal_expected_cost=(
                exponential_transferred_value
            ),
            exponential_policy_loss_absolute=exponential_loss_abs,
            exponential_policy_loss_percent=exponential_loss_pct,
            exponential_vs_bimodal_optimal_cost_gap_percent=(
                exponential_optimal_gap_pct
            ),
            exact_elapsed_seconds=exact_elapsed,
            exponential_elapsed_seconds=exponential_elapsed,
            exponential_policy_eval_elapsed_seconds=(
                exponential_policy_eval_elapsed
            ),
        )

        results.append(result)
        exponential_results.append(exponential_result)

        write_csv(results, output_dir)
        write_npz(results, output_dir)
        write_exponential_csv(
            exponential_results,
            exponential_output_dir,
        )
        write_exponential_npz(
            exponential_results,
            exponential_output_dir,
        )

        print("\nCOMPONENT-ERLANG RESULT")
        print(json.dumps(asdict(result), indent=2))
        print("\nEXPONENTIAL RESULT")
        print(json.dumps(asdict(exponential_result), indent=2))

        print(f"\nIntermediate results saved after true SCV={target_scv:g}.")

        del bimodal_data, surrogate_policy, ph_data



    table = write_csv(results, output_dir)
    archive = write_npz(results, output_dir)
    exponential_table = write_exponential_csv(
        exponential_results,
        exponential_output_dir,
    )
    exponential_archive = write_exponential_npz(
        exponential_results,
        exponential_output_dir,
    )
    figures = plot_results(results, output_dir)

    print("\nOutputs:")
    for figure in figures:
        print(figure)
    print(table)
    print(archive)
    print(exponential_table)
    print(exponential_archive)





if __name__ == "__main__":

    main()
