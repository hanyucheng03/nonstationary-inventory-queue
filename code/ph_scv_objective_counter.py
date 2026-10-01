from pathlib import Path
import csv
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import NullFormatter
try:
    import numba
    from numba import prange
    NUMBA_AVAILABLE = True
    numba.set_num_threads(min(4, numba.get_num_threads()))
except ImportError:
    print("import numba error.")
    NUMBA_AVAILABLE = False
    prange = range


N = 23
K = 20
ARRIVAL_SCALE = 2.0
MEAN_SERVICE_HOURS = 0.78
LOST_DEMAND_PENALTY = 6.0
OFFPEAK_COST_PER_CHARGER_HOUR = 2.788
PEAK_COST_PER_CHARGER_HOUR = 9.143
HORIZON_DAYS = 5
REQUESTED_DT = 0.01
ERLANG_SHAPES = [5, 3, 2]
HYPEREXP_SCVS = [2, 3, 5, 10]
SHOW_FIGURE = True
OUTPUT_STEM = "ph_lost_demand_dominates"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "results" / "scv_results"


def compositions_at_most(n, j):
    """All nonnegative j-vectors with sum <= n, including the zero state."""
    if j == 1:
        return [(i,) for i in range(n + 1)]
    return [(i,) + rest for i in range(n + 1)
            for rest in compositions_at_most(n - i, j - 1)]


def distribution(kind, parameter, mean):
    if kind == "Erlang":
        j = int(parameter)
        alpha = np.zeros(j); alpha[0] = 1.0
        rates = np.full(j, j / mean)
        remaining = mean * np.arange(j, 0, -1) / j
        return alpha, rates, remaining, 1.0 / j, f"Erlang-{j}"
    if kind == "Exponential":
        return np.ones(1), np.array([1 / mean]), np.array([mean]), 1., "Exponential"
    c = float(parameter)
    if c <= 1:
        raise ValueError("Hyperexponential SCV must exceed 1")
    p = .5 * (1 + np.sqrt((c - 1) / (c + 1)))
    alpha = np.array([p, 1 - p])
    rates = 2 * alpha / mean
    return alpha, rates, 1 / rates, c, f"H2-SCV-{c:g}"


def _backward_kernel(states, full, arrival_dest, service_dest, alpha, rates,
                     remaining, steps, steps_per_hour, k, scale, peak_price,
                     low_price, penalty):
    size, j = states.shape
    values = np.zeros((3, size))
    current = np.empty_like(values)
    gains = np.empty((size, j))
    for i in range(size):
        for phase in range(j):
            values[2,i] += states[i,phase] * remaining[phase]
    dt = 1.0 / steps_per_hour
    for step in range(steps-1, -1, -1):
        hour = (step // steps_per_hour) % 24
        lam = (5.68 if hour < 9 else 14.51) * scale
        price = peak_price if 15 <= hour < 20 else low_price
        for i in prange(size):
            cc, ll, rr = values[0,i], values[1,i], values[2,i]
            dc, dl, dr = 0.0, penalty * lam if full[i] else 0.0, 0.0
            for phase in range(j):
                dest = service_dest[phase,i]
                gains[i,phase] = (rates[phase] * (cc + ll - values[0,dest] - values[1,dest]) - price
                                  if states[i,phase] > 0 else -np.inf)
                if not full[i]:
                    dest = arrival_dest[phase,i]
                    w = lam * alpha[phase]
                    dc += w*(values[0,dest]-cc)
                    dl += w*(values[1,dest]-ll)
                    dr += w*(values[2,dest]-rr)
            free = k
            for rank in range(j):
                best, phase = 0.0, -1
                for candidate in range(j):
                    if gains[i,candidate] > best:
                        best, phase = gains[i,candidate], candidate
                if phase < 0 or free == 0:
                    break
                amount = min(free, states[i,phase])
                dest = service_dest[phase,i]
                w = rates[phase] * amount
                dc += price*amount + w*(values[0,dest]-cc)
                dl += w*(values[1,dest]-ll)
                dr += w*(values[2,dest]-rr)
                free -= amount
                gains[i,phase] = -np.inf
            current[0,i] = cc + dt*dc
            current[1,i] = ll + dt*dl
            current[2,i] = rr + dt*dr
        values, current = current, values
    return values


if NUMBA_AVAILABLE:
    _kernel_serial = numba.njit(cache=True)(_backward_kernel)
    _kernel_parallel = numba.njit(cache=True, parallel=True)(_backward_kernel)


def solve_case(kind, parameter, n=N, k=K, arrival_scale=ARRIVAL_SCALE,
               days=HORIZON_DAYS, requested_dt=REQUESTED_DT, use_numba=True):
    if not 1 <= k <= n:
        raise ValueError("Require 1 <= K <= N")
    if requested_dt <= 0 or days <= 0 or int(days) != days:
        raise ValueError("Use positive dt and a positive integer number of days")
    alpha, rates, remaining, scv, label = distribution(kind, parameter, MEAN_SERVICE_HOURS)
    j = len(alpha)
    states = np.array(compositions_at_most(n, j), dtype=np.int32)
    size = len(states)
    lookup = {tuple(x): i for i, x in enumerate(states)}
    zero = lookup[(0,) * j]
    full = states.sum(axis=1) == n
    arrival_dest = np.empty((j, size), dtype=np.int32)
    service_dest = np.empty((j, size), dtype=np.int32)
    for idx, x in enumerate(states):
        for phase in range(j):
            y = x.copy()
            if not full[idx]:
                y[phase] += 1
            arrival_dest[phase, idx] = lookup[tuple(y)]
            y = x.copy()
            if x[phase] > 0:
                y[phase] -= 1
                if kind == "Erlang" and phase + 1 < j:
                    y[phase + 1] += 1
            service_dest[phase, idx] = lookup[tuple(y)]

    max_exit = 14.51 * arrival_scale + k * float(rates.max())
    steps_per_hour = math.ceil(max(1 / requested_dt, max_exit / .95))
    dt = 1 / steps_per_hour
    steps = int(24 * days * steps_per_hour)
    if NUMBA_AVAILABLE and use_numba:
        kernel = _kernel_parallel if size > 10000 else _kernel_serial
        values = kernel(states, full, arrival_dest, service_dest, alpha, rates,
                        remaining, steps, steps_per_hour, k, arrival_scale,
                        PEAK_COST_PER_CHARGER_HOUR, OFFPEAK_COST_PER_CHARGER_HOUR,
                        LOST_DEMAND_PENALTY)
    else:
        values = np.zeros((3, size))
        values[2] = states @ remaining
        idxs = np.arange(size)
        for step in range(steps - 1, -1, -1):
            hour = (step // steps_per_hour) % 24
            lam = (5.68 if hour < 9 else 14.51) * arrival_scale
            price = PEAK_COST_PER_CHARGER_HOUR if 15 <= hour < 20 else OFFPEAK_COST_PER_CHARGER_HOUR
            v = values[0] + values[1]
            gain = rates[:, None] * (v[None, :] - v[service_dest]) - price
            gain = np.where(states.T > 0, gain, -np.inf)
            order = np.argsort(-gain, axis=0, kind="stable")
            action = np.zeros((j, size))
            free = np.full(size, k)
            for rank in range(j):
                phase = order[rank]
                amount = np.minimum(free, states[idxs, phase])
                amount = np.where(gain[phase, idxs] > 0, amount, 0)
                action[phase, idxs] = amount
                free -= amount
            arrival_rate = lam * (~full)
            service_rate = rates[:, None] * action
            jump = np.sum(alpha[None, :, None] * values[:, arrival_dest], axis=1) - values
            derivative = arrival_rate * jump
            derivative += np.sum(service_rate[None, :, :] *
                                 (values[:, service_dest] - values[:, None, :]), axis=1)
            derivative[0] += price * action.sum(axis=0)
            derivative[1] += LOST_DEMAND_PENALTY * lam * full
            values += dt * derivative
    charge, lost, residual = map(float, values[:, zero])
    return dict(distribution=label,scv=scv,N=n,K=k,arrival_scale=arrival_scale,
                mean_service_hours=MEAN_SERVICE_HOURS,horizon_days=days,
                penalty=LOST_DEMAND_PENALTY,offpeak_rate=OFFPEAK_COST_PER_CHARGER_HOUR,
                peak_rate=PEAK_COST_PER_CHARGER_HOUR,dt=dt,states=size,
                charging_cost=charge,lost_demand_cost=lost,total_cost=charge+lost,
                lost_to_charging_ratio=lost/charge,
                expected_lost_requests=lost/LOST_DEMAND_PENALTY,
                terminal_remaining_work_hours=residual)



def configure_plots() -> None:
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
            "legend.fontsize": 24,
            "figure.dpi": 140,
            "savefig.dpi": 420,
        }
    )


def scv_tick_labels(results):
    labels = []
    for result in results:
        name = result["distribution"]
        if name.startswith("Erlang-"):
            order = int(name.split("-", 1)[1])
            labels.append(rf"$1/{order}$")
        else:
            labels.append(f"{result['scv']:g}")
    return labels


def format_scv_axis(ax, ordered, show_xlabel):
    x = np.asarray([result["scv"] for result in ordered])
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(scv_tick_labels(ordered))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if show_xlabel:
        ax.set_xlabel("SCV of replenishment time")
    ax.margins(y=0.08)
    ax.grid(alpha=0.2)

def main():
    started = time.perf_counter()
    cases = [("Erlang", k) for k in ERLANG_SHAPES]
    cases += [("Exponential", 1)] + [("Hyperexponential", c) for c in HYPEREXP_SCVS]
    results = []
    for kind, parameter in cases:
        print(f"Solving {kind} {parameter} ...",flush=True)
        row = solve_case(kind, parameter)
        results.append(row)
        print(f"{row['distribution']:14s} SCV={row['scv']:6.3f} "
              f"charge={row['charging_cost']:9.3f} lost={row['lost_demand_cost']:9.3f} "
              f"total={row['total_cost']:9.3f} ratio={row['lost_to_charging_ratio']:.2f}",flush=True)
    results.sort(key=lambda d: d['scv'])
    OUTPUT_DIRECTORY.mkdir(parents=True,exist_ok=True)
    with (OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    configure_plots()
    fig, ax = plt.subplots(figsize=(10, 6.1))
    x = np.asarray([d["scv"] for d in results])
    component_series = {
        "Total cost": (
            np.asarray([d["total_cost"] for d in results]),
            "#222222",
            "o",
        ),
        "Charging cost": (
            np.asarray([d["charging_cost"] for d in results]),
            "#2474b5",
            "s",
        ),
        "Lost-demand cost": (
            np.asarray([d["lost_demand_cost"] for d in results]),
            "#c53b3b",
            "^",
        ),
    }
    for label, (absolute, color, marker) in component_series.items():
        ax.plot(
            x,
            absolute,
            color=color,
            marker=marker,
            markersize=7.5,
            linewidth=2.0,
            label=label,
        )

    format_scv_axis(ax, results, show_xlabel=True)
    ax.set_ylabel("Optimal expected cost (USD)")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.7), frameon=True)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.png",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved PNG, PDF and CSV to {OUTPUT_DIRECTORY}; elapsed {time.perf_counter()-started:.1f}s")
    if SHOW_FIGURE: plt.show()
    else: plt.close(fig)


if __name__ == "__main__":
    main()
