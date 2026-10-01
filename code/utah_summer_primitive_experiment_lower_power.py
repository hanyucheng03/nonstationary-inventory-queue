import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


DAYS = 5
T = 24.0 * DAYS
DT = 0.01
N = 23
K = 20
MU = 1.0 / 4.69
P_L = 3.01
CHARGER_POWER_KW = 7.0

# Rocky Mountain Power Schedule 2E, Rate Option 1 (Utah).
HIGH_ELECTRICITY_PRICE = 0.222755
LOW_ELECTRICITY_PRICE = 0.067881

OUTPUT_DIR = Path("numerical_utah_summer_figures")


def configure_plot_style():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    times_font = (
        "Times New Roman"
        if "Times New Roman" in available_fonts
        else "Nimbus Roman"
    )
    plt.rcParams.update({
        "font.family": times_font,
        "font.serif": [times_font],
        "mathtext.fontset": "custom",
        "mathtext.rm": times_font,
        "mathtext.it": f"{times_font}:italic",
        "mathtext.bf": f"{times_font}:bold",
        "mathtext.cal": f"{times_font}:italic",
        "mathtext.sf": times_font,
        "mathtext.tt": times_font,
        "font.size": 28,
        "axes.labelsize": 30,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 26,
        "figure.dpi": 120,
        "savefig.dpi": 420,
        "savefig.pad_inches": 0.08,
    })


def demand_rate(t):
    hour = np.mod(t, 24.0)
    return np.where(hour < 9.0, 5.68, 14.51)


def electricity_price(t):
    hour = np.mod(t, 24.0)
    return np.where(
        (15.0 <= hour) & (hour < 20.0),
        HIGH_ELECTRICITY_PRICE,
        LOW_ELECTRICITY_PRICE,
    )


def operating_cost_per_server(t):
    return CHARGER_POWER_KW * electricity_price(t)


def full_feasible_action_vector():
    return np.minimum(np.arange(N + 1), K).astype(np.int16)


def policy_to_threshold(policy):
    active = policy[:, 1:] > 0
    threshold = np.full(policy.shape[0], N + 1, dtype=np.int16)
    rows = np.any(active, axis=1)
    threshold[rows] = np.argmax(active[rows], axis=1) + 1
    return threshold


def solve_hjb():
    n_steps = int(round(T / DT))
    dt = T / n_steps
    time_grid = np.linspace(0.0, T, n_steps + 1)
    decision_times = time_grid[:-1]
    lambda_values = demand_rate(decision_times).astype(float)
    operating_cost_values = operating_cost_per_server(decision_times).astype(float)

    V = np.zeros((n_steps + 1, N + 1), dtype=float)
    policy = np.zeros((n_steps, N + 1), dtype=np.int16)
    full_action = full_feasible_action_vector()

    for n in range(n_steps - 1, -1, -1):
        lam = lambda_values[n]
        e = operating_cost_values[n]
        value_next = V[n + 1]
        marginal = np.zeros(N + 1)
        marginal[1:] = value_next[1:] - value_next[:-1]
        action = np.where(MU * marginal > e, full_action, 0).astype(np.int16)
        policy[n] = action

        lower = np.concatenate(([value_next[0]], value_next[:-1]))
        higher = np.concatenate((value_next[1:], [value_next[-1]]))
        running = e * action.astype(float)
        running[N] += P_L * lam
        V[n] = value_next + dt * (
            running
            + lam * (higher - value_next)
            + MU * action * (lower - value_next)
        )

    return {
        "time_grid": time_grid,
        "decision_times": decision_times,
        "V": V,
        "Delta": V[:, 1:] - V[:, :-1],
        "policy": policy,
        "threshold": policy_to_threshold(policy),
    }


def poisson_tail_at_least(n, mean):
    term = math.exp(-mean)
    cumulative = term
    for k in range(1, n):
        term *= mean / k
        cumulative += term
    return min(1.0, max(0.0, 1.0 - cumulative))


def build_segments():
    boundaries = {0.0, T}
    for day in range(DAYS):
        boundaries.add(24.0 * day + 15.0)
        boundaries.add(24.0 * day + 20.0)
    boundaries = sorted(x for x in boundaries if 0.0 <= x <= T)
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        hour = ((start + end) / 2.0) % 24.0
        segments.append({
            "start": start,
            "end": end,
            "type": "peak" if 15.0 <= hour < 20.0 else "off-peak",
        })
    return segments


def construct_primitive_certificates(decision_times):
    """Return only actions certified by primitive sufficient conditions."""
    segments = build_segments()
    certified = np.zeros(len(decision_times), dtype=bool)
    certified_threshold = np.full(len(decision_times), np.nan)
    diagnostics = []

    effective_high_cost = CHARGER_POWER_KW * HIGH_ELECTRICITY_PRICE / MU
    effective_low_cost = CHARGER_POWER_KW * LOW_ELECTRICITY_PRICE / MU
    peak_mean = 14.51 * 5.0
    offpeak_lower_bound = P_L * poisson_tail_at_least(N, peak_mean)

    for index, segment in enumerate(segments):
        start, end = segment["start"], segment["end"]
        mask = (decision_times >= start) & (decision_times < end)

        if segment["type"] == "peak":
            upper_bound = P_L
            passed = effective_high_cost >= upper_bound
            if passed:
                certified[mask] = True
                certified_threshold[mask] = N + 1
            bound = upper_bound
            conclusion = "shutdown" if passed else "unresolved"
        else:
            has_following_peak = (
                index + 1 < len(segments)
                and segments[index + 1]["type"] == "peak"
            )
            passed = has_following_peak and effective_low_cost < offpeak_lower_bound
            if passed:
                certified[mask] = True
                certified_threshold[mask] = 1
            bound = offpeak_lower_bound if has_following_peak else math.nan
            conclusion = "full activation" if passed else "unresolved"

        diagnostics.append({
            "start": start,
            "end": end,
            "type": segment["type"],
            "effective_cost": effective_high_cost if segment["type"] == "peak" else effective_low_cost,
            "primitive_bound": bound,
            "conclusion": conclusion,
        })

    return {
        "segments": segments,
        "certified": certified,
        "threshold": certified_threshold,
        "diagnostics": diagnostics,
        "effective_high_cost": effective_high_cost,
        "effective_low_cost": effective_low_cost,
        "peak_mean": peak_mean,
        "offpeak_lower_bound": offpeak_lower_bound,
    }


def shade_peaks(ax, segments):
    for segment in segments:
        if segment["type"] == "peak":
            ax.axvspan(
                segment["start"] / 24.0,
                segment["end"] / 24.0,
                color="#DCECF7",
                alpha=0.85,
                linewidth=0,
                zorder=0,
            )


def shade_peaks_across_axes(fig, axes, segments, total_days):
    """Draw peak-period bands continuously across stacked subplots."""
    for ax in axes:
        ax.set_facecolor("none")

    top_pos = axes[0].get_position()
    bottom_pos = axes[-1].get_position()
    left = top_pos.x0
    right = top_pos.x1
    bottom = bottom_pos.y0
    top = top_pos.y1
    width = right - left
    height = top - bottom

    for segment in segments:
        if segment["type"] == "peak":
            x0 = left + width * (segment["start"] / 24.0) / total_days
            x1 = left + width * (segment["end"] / 24.0) / total_days
            rect = plt.Rectangle(
                (x0, bottom),
                x1 - x0,
                height,
                transform=fig.transFigure,
                facecolor="#DCECF7",
                edgecolor="none",
                alpha=0.85,
                zorder=-10,
            )
            fig.add_artist(rect)


def plot_marginal_costs(results, certificates):
    configure_plot_style()
    times = results["time_grid"] / 24.0
    fig, ax = plt.subplots(figsize=(10, 6.1), constrained_layout=True)
    shade_peaks(ax, certificates["segments"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for state, color in zip([1, 5, 10, 15, 20], colors):
        ax.plot(times, results["Delta"][:, state - 1], lw=2.0, color=color, label=rf"$\Delta_{{{state}}}(t)$")
    effective_cost = CHARGER_POWER_KW * electricity_price(results["time_grid"]) / MU
    ax.step(times, effective_cost, where="post", color="black", lw=2.6, ls="--", label=r"$e(t)/\mu$")
    ax.set_xlim(0, DAYS)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cost per replenished battery ($)")
    ax.set_xticks(np.arange(DAYS + 1))
    ax.grid(alpha=0.20)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0.25,
        columnspacing=1.25,
        handlelength=2.0,
        handletextpad=0.55,
        frameon=True,
        framealpha=0.92,
        edgecolor="#B0B0B0",
        fancybox=False,
    )
    path = OUTPUT_DIR / "experiment1_marginal_costs_5days_summer_lower_power.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_optimal_threshold(results, certificates):
    configure_plot_style()
    times = np.append(results["decision_times"], T) / 24.0
    threshold = np.append(results["threshold"], results["threshold"][-1])
    fig, ax = plt.subplots(figsize=(10, 6.1))
    shade_peaks(ax, certificates["segments"])
    ax.step(times, threshold, where="post", color="#1f77b4", lw=2.5, label="Optimal control")
    ax.set_xlim(0, DAYS)
    ax.set_ylim(0.5, N + 1.5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Server-activation threshold")
    ax.set_xticks(np.arange(DAYS + 1))
    ax.set_yticks([1, 5, 10, 15, 20, N + 1])
    ax.grid(alpha=0.20)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    path = OUTPUT_DIR / "experiment1_threshold_5days_summer_lower_power.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_primitive_comparison(results, certificates):
    configure_plot_style()
    times = results["decision_times"] / 24.0
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6.1), sharex=True, sharey=True)

    for ax in (ax_top, ax_bottom):
        ax.set_xlim(0, DAYS)
        ax.set_ylim(0.5, N + 1.5)
        ax.set_yticks([1, 5, 10, 15, 20, N + 1])
        ax.grid(alpha=0.20, zorder=1)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98, hspace=0.10)
    shade_peaks_across_axes(fig, [ax_top, ax_bottom], certificates["segments"], DAYS)

    optimal_line = ax_top.step(
        times,
        results["threshold"],
        where="post",
        color="#1f77b4",
        lw=2.4,
        label="Optimal control",
        zorder=3,
    )[0]
    peak_proxy = plt.Line2D([0], [0], color="#DCECF7", lw=10, label="Peak period")
    ax_top.legend(handles=[optimal_line, peak_proxy], loc="upper left", frameon=True)

    ax_bottom.plot(
        times,
        certificates["threshold"],
        color="#ff7f0e",
        lw=2.8,
        ls="--",
        label="Primitive-condition certificate",
        zorder=3,
    )
    ax_bottom.axvspan(
        116.0 / 24.0,
        120.0 / 24.0,
        facecolor="#E6E6E6",
        alpha=0.8,
        hatch="//",
        edgecolor="#888888",
        label="Not certified",
        zorder=2,
    )
    ax_bottom.legend(loc="upper left", frameon=True)
    ax_bottom.set_xlabel("Time (days)")
    ax_bottom.set_xticks(np.arange(DAYS + 1))
    fig.text(0.04, 0.52, "Server-activation threshold", rotation="vertical", va="center", fontsize=30)

    path = OUTPUT_DIR / "optimal_vs_primitive_certificates_summer_lower_power.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = solve_hjb()
    certificates = construct_primitive_certificates(results["decision_times"])
    agreement = np.mean(
        certificates["threshold"][certificates["certified"]]
        == results["threshold"][certificates["certified"]]
    )
    print(f"Effective peak cost: {certificates['effective_high_cost']:.9f}")
    print(f"Effective off-peak cost: {certificates['effective_low_cost']:.9f}")
    print(f"Peak arrival mean: {certificates['peak_mean']:.6f}")
    print(f"Off-peak primitive lower bound: {certificates['offpeak_lower_bound']:.12f}")
    print(f"Certification coverage: {np.mean(certificates['certified']):.9%}")
    print(f"Agreement on certified times: {agreement:.9%}")
    for diagnostic in certificates["diagnostics"]:
        print(diagnostic)
    for path in [
        plot_marginal_costs(results, certificates),
        plot_optimal_threshold(results, certificates),
        plot_primitive_comparison(results, certificates),
    ]:
        print(path)


if __name__ == "__main__":
    main()