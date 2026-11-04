#!/usr/bin/env python3
"""Repeatable synthetic benchmark; outputs simulated evidence only."""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from miqaat.simulator import SyntheticWorld, SCENARIOS, advance_state
from miqaat.risk import score, choose_local_action, ACTIONS
from miqaat.intervention_optimizer import optimize_intervention


def rollout(state, action, horizon):
    current, values = state, []
    for _ in range(horizon):
        current = advance_state(current, action)
        values.append(score(current, action.name))
    return values


def run_trial(seed, scenario, steps, horizon):
    world, rows = SyntheticWorld(seed=seed, scenario=scenario), []
    for _ in range(steps):
        state = world.step()
        baseline = score(state)
        best, _ = optimize_intervention(state, horizon=horizon)
        local, _ = choose_local_action(state, horizon)
        no_action = rollout(state, ACTIONS[0], horizon)
        network_action = next(action for action in ACTIONS if action.name == best.action)
        network = rollout(state, network_action, horizon)
        local_action = next(action for action in ACTIONS if action.name == local.intervention)
        local_rows = rollout(state, local_action, horizon)
        rows.append({
            "timestamp": state.timestamp,
            "current_risk": baseline.total_risk,
            "network_horizon_mean": statistics.mean(item.total_risk for item in network),
            "local_horizon_mean": statistics.mean(item.total_risk for item in local_rows),
            "no_action_horizon_mean": statistics.mean(item.total_risk for item in no_action),
            "network_horizon_peak": max(item.total_risk for item in network),
            "local_horizon_peak": max(item.total_risk for item in local_rows),
            "no_action_horizon_peak": max(item.total_risk for item in no_action),
            "warning_minutes_heuristic": baseline.warning_minutes,
            "network_action": best.action,
            "network_objective": best.objective,
            "local_action": local.intervention,
            "sensor_confidence": baseline.confidence,
            "risk_level": baseline.level,
        })
    return rows


def bootstrap_ci(values, replicates=5000, seed=2026):
    """Percentile bootstrap over independent seed-level clusters."""
    if not values:
        return [0.0, 0.0, 0.0]
    center = statistics.mean(values)
    if len(values) == 1:
        return [center, center, center]
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(replicates))
    lower = samples[max(0, int(0.025 * replicates) - 1)]
    upper = samples[min(replicates - 1, int(0.975 * replicates))]
    return [center, lower, upper]


def aggregate(rows):
    if not rows:
        raise ValueError("cannot aggregate an empty benchmark")
    def mean(key):
        return statistics.mean(row[key] for row in rows)

    baseline = mean("no_action_horizon_mean")
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)
    seed_network_means = [
        statistics.mean(row["network_horizon_mean"] for row in seed_rows)
        for seed_rows in by_seed.values()
    ]
    paired_no_action = [
        statistics.mean(row["network_horizon_mean"] - row["no_action_horizon_mean"] for row in seed_rows)
        for seed_rows in by_seed.values()
    ]
    paired_local = [
        statistics.mean(row["network_horizon_mean"] - row["local_horizon_mean"] for row in seed_rows)
        for seed_rows in by_seed.values()
    ]
    network_mean = mean("network_horizon_mean")
    return {
        "intervals": len(rows),
        "mean_current_risk": mean("current_risk"),
        "mean_no_action_horizon_risk": baseline,
        "mean_network_policy_horizon_risk": network_mean,
        "network_policy_seed_cluster_bootstrap_95pct_ci": bootstrap_ci(seed_network_means),
        "independent_seed_count": len(seed_network_means),
        "paired_network_minus_no_action_seed_cluster_bootstrap_95pct_ci": bootstrap_ci(paired_no_action),
        "paired_network_minus_local_seed_cluster_bootstrap_95pct_ci": bootstrap_ci(paired_local),
        "mean_local_policy_horizon_risk": mean("local_horizon_mean"),
        "network_vs_no_action_reduction_pct": 100 * (baseline - network_mean) / baseline if baseline else 0,
        "local_vs_no_action_reduction_pct": 100 * (baseline - mean("local_horizon_mean")) / baseline if baseline else 0,
        "critical_interval_rate_pct": 100 * sum(row["risk_level"] == "CRITICAL" for row in rows) / len(rows),
        "low_confidence_interval_rate_pct": 100 * sum(row["sensor_confidence"] < 0.75 for row in rows) / len(rows),
        "mean_heuristic_warning_category_minutes": mean("warning_minutes_heuristic"),
        "uncertainty_method": "percentile bootstrap over independent seed-level clusters; not over individual time steps",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "mixed", "all"), default="all")
    parser.add_argument("--out", default="benchmark_results")
    args = parser.parse_args()
    if min(args.seeds, args.steps, args.horizon) <= 0:
        parser.error("--seeds, --steps and --horizon must be positive")
    scenarios = [item for item in (*SCENARIOS, "mixed") if args.scenario in ("all", item)]
    all_rows, summaries = [], []
    for scenario in scenarios:
        scenario_rows = []
        for seed in range(args.seeds):
            scenario_rows.extend(
                {"seed": 2026 + seed, **row}
                for row in run_trial(2026 + seed, scenario, args.steps, args.horizon)
            )
        all_rows.extend({"scenario": scenario, **row} for row in scenario_rows)
        summaries.append({"scenario": scenario, **aggregate(scenario_rows)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "benchmark_steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        "label": "SIMULATED SYNTHETIC BENCHMARK — NOT REAL-WORLD VALIDATION",
        "parameters": {"seeds_per_scenario": args.seeds, "steps_per_seed": args.steps, "horizon_steps": args.horizon},
        "scenarios": summaries,
        "overall": aggregate(all_rows),
        "limitations": [
            "Synthetic dynamics, risk coefficients, route topology and intervention effects are assumptions, not calibrated Hajj measurements.",
            "The optimizer evaluates counterfactual simulator rollouts; it is not a learned future-crowd forecast.",
            "Warning categories are heuristic risk mappings, not measured incident lead times.",
            "Bootstrap intervals quantify seed-level simulation variability, not real-world uncertainty.",
        ],
    }
    (out / "benchmark_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
