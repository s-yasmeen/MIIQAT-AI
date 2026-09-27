#!/usr/bin/env python3
"""Repeatable synthetic benchmark; outputs simulated evidence only."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from miqaat.simulator import SyntheticWorld, SCENARIOS, advance_state
from miqaat.risk import score, choose_action, choose_local_action, ACTIONS


def rollout(state, action, horizon):
    current = state
    values = []
    for _ in range(horizon):
        current = advance_state(current, action)
        values.append(score(current, action.name))
    return values


def run_trial(seed, scenario, steps, horizon):
    world = SyntheticWorld(seed=seed, scenario=scenario)
    rows = []
    for _ in range(steps):
        state = world.step()
        baseline = score(state)
        best, _ = choose_action(state, horizon)
        local, _ = choose_local_action(state, horizon)
        no_op = ACTIONS[0].__class__("no_action", "Do nothing")
        no_action = rollout(state, no_op, horizon)
        network_action = next(a for a in ACTIONS if a.name == best.intervention)
        network = rollout(state, network_action, horizon)
        local_action = next(a for a in ACTIONS if a.name == local.intervention)
        local_rows = rollout(state, local_action, horizon)
        rows.append({
            "timestamp": state.timestamp,
            "current_risk": baseline.total_risk,
            "network_horizon_mean": statistics.mean(x.total_risk for x in network),
            "local_horizon_mean": statistics.mean(x.total_risk for x in local_rows),
            "no_action_horizon_mean": statistics.mean(x.total_risk for x in no_action),
            "network_horizon_peak": max(x.total_risk for x in network),
            "local_horizon_peak": max(x.total_risk for x in local_rows),
            "no_action_horizon_peak": max(x.total_risk for x in no_action),
            "warning_minutes": baseline.warning_minutes,
            "network_action": best.intervention,
            "local_action": local.intervention,
            "sensor_confidence": baseline.confidence,
            "risk_level": baseline.level,
        })
    return rows


def aggregate(rows):
    def mean(key):
        return statistics.mean(row[key] for row in rows)
    baseline = mean("no_action_horizon_mean")
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row["network_horizon_mean"])
    seed_means = [statistics.mean(values) for values in by_seed.values()]
    seed_sd = statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0
    ci_half = 1.96 * seed_sd / math.sqrt(len(seed_means)) if seed_means else 0.0
    network_mean = mean("network_horizon_mean")
    return {
        "intervals": len(rows),
        "mean_current_risk": mean("current_risk"),
        "mean_no_action_horizon_risk": baseline,
        "mean_network_policy_horizon_risk": network_mean,
        "network_policy_seed_level_95pct_normal_ci": [max(0.0, network_mean - ci_half), min(1.0, network_mean + ci_half)],
        "independent_seed_count": len(seed_means),
        "mean_local_policy_horizon_risk": mean("local_horizon_mean"),
        "network_vs_no_action_reduction_pct": 100 * (baseline - mean("network_horizon_mean")) / baseline if baseline else 0,
        "local_vs_no_action_reduction_pct": 100 * (baseline - mean("local_horizon_mean")) / baseline if baseline else 0,
        "critical_interval_rate_pct": 100 * sum(row["risk_level"] == "CRITICAL" for row in rows) / len(rows),
        "low_confidence_interval_rate_pct": 100 * sum(row["sensor_confidence"] < .75 for row in rows) / len(rows),
        "warning_minutes_mean": mean("warning_minutes"),
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
    scenarios = [x for x in (*SCENARIOS, "mixed") if args.scenario in ("all", x)]
    all_rows = []
    summaries = []
    for scenario in scenarios:
        scenario_rows = []
        for seed in range(args.seeds):
            scenario_rows.extend({"seed": 2026 + seed, **row} for row in run_trial(2026 + seed, scenario, args.steps, args.horizon))
        all_rows.extend({"scenario": scenario, **row} for row in scenario_rows)
        summaries.append({"scenario": scenario, **aggregate(scenario_rows)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "benchmark_steps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        "label": "SIMULATED SYNTHETIC BENCHMARK — NOT REAL-WORLD VALIDATION",
        "parameters": {"seeds_per_scenario": args.seeds, "steps_per_seed": args.steps, "horizon_steps": args.horizon},
        "scenarios": summaries,
        "overall": aggregate(all_rows),
        "limitations": [
            "Synthetic dynamics and risk coefficients are assumptions, not calibrated Hajj site measurements.",
            "The horizon rollout reuses current sensor state; it is a decision-support experiment, not a validated forecast.",
            "Warning minutes are heuristic labels and are not measured incident lead time.",
        ],
    }
    (out / "benchmark_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
