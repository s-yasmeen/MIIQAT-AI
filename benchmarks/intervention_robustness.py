#!/usr/bin/env python3
"""Out-of-model sensitivity test for synthetic intervention policies.

The optimizer selects actions using the repository's nominal simulator
(transfer_fraction=0.25). The selected action is then evaluated under alternative
transfer fractions. This is a simulator-misspecification stress test, not field
validation or evidence of real-world causal effectiveness.
"""
import argparse
import json
import statistics
from pathlib import Path

from miqaat.simulator import SyntheticWorld, SCENARIOS, advance_state
from miqaat.risk import score, choose_local_action, ACTIONS
from miqaat.intervention_optimizer import optimize_intervention
from benchmark_miqaat import bootstrap_ci


def rollout(state, action, horizon, transfer_fraction):
    current = state
    risks, violations = [], 0
    for _ in range(horizon):
        current = advance_state(current, action, transfer_fraction=transfer_fraction)
        risks.append(score(current, action.name).total_risk)
        violations += sum(
            zone.crowd_count > zone.capacity for zone in current.zones.values()
        )
    return statistics.mean(risks), max(risks), violations


def run_trial(seed, scenario, steps, horizon, transfer_fraction):
    world = SyntheticWorld(seed=seed, scenario=scenario)
    rows = []
    for _ in range(steps):
        state = world.step()
        selected, _ = optimize_intervention(state, horizon=horizon)
        local, _ = choose_local_action(state, horizon)
        network_action = next(item for item in ACTIONS if item.name == selected.action)
        local_action = next(item for item in ACTIONS if item.name == local.intervention)
        no_action = ACTIONS[0]
        network = rollout(state, network_action, horizon, transfer_fraction)
        local_result = rollout(state, local_action, horizon, transfer_fraction)
        baseline = rollout(state, no_action, horizon, transfer_fraction)
        rows.append({
            "network_risk": network[0],
            "local_risk": local_result[0],
            "no_action_risk": baseline[0],
            "network_peak": network[1],
            "local_peak": local_result[1],
            "no_action_peak": baseline[1],
            "network_capacity_violation_steps": network[2],
            "local_capacity_violation_steps": local_result[2],
            "no_action_capacity_violation_steps": baseline[2],
            "network_action": selected.action,
        })
    return rows


def summarize(seed_rows):
    seed_means = {
        policy: [
            statistics.mean(row[f"{policy}_risk"] for row in rows)
            for rows in seed_rows.values()
        ]
        for policy in ("network", "local", "no_action")
    }
    paired_no_action = [
        statistics.mean(row["network_risk"] - row["no_action_risk"] for row in rows)
        for rows in seed_rows.values()
    ]
    paired_local = [
        statistics.mean(row["network_risk"] - row["local_risk"] for row in rows)
        for rows in seed_rows.values()
    ]
    all_rows = [row for rows in seed_rows.values() for row in rows]
    mean = lambda key: statistics.mean(row[key] for row in all_rows)
    no_action_mean = mean("no_action_risk")
    network_mean = mean("network_risk")
    return {
        "independent_seed_count": len(seed_rows),
        "network_mean_risk": network_mean,
        "local_mean_risk": mean("local_risk"),
        "no_action_mean_risk": no_action_mean,
        "network_reduction_vs_no_action_pct": (
            100.0 * (no_action_mean - network_mean) / no_action_mean
            if no_action_mean else 0.0
        ),
        "paired_network_minus_no_action_seed_cluster_bootstrap_95pct_ci": bootstrap_ci(paired_no_action),
        "paired_network_minus_local_seed_cluster_bootstrap_95pct_ci": bootstrap_ci(paired_local),
        "network_capacity_violation_steps": sum(row["network_capacity_violation_steps"] for row in all_rows),
        "local_capacity_violation_steps": sum(row["local_capacity_violation_steps"] for row in all_rows),
        "no_action_capacity_violation_steps": sum(row["no_action_capacity_violation_steps"] for row in all_rows),
        "network_peak_risk_mean": mean("network_peak"),
        "local_peak_risk_mean": mean("local_peak"),
        "no_action_peak_risk_mean": mean("no_action_peak"),
        "interpretation": "synthetic simulator sensitivity only; not real-world validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--transfer-fractions", default="0.10,0.25,0.40")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "mixed", "all"), default="all")
    parser.add_argument("--out", default="intervention_robustness_results")
    args = parser.parse_args()
    if min(args.seeds, args.steps, args.horizon) <= 0:
        parser.error("seeds, steps and horizon must be positive")
    fractions = [float(value) for value in args.transfer_fractions.split(",")]
    if not fractions or any(not 0.0 <= value <= 1.0 for value in fractions):
        parser.error("transfer fractions must be comma-separated values in [0,1]")
    scenarios = [item for item in (*SCENARIOS, "mixed") if args.scenario in ("all", item)]

    report = {
        "label": "SYNTHETIC OUT-OF-MODEL INTERVENTION SENSITIVITY — NOT FIELD VALIDATION",
        "policy_selection_transfer_fraction": 0.25,
        "evaluation_transfer_fractions": {},
        "parameters": {
            "seeds_per_scenario": args.seeds,
            "steps_per_seed": args.steps,
            "horizon_steps": args.horizon,
            "scenarios": scenarios,
        },
        "limitations": [
            "Only the simulator transfer fraction is perturbed; topology, risk coefficients, capacities and intervention mechanisms remain stylized assumptions.",
            "Policies are selected using the nominal simulator and evaluated under perturbed rollout dynamics.",
            "This experiment cannot establish real-world intervention safety, causal impact or operational benefit.",
            "A real-world evaluation requires independently measured intervention outcomes and calibrated network-flow observations.",
        ],
    }
    for fraction in fractions:
        fraction_report = {}
        for scenario in scenarios:
            by_seed = {}
            for seed_index in range(args.seeds):
                seed = 2026 + seed_index
                by_seed[seed] = run_trial(
                    seed, scenario, args.steps, args.horizon, fraction
                )
            fraction_report[scenario] = summarize(by_seed)
        report["evaluation_transfer_fractions"][str(fraction)] = fraction_report

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "intervention_robustness_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
