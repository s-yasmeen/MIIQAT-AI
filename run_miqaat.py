#!/usr/bin/env python3
"""Run one transparent synthetic MIIQAT-AI simulation."""
import argparse
import json

from miqaat.simulator import SyntheticWorld
from miqaat.risk import score
from miqaat.intervention_optimizer import optimize_intervention


def main():
    parser = argparse.ArgumentParser(description="MIIQAT-AI synthetic network simulation")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--scenario", default="mixed",
        choices=("normal", "surge", "shuttle_delay", "blocked_emergency_route",
                 "sensor_outage", "compound", "mixed"),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.horizon <= 0:
        parser.error("--steps and --horizon must be positive")

    world, records = SyntheticWorld(seed=args.seed, scenario=args.scenario), []
    for _ in range(args.steps):
        state = world.step()
        current = score(state)
        recommendation, alternatives = optimize_intervention(state, horizon=args.horizon)
        row = {
            "scenario": args.scenario,
            "timestamp": state.timestamp,
            "current_risk": current.__dict__,
            "recommended_intervention": recommendation.__dict__,
            "alternatives": [item.__dict__ for item in alternatives],
            "data_status": "SIMULATED",
        }
        records.append(row)
        if not args.json:
            print(
                f"t={state.timestamp:03} | risk={current.total_risk:.3f} "
                f"({current.level}) | recommended={recommendation.action} "
                f"| objective={recommendation.objective:.4f} "
                f"| feasible={recommendation.feasible}"
            )
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        print("\nSimulation completed. Inputs and outputs are synthetic; no live sensors were used.")


if __name__ == "__main__":
    main()
