#!/usr/bin/env python3
"""Run one transparent synthetic Miqaat AI simulation."""
import argparse
import json

from miqaat.simulator import SyntheticWorld
from miqaat.risk import score, choose_action


def main():
    parser = argparse.ArgumentParser(description="Miqaat AI synthetic digital-twin prototype")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenario", default="mixed",
                        choices=("normal", "surge", "shuttle_delay",
                                 "blocked_emergency_route", "sensor_outage",
                                 "compound", "mixed"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.horizon <= 0:
        parser.error("--steps and --horizon must be positive")

    world = SyntheticWorld(seed=args.seed, scenario=args.scenario)
    records = []
    for _ in range(args.steps):
        state = world.step()
        current = score(state)
        recommendation, alternatives = choose_action(state, args.horizon)
        row = {
            "scenario": args.scenario,
            "timestamp": state.timestamp,
            "current_risk": current.__dict__,
            "recommended_policy": recommendation.__dict__,
            "alternatives": [result.__dict__ for result in alternatives],
            "data_status": "SIMULATED",
        }
        records.append(row)
        if not args.json:
            print(
                f"t={state.timestamp:03} | current={current.level:<8} "
                f"{current.total_risk:.3f} | projected policy="
                f"{recommendation.intervention} {recommendation.total_risk:.3f} "
                f"(mean risk over {args.horizon} steps)"
            )
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        print("\nSimulation completed. Inputs and outputs are synthetic; no live sensors were used.")


if __name__ == "__main__":
    main()
