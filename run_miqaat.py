#!/usr/bin/env python3
import argparse, json
from miqaat.simulator import SyntheticWorld
from miqaat.risk import score, choose_action

def main():
    p=argparse.ArgumentParser(description="Miqaat AI digital-twin prototype")
    p.add_argument("--steps",type=int,default=30); p.add_argument("--json",action="store_true"); args=p.parse_args()
    world=SyntheticWorld(); records=[]
    for _ in range(args.steps):
        state=world.step(); best, alternatives=choose_action(state); baseline=score(state)
        row={"timestamp":state.timestamp,"baseline":baseline.__dict__,"recommended":best.__dict__,"alternatives":[x.__dict__ for x in alternatives]}; records.append(row)
        if not args.json: print(f"t={state.timestamp:03} | baseline={baseline.level:<8} {baseline.total_risk:.2f} | recommendation={best.intervention} {best.total_risk:.2f} | confidence={best.confidence:.2f}")
    if args.json: print(json.dumps(records,indent=2))
    else: print("\\nMiqaat AI completed. Operational data contains aggregates only; no identities were retained.")
if __name__ == "__main__": main()
