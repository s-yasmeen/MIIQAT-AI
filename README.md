# Miqaat AI

Privacy-preserving, network-level predictive crowd and shuttle-traffic management for Hajj and Umrah scenarios.

Miqaat AI uses a digital twin and multimodal sensor fusion to predict hazardous crowd–traffic conditions, test interventions before applying them, and select the action that reduces total network risk.

## Architecture

`camera/telemetry adapters -> anonymous perception -> sensor confidence -> network flow -> adaptive risk forecast -> what-if policy -> human-approved response`

The prototype has five replaceable layers:

1. **Input adapters:** synthetic stream now; later CCTV/RTSP, bus GPS, gate counters and weather.
2. **Anonymous perception:** density, direction, speed, inflow/outflow, vehicles and route state. No face/plate identity is stored.
3. **Privacy gate:** tests that only aggregate signals leave the system.
4. **Risk engine:** combines density, growth, bus conflict and emergency-lane availability.
5. **Policy engine:** produces human-reviewable actions: monitor, open alternate gate, slow arrivals, pause inflow, reroute shuttles, dispatch responders.

See [ARCHITECTURE.md](ARCHITECTURE.md) and [DATASETS.md](DATASETS.md).

## Run

```bash
python3 miqaatflow_guardian.py
python3 miqaat_ai.py --steps 120
python3 miqaat_ai.py --steps 60 --html miqaat_report.html
python3 miqaat_ai.py --steps 20 --json > run.jsonl
```

This is an executable architecture demonstrator, not a certified safety system. Before operational use, replace the simulator with validated models, real site calibration, human approval, fail-safe testing, and formal privacy/legal review.
