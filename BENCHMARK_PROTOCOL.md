# Benchmark Protocol

## Research question

In the current synthetic three-zone Hajj/Umrah transport network, does selecting an intervention by projected network-wide risk reduce the mean risk over a short rollout more than (a) doing nothing and (b) choosing an action by immediate risk reduction in the currently worst zone?

## Experimental design

- Scenarios: normal flow, surge, shuttle delay, blocked emergency route, sensor outage, compound stressors, and a mixed periodic scenario.
- Seeds: 20 by default; each seed produces 120 time steps per scenario.
- Decision horizon: 5 simulated steps.
- Policies: network-risk selection, local-worst-zone selection, and no action.
- Primary outcome: mean rollout risk (same risk function applied after each modeled step).
- Secondary outcomes: percent risk reduction against no action, peak rollout risk, critical-risk interval rate, low-confidence interval rate, and heuristic warning threshold.
- Repeatability: fixed seed sequence beginning at 2026. CSV stores step-level outcomes; JSON stores aggregate summaries and assumptions.

Run:

```bash
python3 benchmark_miqaat.py
```

For a quick check:

```bash
python3 benchmark_miqaat.py --seeds 2 --steps 30 --scenario mixed --out benchmark_smoke
```

## Interpretation limits

This benchmark tests internal consistency and policy behavior under transparent synthetic assumptions. The scenario generator, intervention effects, network topology, and risk coefficients are not calibrated using Hajj operational measurements. The generated warning threshold is a heuristic risk-category mapping, not observed lead time. These results do not establish accuracy, safety, causal effectiveness, or readiness for deployment. External validation requires permitted data, documented annotations, pre-registered splits, calibrated flow dynamics, uncertainty analysis, and review with domain operators.
