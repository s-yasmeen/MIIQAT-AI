# Benchmark Protocol

## Research question

In the current synthetic directed zone network, does a constraint-aware intervention evaluator reduce simulated network risk and capacity violations compared with (a) doing nothing and (b) selecting an action by immediate risk in the currently worst zone?

## Experimental design

- Scenarios: normal flow, surge, shuttle delay, blocked emergency route, sensor outage, compound stressors, and mixed periodic conditions.
- Seeds: 20 by default; each seed produces 120 time steps per scenario.
- Decision horizon: 5 simulated steps.
- Policies: constraint-aware network objective, local-worst-zone baseline, and no action.
- Primary outcome: mean simulated rollout risk.
- Secondary outcomes: peak rollout risk, mean delay proxy, capacity-violation steps, critical-risk interval rate, low-sensor-reliability rate, and heuristic warning category.
- Repeatability: fixed seed sequence beginning at 2026. CSV stores step-level outcomes; JSON stores aggregate summaries and assumptions.
- Uncertainty: percentile bootstrap intervals are calculated over independent seed-level clusters. Individual time steps are not treated as independent observations.

Run:

~~~bash
python3 benchmark_miqaat.py
~~~

Quick smoke test:

~~~bash
python3 benchmark_miqaat.py --seeds 2 --steps 30 --scenario mixed --out benchmark_smoke
~~~

## Interpretation limits

The benchmark tests internal consistency and policy behavior under transparent synthetic assumptions. The scenario generator, intervention effects, network topology, flow-transfer fraction, and risk coefficients are not calibrated using Hajj operational measurements. Bootstrap intervals quantify variability across synthetic seeds only. They do not establish real-world uncertainty, safety, causal effectiveness, or readiness for deployment.

The intervention evaluator uses a transparent objective combining mean risk, a normalized queue/delay proxy, and capacity-violation penalties. It is not a certified traffic optimizer, does not model pedestrian biomechanics, and does not establish that a real intervention is safe.

The risk warning category is a heuristic mapping, not observed incident lead time. External validation requires permitted data, documented annotations, pre-registered splits, calibrated flow dynamics, independent evaluation, and review with domain operators.

## Graph-temporal forecasting experiment

The feature pipeline builds ordered graph-temporal examples with past-only features and future occupancy targets. Horizons are measured in sample intervals, not minutes. The chronological split purges examples around the split boundary to reduce overlapping-window target leakage.

Dependency-free baselines are in miqaat/forecasting/baselines.py. The optional graph-aggregation + GRU architecture is in miqaat/forecasting/gcn_gru.py and requires PyTorch (pip install -r requirements-ml.txt). The model is a research implementation, not a trained or validated model. Use a fixed node order and a row-normalized adjacency matrix from miqaat.graph_engine.normalized_adjacency.

Recommended evaluation:
1. Split by time and, where possible, hold out entire videos/scenarios/sites.
2. Fit all scalers and learned parameters on training data only.
3. Use a separate calibration subset for conformal intervals.
4. Keep a final untouched test set.
5. Report MAE/RMSE by horizon and zone, calibration coverage/width, and seed/video-cluster uncertainty.
6. Compare persistence, moving average, linear trend, tree-based models where available, temporal-only GRU/LSTM, and graph-temporal GCN-GRU.

## Separate HAJJv2 count-stream experiment

The existing HAJJv2 count-stream benchmark is a separate exploratory experiment on human-verified annotations. It predicts a future count from prior counts within the same video. It is not image-based crowd counting and does not validate the synthetic zone simulator, intervention effects, or the graph-temporal model.
