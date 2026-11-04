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


## Corrections and interpretation controls

The current research branch includes regression fixes for persistent gate/shuttle actions, per-step reroute feasibility, and the finite-sample split-conformal quantile. Tests are included, but these changes must be verified by a successful CI run before they are considered validated.

- Scenario/video/site leakage: use `group_holdout_split` with scenario IDs or explicit video/site IDs so groups do not cross partitions. The ordinary chronological split is a complementary within-group evaluation, not a substitute for grouped holdout.
- Preprocessing leakage: fit `fit_feature_normalizer` on training examples only, then reuse its parameters unchanged for validation, calibration, and test data.
- Feature meaning: `capacity_utilization` is crowd_count divided by nominal zone capacity; it is not persons-per-square-metre density. Targets use the same occupancy ratio.
- Flow accounting: the simulator treats the configured fraction of outflow as potential internal movement. The remainder is an external network exit; blocked transfers are returned to their source. This accounting is an explicit synthetic convention and is not calibrated pedestrian dynamics.
- Warning output: warning categories are heuristic risk bands. They are not lead-time forecasts and must not be described as minutes of advance warning.
- Delay metric: the objective's normalized positive inflow-minus-outflow quantity is a flow-imbalance proxy, not measured waiting time or travel time.
- Uncertainty: an infinite conformal radius is the conservative finite-sample result when the required rank exceeds the calibration sample. Do not silently clip it to a finite maximum. Empirical coverage and interval width must be measured on held-out calibration/test data.


## Out-of-model intervention sensitivity

Run the additional sensitivity experiment with:

~~~bash
python benchmarks/intervention_robustness.py --seeds 10 --steps 60 --horizon 5 --transfer-fractions 0.10,0.25,0.40 --scenario all
~~~

The policy is selected by the nominal optimizer (which uses the default transfer fraction of 0.25), then evaluated under alternative transfer fractions. The report includes paired seed-cluster intervals for risk differences, peak risk, and capacity-violation steps. This tests sensitivity to one simulator assumption; it does not validate the other assumptions or establish real-world benefit. The 0.25 evaluation is the nominal condition, while 0.10 and 0.40 are deliberately perturbed conditions.

The HAJJv2-CrowdCount annotation repository currently identifies its license as TBC. The benchmark script accepts user-supplied train/test CSVs, but automated CI must not fetch or redistribute those files until the applicable license and research-use terms are confirmed. The existing count-stream summary is a separate exploratory result and is not evidence for network-level intervention effectiveness.
