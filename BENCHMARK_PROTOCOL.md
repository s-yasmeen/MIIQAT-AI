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
- Uncertainty: normal-approximation 95% intervals use independent seed-level means; paired policy differences are computed within each seed. Time steps are not treated as independent samples.

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

## Separate HAJJv2 count-stream experiment

The repository also includes an exploratory one-step-ahead forecasting benchmark on the public HAJJv2-CrowdCount annotations. It predicts the next count from prior human-verified counts in the same test video after a three-sample warm-up. The compared baselines are persistence, a three-count moving average, a five-observation linear trend, and the global mean of training annotations. MAE, RMSE, bias, and video-cluster bootstrap intervals are reported.

This is a temporal forecasting task with observed count history. It is not image-based counting, does not reproduce the YOLO-World/SAM3Count/APGCC benchmark, and does not evaluate the MIIQAT risk or intervention policy. In the eight-video main test subset, 124 forecasts were scored after warm-up; persistence MAE was 6.59 people. See `benchmarks/hajj_count_forecast_summary.json` and run `benchmarks/hajj_count_forecast.py` using the train/test CSV files from the upstream annotation repository. The original video files are not redistributed by that repository and were not used in this experiment.
