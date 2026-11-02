# MIIQAT-AI Validation Protocol

## Purpose and scope

This protocol separates software verification, synthetic simulator experiments, forecasting evaluation, uncertainty calibration, and operational validation. Passing unit tests is software verification; it is not evidence that the system predicts real Hajj/Umrah crowd conditions or improves real-world safety.

## A. Completed software verification

The research-upgrade CI workflow compiles the Python modules, runs the standard-library unit tests, and smoke-tests the CLI and synthetic benchmark. The workflow result should be checked for the exact commit under review.

## B. Repeatable synthetic policy experiment

Run:

```bash
python benchmark_miqaat.py --seeds 10 --steps 60 --horizon 5 --scenario all --out validation_results
```

The benchmark covers normal flow, surge, shuttle delay, blocked emergency route, sensor outage, compound stressors, and mixed conditions. It compares the optimizer-selected policy and local policy with no action under the same synthetic starting states. Save `benchmark_steps.csv` and `benchmark_summary.json` as artifacts.

Interpretation safeguards:
- Report all scenarios, not only the aggregate.
- Report independent seed count and seed-cluster bootstrap intervals.
- Negative risk difference means the policy's simulated horizon-mean risk was lower; it does not establish causal effectiveness in the real world.
- Report capacity-violation and feasibility counts explicitly where available; never infer safety from lower risk score alone.
- The optimizer is a counterfactual simulator-based policy evaluator, not a trained crowd forecaster.
- Synthetic dynamics, risk coefficients, routes and intervention effects are assumptions.

## C. Forecasting validation — not yet established by the simulator benchmark

1. Freeze a dataset manifest, source, licensing/access terms, sampling cadence, zone mapping, missing-data rules, and outcome definitions.
2. Split by whole video/scenario/site where possible. No overlapping windows from a held-out group may appear in training, validation, or calibration partitions.
3. Fit all normalization and feature-selection parameters on training data only.
4. Compare persistence, moving average, linear trend, temporal-only model, and graph-temporal model at every prespecified horizon.
5. Report per-horizon and per-zone MAE/RMSE, aggregated results, repeated seeds, and group-cluster confidence intervals.
6. Tune hyperparameters on validation data only. Do not use the final test set for model selection.
7. Keep synthetic simulation results separate from any HAJJv2-derived observations.

## D. Uncertainty calibration — not yet empirically established

Use a separate calibration partition to fit split-conformal residual quantiles. Evaluate interval coverage and mean interval width on a held-out test partition at each horizon and for relevant subgroups. Report sample counts. An infinite radius is a valid conservative finite-sample outcome when the required order statistic is unavailable; do not replace it with a finite maximum residual.

## E. Intervention evaluation — synthetic evidence only until externally calibrated

Use identical initial states and exogenous random conditions for each policy comparison where feasible. Report:
- mean and peak risk score, with seed-level paired differences and intervals;
- capacity exceedance count and duration;
- reroute feasibility failures;
- flow-imbalance proxy, explicitly not measured travel/wait time;
- outcomes by scenario, including failure modes and adverse trade-offs.

If the simulator does not model a physical quantity, do not report it as a measured outcome.

## F. Operational / external validation — not performed

Before operational claims, independently validate input measurements, calibrate flow dynamics against suitable observed data, assess the safety of interventions with domain experts, document fail-safe behavior and latency, and obtain required approvals. No live camera integration, field trial, calibrated operational digital twin, measured warning lead time, or real-world safety improvement is established by this repository.

## Acceptance gates

- **Software gate:** CI passes on the exact candidate commit.
- **Synthetic-experiment gate:** complete scenario-by-seed artifacts are saved and reviewed; all failed/unsafe cases are disclosed.
- **Forecasting gate:** group-held-out comparison supports claims against prespecified baselines.
- **Uncertainty gate:** empirical held-out coverage and interval width are reported.
- **Operational gate:** external calibration and safety validation are completed before deployment claims.

A gate not evaluated must be marked *not yet validated*, not assumed to pass.
