# MIIQAT-AI

**MIIQAT-AI** is a research prototype for graph-based, network-level crowd-risk analysis and intervention evaluation in Hajj and Umrah scenarios.

The current repository contains a reproducible synthetic zone-flow simulator, a transparent rule-based risk baseline, conservative rule/AI risk fusion, capacity-limited graph transfers, a constraint-aware intervention evaluator, graph-temporal data preparation, simple forecasting baselines, and an optional graph-aggregation + GRU model. It is not connected to live cameras, GPS, gate counters, or operational systems.

## Implemented modules

- Synthetic scenarios: normal flow, surge, shuttle delay, blocked emergency route, sensor outage, compound stressors, and mixed periodic conditions.
- General directed graph validation and iterative downstream risk propagation.
- Interventions that modify only their specified flow controls; population advances once per simulation step.
- Capacity-limited inter-zone transfers; blocked transfers remain at the source zone.
- Rule/AI risk fusion that prevents a HIGH or CRITICAL rule alert from being downgraded by the AI estimate.
- Constraint-aware comparison of no action, gate opening, pedestrian rerouting and shuttle-flow reduction.
- Graph-temporal feature construction with future targets and chronological purging.
- Persistence, moving-average and linear-trend forecasting baselines.
- Optional PyTorch graph-aggregation + GRU model and training utilities.
- Split-conformal interval utility and additive contribution explanations (not SHAP).

## Run the core prototype

Python 3.10 or later; the core simulator and tests use only the Python standard library.

~~~bash
python3 run_miqaat.py --steps 30 --horizon 5
python3 run_miqaat.py --steps 30 --horizon 5 --json
~~~

## Run tests and synthetic benchmark

~~~bash
python3 -m unittest discover -s tests -v
python3 benchmark_miqaat.py
~~~

Quick benchmark:

~~~bash
python3 benchmark_miqaat.py --seeds 2 --steps 30 --scenario mixed --out benchmark_smoke
~~~

The benchmark writes per-step CSV and aggregate JSON outputs. Confidence intervals use seed-cluster bootstrap resampling. They quantify variability under the synthetic generator, not real-world uncertainty.

## Optional graph-temporal model

Install the optional dependency:

~~~bash
pip install -r requirements-ml.txt
~~~

The model and training utilities are in miqaat/forecasting. They have not been trained or validated merely by being included in the repository. Training, hyperparameter selection, calibration and held-out evaluation remain research tasks.

## Separate HAJJv2 count-stream experiment

The existing benchmarks/hajj_count_forecast.py evaluates temporal baselines on human-verified HAJJv2 crowd-count annotations. It predicts future counts from prior annotated counts within a video. It does not run an image-based counter and does not validate MIIQAT's risk or intervention models. See DATASETS.md and BENCHMARK_PROTOCOL.md.

## Research gaps and limitations

See RESEARCH_GAPS.md for the literature-grounded research scope. The present simulator's flow rules, topology, intervention effects, risk coefficients and thresholds are assumptions, not calibrated Hajj measurements. No live sensing, validated incident prediction, operational digital twin, real-time safety assurance or field-tested intervention is claimed.

Do not present synthetic output as recorded Hajj incidents or evidence that the system prevents harm. Operational use would require authorized data, calibrated models, independent validation, uncertainty and failure analysis, human oversight, and privacy/security review.


## Research correctness and validation status

Recent research-branch corrections address persistent-intervention compounding, stepwise reroute feasibility, conformal finite-sample handling, group-held-out evaluation, training-only normalization, and explicit definitions of occupancy ratio, flow accounting, warning categories, and delay proxies.

These corrections are not equivalent to model validation. Confirm the latest CI/test run before merging. Graph-temporal model performance, empirical uncertainty calibration, Hajj-specific calibration, and operational intervention effectiveness remain unverified.
