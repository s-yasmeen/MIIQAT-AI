# MIIQAT-AI Research Gap and Contribution Definition

## 1. Positioning against prior work

The literature already contains Hajj/Umrah crowd-monitoring, anomaly detection, crowd counting, temporal density prediction, directional-flow analysis, and mathematical intervention optimization. MIIQAT-AI must not claim any of those capabilities as novel in isolation.

| Prior work | Established contribution | Consequence for MIIQAT-AI |
|---|---|---|
| Halboob et al. (2024), IEEE Access, DOI 10.1109/ACCESS.2024.3350188 | Umrah crowd-management intelligence framework using data collection, correlation, rules and alerts | Basic monitoring, rule-based anomaly detection and alerts are not novel claims |
| Alzahrani & Algethami (2025), Electronics 14(13), 2507 | Hajj crowd anomaly detection and LSTM-based forecasting | Generic anomaly detection plus forecasting is not a sufficient novelty claim |
| IEEE ICAIII (2026), DOI 10.1109/ICAIII69475.2026.11521402 | Random Forest classification of future crowd density 60 minutes ahead | Density prediction alone is not a defensible novelty claim |
| Alturki et al. (2026), DOI 10.1007/s44443-026-00793-5 | HAJJv2-derived directional anomaly and counter-flow early-warning research | Directional anomaly detection and proactive warnings alone are not novel |
| Qamloa et al. (2026), DOI 10.1080/16583655.2026.2718559 | Fractional optimal control for guidance, diversion and emergency interventions | Mathematical crowd intervention optimization alone is not novel |
| IEEE Access Hajj/Umrah AI survey (2024), DOI 10.1109/ACCESS.2024.3487923 | Broad taxonomy of AI applications and open challenges | The final paper needs a systematic, updated comparison, not a small set of citations |

## 2. Candidate research gaps to test

These are narrower candidate gaps emerging from the reviewed sources. They are not claims that no related work exists. A systematic search across IEEE Xplore, Scopus, Web of Science, Springer, ScienceDirect and relevant proceedings is required before publication.

### G1 — Connected-zone forecasting rather than isolated-zone prediction

Investigate whether explicit directed connectivity and neighbouring-zone observations improve future zone occupancy/risk prediction over temporal-only models. The empirical test is a graph model versus matched non-graph baselines and a graph-ablation model.

### G2 — Forecast uncertainty that is calibrated separately from sensor reliability

Distinguish sensor reliability (confidence in current observations) from predictive uncertainty (uncertainty about future outcomes). Use a held-out calibration partition and report interval coverage and width by horizon and zone. Do not describe the included conformal utility as calibrated until calibration experiments are run.

### G3 — Constraint-aware intervention evaluation with downstream effects

Evaluate candidate actions across the connected network, including destination capacity, blocked transfers, downstream capacity violations and delay proxies. Compare against no action and a local-worst-zone policy. This is a synthetic counterfactual experiment until the simulator is calibrated and independently validated.

### G4 — Auditable explanations linked to model outputs and intervention trade-offs

Provide feature contributions, forecast uncertainty, feasibility status and a record of why an action was selected. The current additive explanation utility is not SHAP and is not causal. Model-specific explainability must be selected after the predictive model is trained.

## 3. Research questions

- RQ1: Does graph-temporal modelling improve multi-horizon zone occupancy forecasting over temporal-only baselines under a leakage-controlled split?
- RQ2: Does graph connectivity contribute measurable predictive value, as tested by removing or perturbing adjacency information?
- RQ3: Can conformal intervals achieve their nominal empirical coverage on an untouched test set, and how does interval width vary by horizon and zone?
- RQ4: Under explicit synthetic network constraints, does network-wide intervention evaluation reduce risk and capacity-violation burden relative to no action and local selection?

## 4. Hypotheses

- H1: A graph-temporal model will outperform matched temporal-only baselines on at least one pre-registered multi-zone forecasting metric; the result must be reported by horizon and scenario.
- H2: Removing graph connectivity will reduce performance in scenarios where inter-zone dependencies are deliberately present.
- H3: A separately calibrated conformal method will achieve empirical coverage close to its nominal level on exchangeable held-out examples; coverage under distribution shift must be reported separately.
- H4: The constraint-aware policy will reduce the pre-specified synthetic objective without increasing hard-constraint violations relative to the baselines.

These are testable hypotheses, not results.

## 5. Experimental requirements before claiming a contribution

1. Freeze the corrected simulator and record the exact code commit.
2. Document graph topology, units, sampling cadence, transfer assumptions and intervention semantics.
3. Create a synthetic ground-truth benchmark where future states are generated independently of the model under test.
4. Compare persistence, moving average, linear trend, tree-based models where available, temporal GRU/LSTM and GCN-GRU.
5. Use chronological splits with a purge gap and hold out complete scenarios or videos when possible.
6. Report MAE/RMSE by horizon and zone, calibration coverage/width, capacity violations, risk, delay proxy and uncertainty intervals.
7. Run ablations for graph adjacency, temporal modelling, neighbouring-zone propagation, uncertainty and intervention optimization.
8. Repeat over independent seeds and report seed-cluster uncertainty.
9. Separate synthetic simulation results from HAJJv2 count-stream results and any future operational data.
10. Obtain domain review before interpreting simulated interventions as operationally meaningful.

## 6. Current implementation versus research still outstanding

Implemented in the research-upgrade branch: simulator corrections, general graph utilities, risk propagation, graph-temporal feature construction, dependency-free forecast baselines, optional GCN-GRU architecture, conformal interval utility, additive contribution explanations, constraint-aware intervention evaluation and regression tests.

Still outstanding: executing and passing the complete test suite, training the GCN-GRU, tuning hyperparameters, calibrating intervals, running final held-out comparisons and ablations, validating the flow simulator against permitted operational data, and completing a systematic literature review.

## 7. Selected references

1. W. Halboob, H. Altaheri, A. Derhab and J. Al-Muhtadi, “Crowd Management Intelligence Framework: Umrah Use Case,” IEEE Access, 2024. DOI: 10.1109/ACCESS.2024.3350188.
2. “Leveraging Machine Learning for Optimal Pilgrim Crowd Management,” Electronics, 2025, 14(13), 2507.
3. “Crowd Management During Hajj and Umrah Using Machine Learning,” 2026 International Conference on AI Innovations and Industry. DOI: 10.1109/ICAIII69475.2026.11521402.
4. R. Alturki, M. Wedyan and A. Nasayreh, “AI-powered directional anomaly detection in dense hajj crowds for proactive safety management,” Journal of King Saud University Computer and Information Sciences, 2026. DOI: 10.1007/s44443-026-00793-5.
5. A. H. Qamloa, G. M. Bahaab and A. R. Seadawy, “Fractional optimal control of a mathematical model for managing hajj and umrah pilgrims’ crowds,” Journal of Taibah University for Science, 2026. DOI: 10.1080/16583655.2026.2718559.
6. “Enhancing Hajj and Umrah Rituals and Crowd Management Through AI Technologies: A Comprehensive Survey of Applications and Future Directions,” IEEE Access, 2024, pp. 161820–161841. DOI: 10.1109/ACCESS.2024.3487923.
