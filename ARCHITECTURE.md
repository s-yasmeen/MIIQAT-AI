# MIIQAT-AI Architecture and Implementation Status

## Research objective

MIIQAT-AI investigates network-level crowd-risk forecasting and intervention evaluation across connected Hajj/Umrah zones. The repository is a research prototype using synthetic zone-flow data plus a separate exploratory HAJJv2 count-stream benchmark. It is not connected to live operational systems.

## Current implementation

| Component | Status |
|---|---|
| Synthetic three-zone scenario generator | Implemented |
| Reproducible scenario seeds | Implemented |
| Flow-only intervention application | Implemented |
| Capacity-limited directed inter-zone transfers | Implemented as a stylized assumption |
| General directed graph validation | Implemented |
| Cyclic fixed-point risk propagation | Implemented |
| Transparent rule-based risk baseline | Implemented; coefficients remain assumptions |
| Rule/AI risk fusion with no-downgrade safety floor | Implemented as a decision-support guardrail |
| Constraint-aware intervention evaluator | Implemented for encoded constraints |
| Graph-temporal feature and target construction | Implemented |
| Chronological split with purge gap | Implemented |
| Persistence, moving-average and linear-trend baselines | Implemented |
| Optional graph-aggregation + GRU model | Implemented as code; training and validation not yet performed |
| Optional PyTorch training/evaluation utilities | Implemented as code; not yet run on a research dataset |
| Split-conformal interval utility | Implemented; calibration not yet performed |
| Additive feature contribution explanations | Implemented for additive scores; not SHAP |
| Live CCTV/GPS/gate/weather adapters | Not implemented |
| Hajj-calibrated flow simulator | Not implemented |
| Operational digital-twin validation | Not implemented |
| Trained GCN-GRU model and held-out results | Not implemented |
| Field-tested safety or deployment approval | Not implemented |

## Logical architecture

~~~text
Zone observations / public annotations / synthetic scenario generator
        |
        v
Validation, feature construction and fixed node ordering
        |
        +-------------------------------+
        |                               |
        v                               v
Rule-based safety baseline       Graph-temporal forecasting
        |                        (optional GCN-GRU)
        |                               |
        +---------------+---------------+
                        v
          Conservative rule/AI risk fusion
                        |
                        v
        Constraint-aware intervention evaluation
                        |
                        v
      Scenario simulation and downstream impact metrics
                        |
                        v
     Explainable dashboard / human-reviewed decision support
~~~

## Modelling assumptions

- Each zone is a node. Directed edges represent potential movement between zones.
- Missing edge weights are treated as equal route shares.
- A configurable fraction of outflow is treated as potential inter-zone transfer.
- Transfers that cannot be admitted because a target is at capacity remain in the source zone.
- Shuttle flow is represented separately from total zone inflow where the synthetic generator provides that component.
- The rule-based risk coefficients and risk thresholds are transparent assumptions, not learned parameters or operational standards.
- The intervention delay term is a normalized queue proxy, not measured travel time.
- The risk-fusion floor prevents an AI forecast from downgrading a HIGH or CRITICAL rule alert; this is a software guardrail, not a certified safety case.

## Safety and data governance

The prototype uses aggregate zone-level state and does not require face recognition or persistent individual identities. This design choice is not a tested privacy guarantee. Live data integration would require data minimization, documented authorization, retention limits, access control, security review, and a validated degraded-mode policy.

All outputs must be labelled synthetic or research-only unless supported by separately documented real-world validation. No intervention should be executed automatically in an operational environment; any future use requires qualified human oversight and site-authorized safety procedures.


## Recent correctness fixes

The research branch now:
1. Applies gate/shuttle controls to temporary per-step state without mutating stored flow inputs or repeatedly reducing queues.
2. Rechecks reroute feasibility at every rollout step.
3. Uses the finite-sample split-conformal order statistic, returning an infinite radius when rank (n+1) is required and rejecting non-finite residuals/predictions.
4. Provides scenario/video group-held-out splitting in addition to chronological purging.
5. Provides training-only feature-normalization helpers.
6. Explicitly defines occupancy ratio, external exits, internal transfers, and blocked-transfer accounting.
7. Labels warning categories and delay terms as heuristics/proxies, not measured incident lead time or travel time.

These are implementation corrections and research controls, not evidence of external validity. The latest commit still requires a successful test/CI run and empirical model evaluation.
