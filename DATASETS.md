# Dataset and Evaluation Plan

No single public dataset is known to provide synchronized Hajj crowd counts, inter-zone flow, shuttle telemetry, route capacities, emergency-access state, intervention outcomes and future-risk labels. Dataset roles must therefore remain separate.

| Data source | Appropriate use | Does not establish |
|---|---|---|
| Existing MIIQAT synthetic scenarios | Simulator regression, controlled network-flow experiments | Real Hajj dynamics |
| HAJJv2 videos/annotations | Hajj-scene perception and task-specific visual evaluation, subject to access/licensing | Network intervention effectiveness |
| HAJJv2-CrowdCount annotations | Count-stream experiments and count-model benchmarking | Hajj-wide crowd-risk forecasting |
| UCF-QNRF / ShanghaiTech | General crowd-counting benchmarks | Site-specific Hajj calibration |
| ETH / UCY trajectories | Pedestrian trajectory methods on their original domains | Hajj route or capacity calibration |
| Synthetic graph sequences | Controlled graph-temporal forecasting and ablation | Operational forecast accuracy |

## Separation of tasks

1. **Perception/counting:** estimate aggregate counts from images or video. Report dataset-specific visual metrics.
2. **Temporal forecasting:** predict future counts from historical count observations. Use chronological and video-separated splits.
3. **Network forecasting:** predict zone-level future states using connected-zone features. This requires graph sequences with documented topology.
4. **Intervention evaluation:** compare counterfactual actions in a calibrated or explicitly synthetic simulator. Do not infer causal intervention effects from observational count datasets.

## Required schema for future graph sequences

~~~text
timestamp, zone_id, capacity, crowd_count, inflow, outflow,
mean_speed, bus_queue, emergency_route_open,
camera_confidence, gps_confidence, gate_confidence,
heat_index, incoming_edges, outgoing_edges, scenario_id
~~~

Add directional-flow variables only when supported by a documented annotation or validated sensor process. Keep individual identities and persistent trajectories out of the research dataset unless a separately approved necessity and governance case exists.

## Split and leakage controls

- Preserve temporal ordering.
- Hold out complete videos, sites or scenarios where the data permit.
- Fit scalers and learned preprocessing only on training data.
- Use a purge gap at least as large as the maximum forecast horizon for overlapping windows.
- Reserve a calibration subset for uncertainty intervals.
- Keep the final test set untouched until model selection is complete.

Every dataset entry must record its source, license/access conditions, annotation provenance, sampling cadence, unit, missingness, and permitted research use.
