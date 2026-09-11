# Dataset Plan

No single public dataset contains Hajj crowds, shuttle telemetry, emergency routes, intervention outcomes and future-risk labels. Public data will train perception; controlled digital-twin data will evaluate prediction and interventions.

| Module | Recommended data | Output |
|---|---|---|
| Hajj perception | [HAJJv2](https://github.com/KAU-Smart-Crowd/HAJJv2_dataset) | Hajj scenes, abnormal movement and objects |
| Hajj counting | [HAJJv2-CrowdCount](https://arxiv.org/abs/2607.07322) | Dense-count accuracy and confidence |
| Traffic anomalies | [AI City Track 3](https://www.aicitychallenge.org/2026-track3/) | Traffic anomalies and explanations |
| Pedestrian safety | [AI City Track 2](https://www.aicitychallenge.org/2026-track2/) | Pedestrian–vehicle interactions |
| Vehicle flow | [CityFlow](https://arxiv.org/abs/1903.09254) | Multi-camera vehicle flow |
| Dense-crowd robustness | [UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/) | Extreme-density robustness |
| Risk forecasting | Custom digital-twin sequences | Time-to-risk and propagation |
| Intervention testing | Custom counterfactual scenarios | Before/after total network risk |

## Custom scenario schema

```text
timestamp, zone_id, capacity, crowd_count, inflow, outflow,
mean_speed, dominant_direction, bus_queue, emergency_route_open,
camera_confidence, gps_confidence, gate_confidence, heat_index,
intervention, downstream_risk, total_network_risk
```

Scenarios: normal flow, rising inflow, opposing flow, shuttle delay, blocked emergency route, gate opening, pedestrian rerouting, shuttle throttling, camera failure and GPS failure. Synthetic scenarios must be labelled simulated and never presented as recorded Hajj incidents.
