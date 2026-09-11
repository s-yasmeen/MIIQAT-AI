# Miqaat AI Architecture

Privacy-preserving, network-level decision support for Hajj and Umrah crowd–traffic management.

```text
Cameras + shuttle GPS + gate counters + weather
  -> anonymous state estimation
  -> sensor reliability and degraded-mode fallback
  -> live zone graph and digital twin
  -> adaptive network-risk prediction
  -> what-if intervention simulation
  -> compare total risk and risk propagation
  -> explainable recommendation
  -> human approval and response
  -> observe outcome and re-estimate
```

Each zone is a node; gates, corridors, roads and shuttle routes are edges. State includes density, capacity, inflow, outflow, speed, direction, vehicle queue, emergency access, weather and sensor confidence.

For each candidate action—open gate, reroute pedestrians, slow shuttles or do nothing—the digital twin propagates flow and computes:

```text
network risk = zone risk + downstream propagation + vehicle conflict
             + emergency-route penalty + uncertainty penalty
```

Miqaat recommends the safest network-wide action, not merely the action that improves one zone. The first risk model is interpretable and rule/ML based. LSTM or a temporal Transformer is optional after sufficient sequential data is available.

Modes: live monitoring, what-if simulation, staff training, and sensor-degraded operation. No faces, plates, names or persistent individual trajectories are retained.
