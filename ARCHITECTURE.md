# Miqaat AI Architecture

## What is implemented

The current repository implements a **synthetic three-zone simulator**, an interpretable risk-scoring function, short what-if rollouts for candidate actions, and a benchmark against two simulated baselines. It does not ingest CCTV, GPS, gate-counter, or weather feeds. It does not implement a camera perception model or an independently tested privacy gate.

## Intended system architecture

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

Each zone is a node; gates, corridors, roads and shuttle routes are edges. Proposed state includes density, capacity, inflow, outflow, speed, direction, vehicle queue, emergency access, weather and sensor confidence.

For each candidate action—open a gate, reroute pedestrians, slow shuttles or do nothing—the prototype's stylized simulator propagates state over a short horizon and computes an assumed risk score:

```text
network risk = zone risk + downstream propagation + vehicle conflict
             + emergency-route penalty + uncertainty penalty
```

The score weights and simulated flow-transfer rules are transparent assumptions. They are not calibrated against field measurements. The current version is for reproducible software and policy experiments only; it is not a live digital twin or operational safety system.

## Planned components

Live sensor adapters, validated anonymous perception, a production privacy gate, site-specific calibration, stronger uncertainty handling, and human approval workflows remain future work. No faces, plates, names or persistent individual trajectories are needed by the present synthetic benchmark, but this does not constitute a tested privacy guarantee.
