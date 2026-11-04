"""Transparent rule-based risk baseline; coefficients are assumptions, not learned values."""
from dataclasses import dataclass

from .graph_engine import propagate_max_risk
from .models import Intervention, NetworkState, RiskResult
from .simulator import advance_state


@dataclass(frozen=True)
class RiskWeights:
    density: float = 0.48
    growth: float = 0.18
    vehicle_conflict: float = 0.16
    environment: float = 0.08
    downstream: float = 0.10
    uncertainty: float = 0.12
    emergency_route_penalty: float = 0.18

    def __post_init__(self):
        names = ("density", "growth", "vehicle_conflict", "environment",
                 "downstream", "uncertainty", "emergency_route_penalty")
        for name in names:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} weight cannot be negative")
        if self.downstream >= 1.0:
            raise ValueError("downstream coupling must be less than 1")


DEFAULT_WEIGHTS = RiskWeights()


def _base_zone_risk(zone, heat: float, weights: RiskWeights) -> float:
    density = min(1.5, zone.density)
    growth = max(0.0, zone.inflow - zone.outflow) / max(1, zone.capacity)
    conflict = min(1.0, zone.bus_queue / 8.0 + max(0.0, zone.inflow - zone.outflow) / 50.0)
    reliability = min(zone.camera_confidence, zone.gps_confidence, zone.gate_confidence)
    uncertainty_penalty = weights.uncertainty * (1.0 - reliability)
    emergency_penalty = weights.emergency_route_penalty if not zone.emergency_route_open else 0.0
    value = (
        weights.density * density + weights.growth * growth
        + weights.vehicle_conflict * conflict + weights.environment * heat
        + uncertainty_penalty + emergency_penalty
    )
    return min(1.0, max(0.0, value))


def score(state: NetworkState, intervention: str = "baseline", weights: RiskWeights = DEFAULT_WEIGHTS):
    if not state.zones:
        raise ValueError("cannot score an empty network")
    base = {zone_id: _base_zone_risk(zone, state.heat_index, weights)
            for zone_id, zone in state.zones.items()}
    propagated = propagate_max_risk(base, state.edges, coupling=weights.downstream)
    total = min(1.0, sum(propagated.values()) / len(propagated))
    confidence = sum(
        min(zone.camera_confidence, zone.gps_confidence, zone.gate_confidence)
        for zone in state.zones.values()
    ) / len(state.zones)
    # Heuristic categories, not measured time-to-incident estimates.
    if total >= 0.72:
        level, warning = "CRITICAL", 1
    elif total >= 0.48:
        level, warning = "HIGH", 3
    elif total >= 0.28:
        level, warning = "MEDIUM", 5
    else:
        level, warning = "LOW", 0
    worst = max(propagated, key=propagated.get)
    zone = state.zones[worst]
    explanation = (
        f"Highest-risk zone {worst}: utilization={zone.density:.2f}, "
        f"inflow={zone.inflow:.1f}, outflow={zone.outflow:.1f}, "
        f"bus_queue={zone.bus_queue}, "
        f"emergency_route={'open' if zone.emergency_route_open else 'blocked'}."
    )
    if confidence < 0.75:
        explanation += " Sensor reliability is reduced; manual verification is recommended."
    return RiskResult(
        round(total, 4), level, warning,
        {key: round(value, 4) for key, value in propagated.items()},
        explanation, round(confidence, 4), intervention,
    )


ACTIONS = [
    Intervention("no_action", "No intervention"),
    Intervention("open_gate_B", "Open alternate gate B", gate_open="B"),
    Intervention("reroute_A_to_C", "Redirect a bounded share of A inflow to C",
                 reroute_from="A", reroute_to="C", reroute_fraction=0.25),
    Intervention("slow_shuttles", "Reduce tracked shuttle inflow by 35%",
                 shuttle_factor=0.65, shuttle_zone="B"),
]


def what_if(state: NetworkState, horizon: int = 5):
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    results = []
    for action in ACTIONS:
        rollout, scores = state, []
        for _ in range(horizon):
            rollout = advance_state(rollout, action)
            scores.append(score(rollout, action.name))
        mean_risk = sum(item.total_risk for item in scores) / len(scores)
        last = scores[-1]
        results.append(RiskResult(
            round(mean_risk, 4), last.level, last.warning_minutes, last.zone_risk,
            f"{last.explanation} Simulated horizon peak risk={max(x.total_risk for x in scores):.4f}.",
            last.confidence, action.name,
        ))
    return results


def choose_action(state: NetworkState, horizon: int = 5):
    alternatives = what_if(state, horizon)
    return min(alternatives, key=lambda item: item.total_risk), alternatives


def choose_local_action(state: NetworkState, horizon: int = 5):
    """Baseline chooses by immediate risk change in the currently worst zone."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    baseline = score(state)
    worst_zone = max(baseline.zone_risk, key=baseline.zone_risk.get)
    evaluated = []
    for action in ACTIONS:
        candidate = score(advance_state(state, action), action.name)
        evaluated.append((candidate.zone_risk.get(worst_zone, 0.0), action.name, action))
    chosen = min(evaluated, key=lambda item: (item[0], item[1]))[2]
    rollout, scores = state, []
    for _ in range(horizon):
        rollout = advance_state(rollout, chosen)
        scores.append(score(rollout, chosen.name))
    last = scores[-1]
    result = RiskResult(
        round(sum(item.total_risk for item in scores) / len(scores), 4),
        last.level, last.warning_minutes, last.zone_risk,
        f"Local baseline selected {chosen.name} using immediate worst-zone risk.",
        last.confidence, chosen.name,
    )
    return result, what_if(state, horizon)
