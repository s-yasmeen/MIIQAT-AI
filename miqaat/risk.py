from .models import NetworkState, RiskResult, Intervention
from .simulator import apply_intervention

def _zone_risk(z, downstream_risk: float, heat: float) -> float:
    density = min(1.0, z.density)
    growth = max(0.0, z.inflow - z.outflow) / max(1, z.capacity)
    conflict = min(1.0, z.bus_queue / 8.0 + max(0.0, z.inflow-z.outflow) / 50.0)
    emergency_penalty = 0.18 if not z.emergency_route_open else 0.0
    uncertainty = 0.12 * (1 - min(z.camera_confidence, z.gps_confidence, z.gate_confidence))
    return min(1.0, .48*density + .18*growth + .16*conflict + .08*heat + .10*downstream_risk + emergency_penalty + uncertainty)

def score(state: NetworkState, intervention: str = "baseline") -> RiskResult:
    risks = {}
    for zone_id in reversed(list(state.zones)):
        downstream = max((risks.get(x, 0) for x in state.edges.get(zone_id, [])), default=0)
        risks[zone_id] = _zone_risk(state.zones[zone_id], downstream, state.heat_index)
    total = min(1.0, sum(risks.values()) / max(1, len(risks)))
    confidence = sum(min(z.camera_confidence, z.gps_confidence, z.gate_confidence) for z in state.zones.values()) / len(state.zones)
    if total >= .72: level, warning = "CRITICAL", 1
    elif total >= .48: level, warning = "HIGH", 3
    elif total >= .28: level, warning = "MEDIUM", 5
    else: level, warning = "LOW", 0
    worst = max(risks, key=risks.get)
    z = state.zones[worst]
    explanation = f"Zone {worst}: density={z.density:.2f}, inflow={z.inflow:.1f}, outflow={z.outflow:.1f}, bus_queue={z.bus_queue}, emergency_route={'open' if z.emergency_route_open else 'blocked'}."
    if confidence < .75: explanation += " Sensor confidence is reduced; manual verification is recommended."
    return RiskResult(round(total, 3), level, warning, {k: round(v,3) for k,v in risks.items()}, explanation, round(confidence,3), intervention)

def what_if(state: NetworkState) -> list[RiskResult]:
    actions = [
        Intervention("open_gate_B", "Open alternate gate B", gate_open="B"),
        Intervention("reroute_A_to_C", "Redirect 25% of A inflow to C", reroute_from="A", reroute_to="C"),
        Intervention("slow_shuttles", "Reduce shuttle inflow by 35%", shuttle_factor=.65),
    ]
    return [score(apply_intervention(state, a), a.name) for a in actions]

def choose_action(state: NetworkState) -> tuple[RiskResult, list[RiskResult]]:
    baseline = score(state)
    alternatives = what_if(state)
    candidates = [baseline] + alternatives
    best = min(candidates, key=lambda x: x.total_risk)
    return best, alternatives
