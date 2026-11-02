"""Constraint-aware evaluation of candidate interventions in the synthetic simulator."""
from dataclasses import dataclass

from .models import Intervention
from .risk import ACTIONS, score
from .simulator import advance_state


@dataclass
class InterventionEvaluation:
    action: str
    feasible: bool
    mean_risk: float
    peak_risk: float
    mean_delay_proxy: float
    capacity_violation_steps: int
    objective: float
    reason: str


def _feasibility(state, action: Intervention):
    if action.reroute_from or action.reroute_to:
        if not action.reroute_from or not action.reroute_to:
            return False, "rerouting requires both source and target zones"
        if action.reroute_from not in state.zones or action.reroute_to not in state.zones:
            return False, "unknown reroute zone"
        source = state.zones[action.reroute_from]
        target = state.zones[action.reroute_to]
        requested = source.inflow * action.reroute_fraction
        room = max(0.0, target.capacity - target.crowd_count - target.inflow)
        if requested > room + 1e-9:
            return False, "reroute target lacks capacity for the requested diversion"
    if action.gate_open and action.gate_open not in state.zones:
        return False, "unknown gate zone"
    if action.shuttle_factor != 1.0 and action.shuttle_zone not in state.zones:
        return False, "unknown shuttle zone"
    return True, "feasible under the encoded constraints"


def evaluate_interventions(
    state,
    actions=None,
    horizon: int = 5,
    delay_weight: float = 0.10,
    capacity_penalty: float = 1.0,
):
    if horizon <= 0 or delay_weight < 0 or capacity_penalty < 0:
        raise ValueError("invalid horizon or objective weights")
    actions = ACTIONS if actions is None else actions
    evaluations = []
    for action in actions:
        feasible, reason = _feasibility(state, action)
        if not feasible:
            evaluations.append(
                InterventionEvaluation(action.name, False, float("inf"), float("inf"),
                                       float("inf"), 0, float("inf"), reason)
            )
            continue
        current = state
        risks, delay_proxies, violations = [], [], 0
        for _ in range(horizon):
            current = advance_state(current, action)
            risks.append(score(current, action.name).total_risk)
            delay_proxies.append(
                sum(max(0.0, zone.inflow - zone.outflow) / zone.capacity
                    for zone in current.zones.values())
            )
            violations += sum(zone.crowd_count > zone.capacity for zone in current.zones.values())
        mean_risk = sum(risks) / horizon
        peak_risk = max(risks)
        mean_delay = sum(delay_proxies) / horizon
        objective = mean_risk + delay_weight * mean_delay + capacity_penalty * violations / horizon
        evaluations.append(
            InterventionEvaluation(
                action.name, True, mean_risk, peak_risk, mean_delay,
                violations, objective, reason
            )
        )
    return evaluations


def optimize_intervention(state, actions=None, horizon: int = 5, **weights):
    evaluations = evaluate_interventions(state, actions, horizon, **weights)
    feasible = [item for item in evaluations if item.feasible]
    if not feasible:
        raise RuntimeError("no feasible intervention, including no-action baseline")
    return min(feasible, key=lambda item: (item.objective, item.action)), evaluations
