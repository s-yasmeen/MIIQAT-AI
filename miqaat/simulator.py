"""Stylized, deterministic zone-flow simulator; not a calibrated pedestrian model."""
import random
from copy import deepcopy

from .graph_engine import validate_graph
from .models import Intervention, NetworkState, ZoneState


SCENARIOS = (
    "normal",
    "surge",
    "shuttle_delay",
    "blocked_emergency_route",
    "sensor_outage",
    "compound",
)
NO_ACTION = Intervention("no_action", "No intervention")


class SyntheticWorld:
    """Synthetic three-zone development world with reproducible scenario injection."""

    def __init__(self, seed: int = 2026, scenario: str = "mixed"):
        if scenario not in (*SCENARIOS, "mixed"):
            raise ValueError(f"unknown scenario: {scenario}")
        self.rng = random.Random(seed)
        self.seed = seed
        self.scenario = scenario
        self.t = 0
        self.state = NetworkState(
            0,
            {
                "A": ZoneState("A", 900, 420, 45, 48, 1.2, "north", 2, True),
                "B": ZoneState("B", 500, 180, 22, 20, 0.8, "east", 3, True, shuttle_inflow=4.0),
                "C": ZoneState("C", 300, 80, 10, 10, 1.0, "south", 0, True),
            },
            {"A": ["B"], "B": ["C"], "C": []},
        )
        validate_graph(self.state.zones, self.state.edges)

    def _active(self, name: str) -> bool:
        return self.scenario in (name, "compound")

    def step(self) -> NetworkState:
        self.t += 1
        periodic_surge = 3.0 if 18 <= self.t % 45 <= 30 else 1.0
        surge = periodic_surge if self.scenario == "mixed" else (3.0 if self._active("surge") else 1.0)
        delay = self._active("shuttle_delay") or (
            self.scenario == "mixed" and 22 <= self.t % 50 <= 32
        )
        blocked = self._active("blocked_emergency_route") or (
            self.scenario == "mixed" and self.t % 37 in (25, 26, 27)
        )

        for zone in self.state.zones.values():
            zone.camera_confidence = max(0.35, min(1, 0.92 + self.rng.uniform(-0.08, 0.05)))
            zone.gps_confidence = max(0.45, min(1, 0.94 + self.rng.uniform(-0.04, 0.03)))
            zone.gate_confidence = max(0.55, min(1, 0.97 + self.rng.uniform(-0.03, 0.01)))

        a, b, c = self.state.zones["A"], self.state.zones["B"], self.state.zones["C"]
        a.inflow, a.outflow = 45 * surge + self.rng.uniform(-5, 5), 47 + self.rng.uniform(-4, 4)
        b.inflow, b.outflow = 22 * surge + self.rng.uniform(-3, 3), 18 + self.rng.uniform(-3, 3)
        c.inflow, c.outflow = 10 * surge, 10
        b.shuttle_inflow = min(b.inflow, 4.0 * surge)
        if delay:
            b.outflow *= 0.45
            b.bus_queue += 4
        b.bus_queue = max(0, b.bus_queue + int(self.rng.choice((-1, 0, 1))))
        c.emergency_route_open = not blocked
        if self._active("sensor_outage") or (
            self.scenario == "mixed" and self.t % 41 in (12, 13)
        ):
            b.camera_confidence = 0.35
            b.gps_confidence = 0.45
        self.state.heat_index = 0.65 + 0.05 * self.rng.random()
        self.state = advance_state(self.state, NO_ACTION)
        return deepcopy(self.state)


def apply_intervention(state: NetworkState, action: Intervention) -> NetworkState:
    """Modify flow controls only; population advances exactly once in advance_state()."""
    s = deepcopy(state)

    if action.gate_open:
        if action.gate_open not in s.zones:
            raise ValueError(f"unknown gate zone: {action.gate_open}")
        zone = s.zones[action.gate_open]
        zone.capacity = int(round(zone.capacity * action.gate_capacity_factor))
        zone.outflow *= action.gate_capacity_factor

    if action.reroute_from or action.reroute_to:
        if not action.reroute_from or not action.reroute_to:
            raise ValueError("rerouting requires both source and target zones")
        if action.reroute_from not in s.zones or action.reroute_to not in s.zones:
            raise ValueError("rerouting references an unknown zone")
        if action.reroute_from == action.reroute_to:
            raise ValueError("reroute source and target must differ")
        source, target = s.zones[action.reroute_from], s.zones[action.reroute_to]
        requested = source.inflow * action.reroute_fraction
        available = max(0.0, target.capacity - target.crowd_count - target.inflow)
        moved = min(requested, available)
        source.inflow -= moved
        target.inflow += moved

    if action.shuttle_factor != 1.0:
        if action.shuttle_zone not in s.zones:
            raise ValueError(f"unknown shuttle zone: {action.shuttle_zone}")
        zone = s.zones[action.shuttle_zone]
        reduced_shuttle_flow = zone.shuttle_inflow * (1.0 - action.shuttle_factor)
        zone.inflow = max(0.0, zone.inflow - reduced_shuttle_flow)
        zone.shuttle_inflow *= action.shuttle_factor
        zone.bus_queue = int(round(zone.bus_queue * action.shuttle_factor))

    return s


def advance_state(
    state: NetworkState,
    action: Intervention = NO_ACTION,
    transfer_fraction: float = 0.25,
) -> NetworkState:
    """Advance one interval with simultaneous, capacity-limited inter-zone transfers.

    A fraction of each zone's outflow is treated as potentially entering connected
    downstream zones. Unaccepted transfer remains in the source zone. The fraction
    is a synthetic assumption and must be calibrated before operational interpretation.
    """
    if not 0.0 <= transfer_fraction <= 1.0:
        raise ValueError("transfer_fraction must be in [0, 1]")
    s = apply_intervention(state, action)
    validate_graph(s.zones, s.edges)

    provisional = {
        zone_id: max(0.0, zone.crowd_count + zone.inflow - zone.outflow)
        for zone_id, zone in s.zones.items()
    }
    planned = {}
    incoming_total = {zone_id: 0.0 for zone_id in s.zones}
    for source_id, source in s.zones.items():
        targets = s.edges.get(source_id, [])
        if not targets:
            continue
        amount = source.outflow * transfer_fraction
        weights = s.edge_weights.get(source_id, {})
        denominator = sum(weights.get(target, 1.0) for target in targets)
        for target in targets:
            share = weights.get(target, 1.0) / denominator
            flow = amount * share
            planned[(source_id, target)] = flow
            incoming_total[target] += flow

    accepted = {}
    for target_id, total in incoming_total.items():
        target = s.zones[target_id]
        room = max(0.0, target.capacity - provisional[target_id])
        scale = min(1.0, room / total) if total > 0 else 1.0
        for (source_id, edge_target), flow in planned.items():
            if edge_target == target_id:
                accepted[(source_id, target_id)] = flow * scale

    blocked_by_source = {zone_id: 0.0 for zone_id in s.zones}
    incoming_by_target = {zone_id: 0.0 for zone_id in s.zones}
    for edge, flow in planned.items():
        source_id, target_id = edge
        admitted = accepted.get(edge, 0.0)
        incoming_by_target[target_id] += admitted
        blocked_by_source[source_id] += flow - admitted

    for zone_id, zone in s.zones.items():
        zone.crowd_count = max(
            0,
            int(round(provisional[zone_id] + incoming_by_target[zone_id] + blocked_by_source[zone_id])),
        )
    s.timestamp += 1
    return s
