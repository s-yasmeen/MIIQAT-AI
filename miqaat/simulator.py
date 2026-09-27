import random
from copy import deepcopy
from .models import NetworkState, ZoneState, Intervention


SCENARIOS = ("normal", "surge", "shuttle_delay", "blocked_emergency_route", "sensor_outage", "compound")


class SyntheticWorld:
    """Deterministic synthetic network used for development and benchmark experiments."""

    def __init__(self, seed: int = 2026, scenario: str = "mixed"):
        self.rng = random.Random(seed)
        self.seed = seed
        self.scenario = scenario
        self.t = 0
        self.state = NetworkState(0, {
            "A": ZoneState("A", 900, 420, 45, 48, 1.2, "north", 2, True),
            "B": ZoneState("B", 500, 180, 22, 20, 0.8, "east", 3, True),
            "C": ZoneState("C", 300, 80, 10, 10, 1.0, "south", 0, True),
        }, {"A": ["B"], "B": ["C"], "C": []})

    def _active(self, name: str) -> bool:
        return self.scenario in (name, "compound")

    def step(self) -> NetworkState:
        self.t += 1
        periodic_surge = 3.0 if 18 <= self.t % 45 <= 30 else 1.0
        surge = periodic_surge if self.scenario == "mixed" else (3.0 if self._active("surge") else 1.0)
        delay = self._active("shuttle_delay") or (self.scenario == "mixed" and 22 <= self.t % 50 <= 32)
        blocked = self._active("blocked_emergency_route") or (self.scenario == "mixed" and self.t % 37 in (25, 26, 27))

        for zone in self.state.zones.values():
            zone.camera_confidence = max(.35, min(1, .92 + self.rng.uniform(-.08, .05)))
            zone.gps_confidence = max(.45, min(1, .94 + self.rng.uniform(-.04, .03)))
            zone.gate_confidence = max(.55, min(1, .97 + self.rng.uniform(-.03, .01)))
        a, b, c = self.state.zones["A"], self.state.zones["B"], self.state.zones["C"]
        a.inflow, a.outflow = 45 * surge + self.rng.uniform(-5, 5), 47 + self.rng.uniform(-4, 4)
        b.inflow, b.outflow = 22 * surge + self.rng.uniform(-3, 3), 18 + self.rng.uniform(-3, 3)
        c.inflow, c.outflow = 10 * surge, 10
        if delay:
            b.outflow *= .45
            b.bus_queue += 4
        for zone in (a, b, c):
            zone.crowd_count = max(0, int(zone.crowd_count + zone.inflow - zone.outflow))
        b.bus_queue = max(0, b.bus_queue + int(self.rng.choice((-1, 0, 1))))
        c.emergency_route_open = not blocked
        if self._active("sensor_outage") or (self.scenario == "mixed" and self.t % 41 in (12, 13)):
            b.camera_confidence = .35
            b.gps_confidence = .45
        self.state.timestamp = self.t
        self.state.heat_index = .65 + .05 * self.rng.random()
        return deepcopy(self.state)


def apply_intervention(state: NetworkState, action: Intervention) -> NetworkState:
    """Apply a bounded intervention to the current step's flow state."""
    s = deepcopy(state)
    if action.gate_open and action.gate_open in s.zones:
        zone = s.zones[action.gate_open]
        zone.capacity = int(zone.capacity * 1.15)
        zone.outflow *= 1.15
    if action.reroute_from in s.zones and action.reroute_to in s.zones:
        source, target = s.zones[action.reroute_from], s.zones[action.reroute_to]
        moved = min(source.inflow * .25, max(0.0, target.capacity - target.crowd_count - target.inflow))
        source.inflow -= moved
        target.inflow += moved
    if "B" in s.zones:
        s.zones["B"].inflow *= action.shuttle_factor
        s.zones["B"].bus_queue = int(s.zones["B"].bus_queue * action.shuttle_factor)
        s.zones["B"].outflow = min(s.zones["B"].outflow / max(action.shuttle_factor, .01), s.zones["B"].outflow + 8)
    for zone in s.zones.values():
        zone.crowd_count = max(0, int(zone.crowd_count + zone.inflow - zone.outflow))
    return s


def advance_state(state: NetworkState, action: Intervention) -> NetworkState:
    """Advance one modeled interval under an action, with downstream flow coupling."""
    s = apply_intervention(state, action)
    a, b, c = s.zones["A"], s.zones["B"], s.zones["C"]
    # A share of upstream outflow becomes downstream inflow; this is an explicit
    # stylized assumption, not a calibrated physical-flow model.
    transfer_ab = min(a.outflow * .25, max(0, b.capacity - b.crowd_count) * .10)
    transfer_bc = min(b.outflow * .25, max(0, c.capacity - c.crowd_count) * .10)
    b.crowd_count = max(0, int(b.crowd_count + transfer_ab - transfer_bc))
    c.crowd_count = max(0, int(c.crowd_count + transfer_bc))
    s.timestamp += 1
    return s
