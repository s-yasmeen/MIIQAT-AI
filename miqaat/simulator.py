import random
from copy import deepcopy
from .models import NetworkState, ZoneState, Intervention

class SyntheticWorld:
    """Small reproducible world for development before real video/GPS adapters."""
    def __init__(self, seed: int = 2026):
        self.rng = random.Random(seed)
        self.t = 0
        self.state = NetworkState(0, {
            "A": ZoneState("A", 900, 420, 45, 48, 1.2, "north", 2, True),
            "B": ZoneState("B", 500, 180, 22, 20, 0.8, "east", 3, True),
            "C": ZoneState("C", 300, 80, 10, 10, 1.0, "south", 0, True),
        }, {"A": ["B"], "B": ["C"], "C": []})

    def step(self) -> NetworkState:
        self.t += 1
        surge = 3.0 if 18 <= self.t % 45 <= 30 else 1.0
        for zone in self.state.zones.values():
            zone.camera_confidence = max(.35, min(1, .92 + self.rng.uniform(-.08, .05)))
            zone.gps_confidence = max(.45, min(1, .94 + self.rng.uniform(-.04, .03)))
            zone.gate_confidence = max(.55, min(1, .97 + self.rng.uniform(-.03, .01)))
        a, b, c = self.state.zones["A"], self.state.zones["B"], self.state.zones["C"]
        a.inflow, a.outflow = 45 * surge + self.rng.uniform(-5, 5), 47 + self.rng.uniform(-4, 4)
        b.inflow, b.outflow = 22 * surge + self.rng.uniform(-3, 3), 18 + self.rng.uniform(-3, 3)
        c.inflow, c.outflow = 10 * surge, 10
        for zone in (a, b, c):
            zone.crowd_count = max(0, int(zone.crowd_count + zone.inflow - zone.outflow))
        b.bus_queue = int(2 + 3 * surge + self.rng.random() * 2)
        c.emergency_route_open = self.t % 37 not in (25, 26, 27)
        self.state.timestamp = self.t
        self.state.heat_index = .65 + .05 * self.rng.random()
        return deepcopy(self.state)

def apply_intervention(state: NetworkState, action: Intervention) -> NetworkState:
    s = deepcopy(state)
    if action.gate_open and action.gate_open in s.zones:
        s.zones[action.gate_open].capacity = int(s.zones[action.gate_open].capacity * 1.15)
    if action.reroute_from in s.zones and action.reroute_to in s.zones:
        moved = int(s.zones[action.reroute_from].inflow * .25)
        s.zones[action.reroute_from].inflow -= moved
        s.zones[action.reroute_to].inflow += moved
    if "B" in s.zones:
        s.zones["B"].inflow *= action.shuttle_factor
        s.zones["B"].bus_queue = int(s.zones["B"].bus_queue * action.shuttle_factor)
    return s
