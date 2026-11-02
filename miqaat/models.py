from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ZoneState:
    zone_id: str
    capacity: int
    crowd_count: int
    inflow: float
    outflow: float
    mean_speed: float
    dominant_direction: str
    bus_queue: int
    emergency_route_open: bool
    camera_confidence: float = 1.0
    gps_confidence: float = 1.0
    gate_confidence: float = 1.0
    shuttle_inflow: float = 0.0

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.crowd_count < 0 or self.bus_queue < 0:
            raise ValueError("crowd_count and bus_queue cannot be negative")
        for name in ("inflow", "outflow", "mean_speed", "shuttle_inflow"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("camera_confidence", "gps_confidence", "gate_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def density(self) -> float:
        """Capacity utilization; values above 1 indicate a capacity exceedance."""
        return self.crowd_count / self.capacity


@dataclass
class NetworkState:
    timestamp: int
    zones: Dict[str, ZoneState]
    edges: Dict[str, List[str]]
    heat_index: float = 0.65
    # Optional directed edge weights. Missing weights are treated as equal shares.
    edge_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.heat_index <= 1.0:
            raise ValueError("heat_index must be in [0, 1]")
        if set(self.edges) - set(self.zones):
            raise ValueError("every graph source must be a known zone")
        for source, targets in self.edges.items():
            unknown = set(targets) - set(self.zones)
            if unknown:
                raise ValueError(f"unknown target zone(s) from {source}: {sorted(unknown)}")
            if source in targets:
                raise ValueError("self-loops are not supported in the route graph")
        for source, weights in self.edge_weights.items():
            if source not in self.zones or set(weights) - set(self.edges.get(source, [])):
                raise ValueError("edge_weights must refer to existing directed edges")
            if any(value <= 0 for value in weights.values()):
                raise ValueError("edge weights must be positive")


@dataclass
class Intervention:
    name: str
    description: str
    gate_open: str | None = None
    reroute_from: str | None = None
    reroute_to: str | None = None
    shuttle_factor: float = 1.0
    reroute_fraction: float = 0.25
    gate_capacity_factor: float = 1.15
    shuttle_zone: str = "B"

    def __post_init__(self):
        if not 0.0 <= self.shuttle_factor <= 1.0:
            raise ValueError("shuttle_factor must be in [0, 1]")
        if not 0.0 <= self.reroute_fraction <= 1.0:
            raise ValueError("reroute_fraction must be in [0, 1]")
        if self.gate_capacity_factor < 1.0:
            raise ValueError("gate_capacity_factor must be at least 1")


@dataclass
class RiskResult:
    total_risk: float
    level: str
    warning_minutes: int
    zone_risk: Dict[str, float]
    explanation: str
    confidence: float
    intervention: str = "baseline"


@dataclass
class SimulationResult:
    baseline: RiskResult
    alternatives: List[RiskResult] = field(default_factory=list)
