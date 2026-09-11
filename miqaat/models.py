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

    @property
    def density(self) -> float:
        return self.crowd_count / max(1, self.capacity)

@dataclass
class NetworkState:
    timestamp: int
    zones: Dict[str, ZoneState]
    edges: Dict[str, List[str]]
    heat_index: float = 0.65

@dataclass
class Intervention:
    name: str
    description: str
    gate_open: str | None = None
    reroute_from: str | None = None
    reroute_to: str | None = None
    shuttle_factor: float = 1.0

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
