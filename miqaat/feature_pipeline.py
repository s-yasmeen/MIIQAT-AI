"""Leakage-aware conversion of ordered zone states into graph-temporal examples."""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

FEATURE_NAMES = (
    "capacity_utilization", "inflow_per_capacity", "outflow_per_capacity",
    "mean_speed", "bus_queue_per_capacity", "sensor_reliability", "heat_index",
)


@dataclass
class GraphSequence:
    timestamp: int
    zone_ids: Tuple[str, ...]
    features: List[List[List[float]]]
    targets: Dict[int, List[float]]


def zone_features(state, zone_id: str) -> List[float]:
    zone = state.zones[zone_id]
    reliability = min(zone.camera_confidence, zone.gps_confidence, zone.gate_confidence)
    capacity = float(zone.capacity)
    return [zone.density, zone.inflow / capacity, zone.outflow / capacity,
            zone.mean_speed, zone.bus_queue / capacity, reliability, state.heat_index]


def build_sequences(states: Sequence, history: int = 12, horizons: Sequence[int] = (1, 3, 6)):
    """Build past-only input windows and future occupancy targets.

    Horizons are sampled state intervals, not minutes. Convert intervals to minutes
    only when the source sampling cadence is known and documented.
    """
    if history <= 0 or not horizons or any(h <= 0 for h in horizons):
        raise ValueError("history and all forecast horizons must be positive")
    ordered = sorted(states, key=lambda item: item.timestamp)
    if len({item.timestamp for item in ordered}) != len(ordered):
        raise ValueError("state timestamps must be unique")
    if not ordered:
        return []
    zone_ids = tuple(sorted(ordered[0].zones))
    for state in ordered:
        if tuple(sorted(state.zones)) != zone_ids:
            raise ValueError("all states must contain the same zone set")
    max_horizon = max(horizons)
    examples = []
    for end in range(history - 1, len(ordered) - max_horizon):
        start = end - history + 1
        window = [[zone_features(ordered[t], zone_id) for zone_id in zone_ids]
                  for t in range(start, end + 1)]
        targets = {
            horizon: [ordered[end + horizon].zones[zone_id].density for zone_id in zone_ids]
            for horizon in horizons
        }
        examples.append(GraphSequence(ordered[end].timestamp, zone_ids, window, targets))
    return examples


def chronological_split(examples: Sequence[GraphSequence], train_fraction: float = 0.70, purge_gap=None):
    """Chronological split with a purge gap to prevent overlapping target leakage."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if len(examples) < 3:
        raise ValueError("at least three examples are required")
    timestamps = [item.timestamp for item in examples]
    if timestamps != sorted(timestamps):
        raise ValueError("examples must be sorted chronologically")
    split_at = int(len(examples) * train_fraction)
    max_horizon = max(examples[0].targets)
    gap = max_horizon if purge_gap is None else purge_gap
    if gap < 0:
        raise ValueError("purge_gap cannot be negative")
    train_end = split_at - gap
    if train_end <= 0 or split_at >= len(examples):
        raise ValueError("not enough examples after applying chronological purge gap")
    return list(examples[:train_end]), list(examples[split_at:])
