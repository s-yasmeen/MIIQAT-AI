"""Leakage-aware conversion of ordered zone states into graph-temporal examples."""
from dataclasses import dataclass
import math
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
    group_id: str | None = None


def zone_features(state, zone_id: str) -> List[float]:
    """Build features; capacity_utilization is crowd_count / capacity (not density per area)."""
    zone = state.zones[zone_id]
    reliability = min(zone.camera_confidence, zone.gps_confidence, zone.gate_confidence)
    capacity = float(zone.capacity)
    values = [zone.crowd_count / capacity, zone.inflow / capacity,
              zone.outflow / capacity, zone.mean_speed, zone.bus_queue / capacity,
              reliability, state.heat_index]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"non-finite feature in zone {zone_id}")
    return values


def build_sequences(states: Sequence, history: int = 12, horizons: Sequence[int] = (1, 3, 6)):
    """Build past-only input windows and future occupancy-ratio targets.

    Horizons are sampled state intervals, not minutes. Convert intervals to minutes
    only when the source sampling cadence is known and documented. A state's optional
    scenario_id or video_id is propagated as group_id for group-held-out evaluation.
    """
    if history <= 0 or not horizons or any(not isinstance(h, int) or h <= 0 for h in horizons):
        raise ValueError("history and all forecast horizons must be positive integers")
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
            horizon: [ordered[end + horizon].zones[zone_id].crowd_count /
                      ordered[end + horizon].zones[zone_id].capacity for zone_id in zone_ids]
            for horizon in horizons
        }
        if not all(math.isfinite(value) for values in targets.values() for value in values):
            raise ValueError("non-finite forecast target")
        state = ordered[end]
        group_id = getattr(state, "scenario_id", None) or getattr(state, "video_id", None)
        examples.append(GraphSequence(
            state.timestamp, zone_ids, window, targets,
            None if group_id is None else str(group_id),
        ))
    return examples


def chronological_split(examples: Sequence[GraphSequence], train_fraction: float = 0.70, purge_gap=None):
    """Chronological split with a purge gap to reduce overlapping target leakage."""
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
    if not isinstance(gap, int) or gap < 0:
        raise ValueError("purge_gap must be a non-negative integer")
    train_end = split_at - gap
    if train_end <= 0 or split_at >= len(examples):
        raise ValueError("not enough examples after applying chronological purge gap")
    return list(examples[:train_end]), list(examples[split_at:])


def group_holdout_split(examples: Sequence[GraphSequence], test_fraction: float = 0.20, group_ids=None):
    """Split by whole scenario/video/site groups; no group may appear on both sides.

    Pass group_ids aligned with examples when metadata is external. Otherwise each
    example must carry group_id. Groups are assigned deterministically by sorted ID;
    use a separately pre-registered assignment for final experiments.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if not examples:
        raise ValueError("examples cannot be empty")
    groups = list(group_ids) if group_ids is not None else [item.group_id for item in examples]
    if len(groups) != len(examples) or any(group is None for group in groups):
        raise ValueError("one non-null group ID is required per example")
    unique = sorted({str(group) for group in groups})
    if len(unique) < 2:
        raise ValueError("at least two distinct groups are required")
    test_count = max(1, min(len(unique) - 1, math.ceil(len(unique) * test_fraction)))
    test_groups = set(unique[-test_count:])
    train = [item for item, group in zip(examples, groups) if str(group) not in test_groups]
    test = [item for item, group in zip(examples, groups) if str(group) in test_groups]
    if not train or not test:
        raise ValueError("group split produced an empty partition")
    return train, test


def fit_feature_normalizer(training_examples: Sequence[GraphSequence]):
    """Fit per-feature mean/std from training windows only.

    Returns (means, scales). Never pass validation, calibration, or test examples
    to this function; reuse the returned values unchanged for every held-out split.
    """
    if not training_examples:
        raise ValueError("training_examples cannot be empty")
    width = len(FEATURE_NAMES)
    columns = [[] for _ in range(width)]
    for example in training_examples:
        for timestep in example.features:
            for node_features in timestep:
                if len(node_features) != width:
                    raise ValueError("feature width does not match FEATURE_NAMES")
                for index, value in enumerate(node_features):
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError("training features must be finite")
                    columns[index].append(value)
    means, scales = [], []
    for values in columns:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 1e-12 else 1.0)
    return means, scales


def normalize_feature_window(window, normalizer):
    """Normalize one [time, nodes, features] window using training-only statistics."""
    means, scales = normalizer
    if len(means) != len(FEATURE_NAMES) or len(scales) != len(FEATURE_NAMES):
        raise ValueError("normalizer width does not match FEATURE_NAMES")
    result = []
    for timestep in window:
        normalized_timestep = []
        for node_features in timestep:
            if len(node_features) != len(FEATURE_NAMES):
                raise ValueError("feature width does not match FEATURE_NAMES")
            values = []
            for value, mean, scale in zip(node_features, means, scales):
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("features to normalize must be finite")
                values.append((value - mean) / scale)
            normalized_timestep.append(values)
        result.append(normalized_timestep)
    return result
