import math

import pytest

from miqaat.models import Intervention
from miqaat.simulator import SyntheticWorld, advance_state
from miqaat.uncertainty import conformal_radius, prediction_interval


def test_persistent_gate_action_does_not_compound_flow_controls():
    state = SyntheticWorld(seed=7, scenario="normal").state
    action = Intervention("open_gate_B", "Open alternate gate", gate_open="B")
    first = advance_state(state, action)
    second = advance_state(first, action)

    # The persistent control is reapplied per interval; underlying scenario
    # capacity/flow inputs are not multiplied again on each rollout step.
    for zone_id in state.zones:
        original = state.zones[zone_id]
        observed = second.zones[zone_id]
        assert observed.capacity == original.capacity
        assert observed.inflow == original.inflow
        assert observed.outflow == original.outflow
        assert observed.shuttle_inflow == original.shuttle_inflow
        assert observed.bus_queue == original.bus_queue


def test_persistent_shuttle_action_does_not_compound_shuttle_flow_or_queue():
    state = SyntheticWorld(seed=7, scenario="normal").state
    action = Intervention(
        "slow_shuttles", "Reduce shuttle inflow", shuttle_factor=0.65, shuttle_zone="B"
    )
    first = advance_state(state, action)
    second = advance_state(first, action)

    assert second.zones["B"].shuttle_inflow == state.zones["B"].shuttle_inflow
    assert second.zones["B"].inflow == state.zones["B"].inflow
    assert second.zones["B"].bus_queue == state.zones["B"].bus_queue


def test_conformal_returns_infinity_when_finite_sample_rank_is_n_plus_one():
    assert math.isinf(conformal_radius([0.1, 0.2], alpha=0.1))


def test_conformal_uses_correct_order_statistic_when_rank_is_available():
    assert conformal_radius([0.3, 0.1, 0.2], alpha=0.25) == pytest.approx(0.3)


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [-float("inf")]])
def test_conformal_rejects_nonfinite_calibration_residuals(values):
    with pytest.raises(ValueError, match="finite"):
        conformal_radius(values)


def test_prediction_interval_rejects_nonfinite_prediction():
    with pytest.raises(ValueError, match="finite"):
        prediction_interval(float("nan"), 0.2)


def test_prediction_interval_rejects_nan_radius():
    with pytest.raises(ValueError, match="NaN"):
        prediction_interval(0.5, float("nan"))


def test_group_holdout_split_never_leaks_group_ids():
    from miqaat.feature_pipeline import GraphSequence, group_holdout_split

    examples = [
        GraphSequence(i, ("A",), [[[float(i)]]], {1: [float(i)]}, group_id=group)
        for i, group in enumerate(["video_a", "video_a", "video_b", "video_b", "video_c", "video_c"])
    ]
    train, test = group_holdout_split(examples, test_fraction=1 / 3)
    train_groups = {item.group_id for item in train}
    test_groups = {item.group_id for item in test}
    assert train_groups.isdisjoint(test_groups)
    assert train and test


def test_feature_pipeline_rejects_nonfinite_features():
    from miqaat.feature_pipeline import zone_features
    from miqaat.models import ZoneState

    state = SyntheticWorld(seed=4, scenario="normal").state
    state.zones["A"].mean_speed = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        zone_features(state, "A")


def test_gate_intervention_does_not_mutate_input_state():
    from miqaat.simulator import apply_intervention

    state = SyntheticWorld(seed=8, scenario="normal").state
    before = (state.zones["B"].capacity, state.zones["B"].outflow)
    apply_intervention(state, Intervention("gate", "Open B", gate_open="B"))
    assert (state.zones["B"].capacity, state.zones["B"].outflow) == before


def test_feature_normalizer_fits_training_examples_only_and_transforms():
    from miqaat.feature_pipeline import (
        GraphSequence, fit_feature_normalizer, normalize_feature_window, FEATURE_NAMES
    )

    first = [1.0] * len(FEATURE_NAMES)
    second = [3.0] * len(FEATURE_NAMES)
    example = GraphSequence(1, ("A",), [[first], [second]], {1: [0.2]})
    means, scales = fit_feature_normalizer([example])
    assert means == pytest.approx([2.0] * len(FEATURE_NAMES))
    transformed = normalize_feature_window([[first]], (means, scales))
    assert transformed[0][0] == pytest.approx([-1.0] * len(FEATURE_NAMES))
