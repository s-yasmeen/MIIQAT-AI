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
