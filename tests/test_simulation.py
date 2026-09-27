import unittest

from miqaat.models import Intervention
from miqaat.risk import score
from miqaat.simulator import SyntheticWorld, advance_state, apply_intervention


class SimulationTests(unittest.TestCase):
    def test_same_seed_reproduces_same_states(self):
        first = SyntheticWorld(seed=2030, scenario="mixed")
        second = SyntheticWorld(seed=2030, scenario="mixed")
        for _ in range(10):
            self.assertEqual(first.step(), second.step())

    def test_scenario_failure_modes_are_injected(self):
        blocked = SyntheticWorld(seed=1, scenario="blocked_emergency_route").step()
        outage = SyntheticWorld(seed=1, scenario="sensor_outage").step()
        self.assertFalse(blocked.zones["C"].emergency_route_open)
        self.assertEqual(outage.zones["B"].camera_confidence, .35)
        self.assertEqual(outage.zones["B"].gps_confidence, .45)

    def test_what_if_action_does_not_mutate_observation(self):
        world = SyntheticWorld(seed=2026, scenario="surge")
        state = world.step()
        original = state.zones["B"].capacity
        changed = apply_intervention(state, Intervention("open", "Open B", gate_open="B"))
        self.assertEqual(state.zones["B"].capacity, original)
        self.assertGreater(changed.zones["B"].capacity, original)

    def test_rollout_advances_state_and_risk_is_bounded(self):
        state = SyntheticWorld(seed=2026).step()
        next_state = advance_state(state, Intervention("none", "Do nothing"))
        self.assertEqual(next_state.timestamp, state.timestamp + 1)
        result = score(next_state)
        self.assertGreaterEqual(result.total_risk, 0)
        self.assertLessEqual(result.total_risk, 1)
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)


if __name__ == "__main__":
    unittest.main()
