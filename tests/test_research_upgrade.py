import unittest

from miqaat.explainability import additive_attributions
from miqaat.feature_pipeline import build_sequences, chronological_split
from miqaat.forecasting.baselines import linear_trend_forecast, moving_average_forecast, persistence_forecast
from miqaat.graph_engine import normalized_adjacency, propagate_max_risk, topological_order
from miqaat.intervention_optimizer import optimize_intervention
from miqaat.models import Intervention, NetworkState, ZoneState
from miqaat.simulator import SyntheticWorld, advance_state, apply_intervention
from miqaat.uncertainty import conformal_radius, prediction_interval


class ResearchUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.world = SyntheticWorld(seed=12, scenario="normal")
        self.state = self.world.step()

    def test_unrelated_intervention_has_no_shuttle_side_effect(self):
        before = self.state.zones["B"]
        changed = apply_intervention(
            self.state, Intervention("open_a", "Open A", gate_open="A")
        )
        after = changed.zones["B"]
        self.assertEqual(after.inflow, before.inflow)
        self.assertEqual(after.outflow, before.outflow)
        self.assertEqual(after.bus_queue, before.bus_queue)

    def test_shuttle_action_changes_only_tracked_shuttle_component(self):
        changed = apply_intervention(
            self.state,
            Intervention("slow", "Reduce shuttles", shuttle_factor=0.5, shuttle_zone="B"),
        )
        self.assertLess(changed.zones["B"].shuttle_inflow, self.state.zones["B"].shuttle_inflow)
        self.assertEqual(changed.zones["A"], self.state.zones["A"])
        self.assertEqual(changed.zones["C"], self.state.zones["C"])

    def test_apply_intervention_does_not_advance_population(self):
        changed = apply_intervention(
            self.state, Intervention("open", "Open B", gate_open="B")
        )
        self.assertEqual(changed.zones["B"].crowd_count, self.state.zones["B"].crowd_count)
        advanced = advance_state(self.state, Intervention("open", "Open B", gate_open="B"))
        self.assertEqual(advanced.timestamp, self.state.timestamp + 1)

    def test_graph_topological_order_and_cycle_detection(self):
        self.assertEqual(topological_order(["A", "B", "C"], {"A": ["B"], "B": ["C"], "C": []}),
                         ["A", "B", "C"])
        with self.assertRaises(ValueError):
            topological_order(["A", "B"], {"A": ["B"], "B": ["A"]})

    def test_risk_propagation_supports_cycles_and_is_bounded(self):
        result = propagate_max_risk({"A": 0.2, "B": 0.4}, {"A": ["B"], "B": ["A"]})
        self.assertTrue(all(0 <= value <= 1 for value in result.values()))

    def test_normalized_adjacency_has_self_loops_and_rows_sum_to_one(self):
        matrix = normalized_adjacency(["A", "B"], {"A": ["B"], "B": []})
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-9 for row in matrix))

    def test_graph_transfer_respects_target_capacity(self):
        state = NetworkState(
            0,
            {
                "A": ZoneState("A", 100, 80, 0, 40, 1, "n", 0, True),
                "B": ZoneState("B", 100, 95, 0, 0, 1, "n", 0, True),
            },
            {"A": ["B"], "B": []},
        )
        next_state = advance_state(state, Intervention("none", "No action"), transfer_fraction=1.0)
        self.assertLessEqual(next_state.zones["B"].crowd_count, 100)
        self.assertGreater(next_state.zones["A"].crowd_count, 40)

    def test_forecast_sequence_uses_future_targets(self):
        states = [self.world.step() for _ in range(20)]
        examples = build_sequences(states, history=4, horizons=(1, 3))
        self.assertTrue(examples)
        first = examples[0]
        self.assertEqual(len(first.features), 4)
        self.assertEqual(set(first.targets), {1, 3})
        train, test = chronological_split(examples, 0.7)
        self.assertLess(train[-1].timestamp, test[0].timestamp)

    def test_baselines_return_requested_horizons(self):
        values = [1, 2, 3]
        self.assertEqual(persistence_forecast(values, (1, 2)), {1: 3.0, 2: 3.0})
        self.assertEqual(moving_average_forecast(values, 2, (1,)), {1: 2.5})
        self.assertAlmostEqual(linear_trend_forecast(values, (1,))[1], 4.0)

    def test_conformal_interval_is_finite_sample_order_statistic(self):
        radius = conformal_radius([1, -2, 3, -4], alpha=0.25)
        self.assertEqual(radius, 4.0)
        self.assertEqual(prediction_interval(10, radius), (6.0, 14.0))

    def test_additive_attributions_sum_to_score_delta(self):
        values = additive_attributions(
            {"density": 0.8, "flow": 0.5},
            {"density": 0.4, "flow": 0.2},
            {"density": 0.3, "flow": 0.1},
        )
        self.assertAlmostEqual(sum(value for _, value in values), 0.22)

    def test_optimizer_returns_a_feasible_candidate(self):
        best, evaluations = optimize_intervention(self.state, horizon=3)
        self.assertTrue(best.feasible)
        self.assertTrue(any(item.action == "no_action" for item in evaluations))


if __name__ == "__main__":
    unittest.main()
