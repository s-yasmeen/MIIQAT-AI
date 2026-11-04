import math
import unittest

from miqaat.feature_pipeline import (
    FEATURE_NAMES, GraphSequence, build_sequences, fit_feature_normalizer,
    group_holdout_split, normalize_feature_window, zone_features,
)
from miqaat.models import Intervention, NetworkState, ZoneState
from miqaat.simulator import SyntheticWorld, advance_state, apply_intervention
from miqaat.uncertainty import conformal_radius, prediction_interval


class ResearchRegressionTests(unittest.TestCase):
    def test_persistent_gate_action_does_not_compound_flow_controls(self):
        state = SyntheticWorld(seed=7, scenario="normal").state
        action = Intervention("open_gate_B", "Open alternate gate", gate_open="B")
        advance_state(state, action)
        second = advance_state(state, action)
        for zone_id in state.zones:
            original, observed = state.zones[zone_id], second.zones[zone_id]
            self.assertEqual(observed.capacity, original.capacity)
            self.assertEqual(observed.inflow, original.inflow)
            self.assertEqual(observed.outflow, original.outflow)
            self.assertEqual(observed.shuttle_inflow, original.shuttle_inflow)
            self.assertEqual(observed.bus_queue, original.bus_queue)

    def test_persistent_shuttle_action_does_not_compound_shuttle_flow_or_queue(self):
        state = SyntheticWorld(seed=7, scenario="normal").state
        action = Intervention(
            "slow_shuttles", "Reduce shuttle inflow", shuttle_factor=0.65, shuttle_zone="B"
        )
        first = advance_state(state, action)
        second = advance_state(first, action)
        self.assertEqual(second.zones["B"].shuttle_inflow, state.zones["B"].shuttle_inflow)
        self.assertEqual(second.zones["B"].inflow, state.zones["B"].inflow)
        self.assertEqual(second.zones["B"].bus_queue, state.zones["B"].bus_queue)

    def test_conformal_returns_infinity_when_finite_sample_rank_is_n_plus_one(self):
        self.assertTrue(math.isinf(conformal_radius([0.1, 0.2], alpha=0.1)))

    def test_conformal_uses_correct_order_statistic_when_rank_is_available(self):
        self.assertAlmostEqual(conformal_radius([0.3, 0.1, 0.2], alpha=0.25), 0.3)

    def test_conformal_rejects_nonfinite_calibration_residuals(self):
        for values in ([float("nan")], [float("inf")], [-float("inf")]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "finite"):
                    conformal_radius(values)

    def test_prediction_interval_rejects_nonfinite_prediction(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            prediction_interval(float("nan"), 0.2)

    def test_prediction_interval_rejects_nan_radius(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            prediction_interval(0.5, float("nan"))

    def test_group_holdout_split_never_leaks_group_ids(self):
        examples = [
            GraphSequence(i, ("A",), [[[float(i)]]], {1: [float(i)]}, group_id=group)
            for i, group in enumerate(["video_a", "video_a", "video_b", "video_b", "video_c", "video_c"])
        ]
        train, test = group_holdout_split(examples, test_fraction=1 / 3)
        train_groups = {item.group_id for item in train}
        test_groups = {item.group_id for item in test}
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(train and test)

    def test_build_sequences_never_crosses_video_boundaries(self):
        states = []
        for video_id, base_count in (("video_a", 10), ("video_b", 70)):
            for timestamp in range(8):
                count = base_count + timestamp
                states.append(NetworkState(
                    timestamp=timestamp,
                    zones={
                        "A": ZoneState(
                            "A", 100, count, 1, 1, 1.0, "east", 0, True
                        ),
                    },
                    edges={"A": []},
                    scenario_id="surge",
                    video_id=video_id,
                ))

        examples = build_sequences(states, history=3, horizons=(1, 2))
        self.assertEqual(len(examples), 8)  # 4 valid windows per video
        groups = {item.group_id for item in examples}
        self.assertEqual(len(groups), 2)

        for example in examples:
            if "video_a" in example.group_id:
                expected_base = 10
            else:
                self.assertIn("video_b", example.group_id)
                expected_base = 70

            # The first feature is occupancy ratio; all input values and future
            # targets must originate from the same video's count sequence.
            input_counts = [
                round(timestep[0][0] * 100)
                for timestep in example.features
            ]
            self.assertEqual(
                input_counts,
                list(range(input_counts[0], input_counts[0] + 3)),
            )
            self.assertGreaterEqual(input_counts[0], expected_base)
            self.assertLess(input_counts[-1], expected_base + 8)
            for horizon in (1, 2):
                target_count = round(example.targets[horizon][0] * 100)
                self.assertGreaterEqual(target_count, expected_base)
                self.assertLess(target_count, expected_base + 8)
                self.assertGreater(target_count, input_counts[-1])

    def test_build_sequences_rejects_duplicate_timestamps_within_group(self):
        state = SyntheticWorld(seed=4, scenario="normal").state
        duplicate = SyntheticWorld(seed=5, scenario="normal").state
        state.video_id = "same_video"
        duplicate.video_id = "same_video"
        duplicate.timestamp = state.timestamp
        with self.assertRaisesRegex(ValueError, "unique within group"):
            build_sequences([state, duplicate], history=1, horizons=(1,))

    def test_feature_pipeline_rejects_nonfinite_features(self):
        state = SyntheticWorld(seed=4, scenario="normal").state
        state.zones["A"].mean_speed = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            zone_features(state, "A")

    def test_gate_intervention_does_not_mutate_input_state(self):
        state = SyntheticWorld(seed=8, scenario="normal").state
        before = (state.zones["B"].capacity, state.zones["B"].outflow)
        apply_intervention(state, Intervention("gate", "Open B", gate_open="B"))
        self.assertEqual((state.zones["B"].capacity, state.zones["B"].outflow), before)

    def test_feature_normalizer_fits_training_examples_and_transforms(self):
        first, second = [1.0] * len(FEATURE_NAMES), [3.0] * len(FEATURE_NAMES)
        example = GraphSequence(1, ("A",), [[first], [second]], {1: [0.2]})
        means, scales = fit_feature_normalizer([example])
        self.assertEqual(means, [2.0] * len(FEATURE_NAMES))
        transformed = normalize_feature_window([[first]], (means, scales))
        for value in transformed[0][0]:
            self.assertAlmostEqual(value, -1.0)

    def test_internal_transfer_and_external_exit_accounting(self):
        state = NetworkState(
            0,
            {
                "A": ZoneState("A", 100, 50, 10, 10, 1.0, "east", 0, True),
                "B": ZoneState("B", 100, 20, 5, 4, 1.0, "east", 0, True),
            },
            {"A": ["B"], "B": []},
        )
        after = advance_state(state, transfer_fraction=1.0)
        self.assertEqual(sum(zone.crowd_count for zone in after.zones.values()), 81)


if __name__ == "__main__":
    unittest.main()
