#!/usr/bin/env python3
import math
import unittest

from timing_mapping import (
    arrival_time_from_k,
    dynamic_t_delay_bounds,
    k_from_arrival_time,
    resolve_policy_timing,
)


K_RANGE = (0.001, 0.01)


class TimingMappingTests(unittest.TestCase):
    def test_paper_formula_round_trip(self):
        for x0, vx0, x_hit in [(-0.30, 4.0, 0.79), (-0.18, 5.2, 0.80), (-0.42, 3.1, 0.795)]:
            for k in (K_RANGE[0], 0.0037, 0.0062, K_RANGE[1]):
                arrival = arrival_time_from_k(x0, vx0, x_hit, k)
                self.assertIsNotNone(arrival)
                recovered = k_from_arrival_time(x0, vx0, x_hit, arrival)
                self.assertIsNotNone(recovered)
                self.assertAlmostEqual(recovered, k, places=8)

    def test_dynamic_delay_endpoints_match_k_endpoints(self):
        params = dict(
            x0=-0.30, vx0=4.0, x_hit=0.79,
            state_t0=0.12, predicted_hit_time=0.40,
            k_range=K_RANGE,
        )
        earliest, latest = dynamic_t_delay_bounds(**params)
        k_early, delay_early, scheduled_early = resolve_policy_timing(
            mode="t_delay", timing_value=-1.0, **params)
        k_late, delay_late, scheduled_late = resolve_policy_timing(
            mode="t_delay", timing_value=1.0, **params)
        self.assertAlmostEqual(k_early, K_RANGE[1], places=8)
        self.assertAlmostEqual(k_late, K_RANGE[0], places=8)
        self.assertAlmostEqual(delay_early, earliest, places=12)
        self.assertAlmostEqual(delay_late, latest, places=12)
        self.assertAlmostEqual(scheduled_early, params["predicted_hit_time"] + delay_early, places=12)
        self.assertAlmostEqual(scheduled_late, params["predicted_hit_time"] + delay_late, places=12)

    def test_delay_mode_never_leaves_k_range_or_returns_nan(self):
        cases = [
            (-0.30, 4.0, 0.79, 0.12, 0.40),
            (-0.18, 5.2, 0.80, 0.08, 0.27),
            (-0.42, 3.1, 0.795, 0.16, 0.55),
        ]
        for x0, vx0, x_hit, state_t0, predicted in cases:
            for i in range(101):
                selector = -1.0 + 2.0 * i / 100.0
                k, delay, scheduled = resolve_policy_timing(
                    mode="t_delay", timing_value=selector,
                    x0=x0, vx0=vx0, x_hit=x_hit,
                    state_t0=state_t0, predicted_hit_time=predicted,
                    k_range=K_RANGE,
                )
                self.assertIsNotNone(k)
                self.assertTrue(K_RANGE[0] - 1e-10 <= k <= K_RANGE[1] + 1e-10)
                self.assertTrue(math.isfinite(delay))
                self.assertTrue(math.isfinite(scheduled))
                arrival = arrival_time_from_k(x0, vx0, x_hit, k)
                self.assertAlmostEqual(delay, arrival + state_t0 - predicted, places=10)

    def test_k_mode_keeps_original_equation(self):
        k = 0.006
        params = dict(
            x0=-0.30, vx0=4.0, x_hit=0.79,
            state_t0=0.12, predicted_hit_time=0.40,
            k_range=K_RANGE,
        )
        resolved_k, delay, scheduled = resolve_policy_timing(
            mode="k", timing_value=k, **params)
        arrival = arrival_time_from_k(params["x0"], params["vx0"], params["x_hit"], k)
        self.assertAlmostEqual(resolved_k, k, places=12)
        self.assertAlmostEqual(scheduled, params["state_t0"] + arrival, places=12)
        self.assertAlmostEqual(delay, arrival + params["state_t0"] - params["predicted_hit_time"], places=12)

    def test_invalid_motion_is_rejected_without_nan(self):
        result = resolve_policy_timing(
            mode="t_delay", timing_value=0.0,
            x0=-0.30, vx0=0.0, x_hit=0.79,
            state_t0=0.12, predicted_hit_time=0.40,
            k_range=K_RANGE,
        )
        self.assertEqual(result, (None, None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
