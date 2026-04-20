from __future__ import annotations
import unittest

import numpy as np

from scripts.train_joint_stage7 import (
    compute_candidate_objective,
    compute_joint_score,
    select_response_candidate_indices,
)


class JointStage7DualTimescaleTests(unittest.TestCase):
    def test_compute_joint_score_prefers_lower_latency(self):
        cfg = {
            'accept_mean_weight': 1.0,
            'accept_p90_weight': 0.25,
            'accept_worst_weight': 0.05,
            'accept_reward_weight': 0.0,
        }
        baseline = {
            'mean_latency': 100.0,
            'p90_latency': 120.0,
            'worst_latency': 150.0,
            'mean_reward': -100.0,
        }
        improved = {
            'mean_latency': 92.0,
            'p90_latency': 112.0,
            'worst_latency': 142.0,
            'mean_reward': -92.0,
        }
        self.assertLess(compute_joint_score(improved, cfg), compute_joint_score(baseline, cfg))

    def test_select_response_candidate_indices_keeps_current(self):
        frozen_latencies = np.array([9.0, 3.0, 5.0, 4.0], dtype=np.float32)
        selected = select_response_candidate_indices(frozen_latencies, response_topk=3)
        self.assertIn(0, selected)
        self.assertEqual(selected[0], 0)
        self.assertEqual(len(selected), 3)

    def test_candidate_objective_can_favor_response_ready_candidate(self):
        cfg = {
            'deployment_frozen_latency_weight': 0.25,
            'deployment_response_latency_weight': 0.75,
            'deployment_switch_penalty': 0.05,
        }
        stable_candidate = compute_candidate_objective(
            frozen_latency=98.0,
            response_latency=90.0,
            switch_cost=0.10,
            cfg=cfg,
        )
        brittle_candidate = compute_candidate_objective(
            frozen_latency=95.0,
            response_latency=102.0,
            switch_cost=0.05,
            cfg=cfg,
        )
        self.assertLess(stable_candidate, brittle_candidate)


if __name__ == '__main__':
    unittest.main()
