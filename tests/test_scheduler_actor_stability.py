import unittest

import torch

from scripts.train_scheduler_actor import blended_phase_weights, blend_teacher_guidance, guidance_from_costs, scheduler_eval_guard


class SchedulerActorStabilityTests(unittest.TestCase):
    def test_blended_phase_weights_interpolate_near_boundary(self):
        cfg = {
            'phase_a_frac': 0.20,
            'phase_b_frac': 0.50,
            'phase_transition_steps': 10,
            'A_plan_weight': 1.0,
            'A_rank_weight': 0.2,
            'A_relative_weight': 0.4,
            'A_entropy_coef': 0.02,
            'A_entropy_floor': 0.18,
            'A_entropy_floor_coef': 0.10,
            'A_planner_tau': 0.40,
            'A_rank_margin': 0.06,
            'A_gumbel_noise': 0.00,
            'B_plan_weight': 0.6,
            'B_rank_weight': 0.7,
            'B_relative_weight': 1.0,
            'B_entropy_coef': 0.03,
            'B_entropy_floor': 0.20,
            'B_entropy_floor_coef': 0.12,
            'B_planner_tau': 0.30,
            'B_rank_margin': 0.10,
            'B_gumbel_noise': 0.10,
        }
        phase, weights, blend_from, blend_to, alpha = blended_phase_weights(step=15, total_steps=100, cfg=cfg)
        self.assertEqual(phase, 'A')
        self.assertEqual(blend_from, 'A')
        self.assertEqual(blend_to, 'B')
        self.assertAlmostEqual(alpha, 0.5, places=6)
        self.assertAlmostEqual(weights['plan_weight'], 0.8, places=6)
        self.assertAlmostEqual(weights['rank_weight'], 0.45, places=6)
        self.assertAlmostEqual(weights['gumbel_noise'], 0.05, places=6)
        self.assertAlmostEqual(weights['real_teacher_mix'], 0.0, places=6)

    def test_eval_guard_rejects_large_regression_and_triggers_rollback(self):
        cfg = {
            'eval_accept_start_step': 400,
            'accept_short_latency_slack': 6.0,
            'accept_anchor_latency_slack': 6.0,
            'rollback_short_latency_slack': 14.0,
            'rollback_anchor_latency_slack': 12.0,
            'rollback_score_slack': 10.0,
        }
        guard = scheduler_eval_guard(
            eval_lat=649.0,
            anchor_lat=600.0,
            eval_score=627.0,
            best_lat=626.7,
            best_anchor_lat=584.5,
            best_score=607.7,
            cfg=cfg,
            global_step=1000,
        )
        self.assertFalse(guard['accept_checkpoint'])
        self.assertTrue(guard['rollback_triggered'])
        self.assertTrue(guard['severe_short_regression'])
        self.assertTrue(guard['severe_anchor_regression'])
        self.assertTrue(guard['severe_score_regression'])

    def test_eval_guard_accepts_score_improvement_within_latency_slack(self):
        cfg = {
            'eval_accept_start_step': 400,
            'accept_short_latency_slack': 8.0,
            'accept_anchor_latency_slack': 8.0,
            'rollback_short_latency_slack': 14.0,
            'rollback_anchor_latency_slack': 12.0,
            'rollback_score_slack': 10.0,
        }
        guard = scheduler_eval_guard(
            eval_lat=632.0,
            anchor_lat=589.0,
            eval_score=606.0,
            best_lat=626.7,
            best_anchor_lat=584.5,
            best_score=607.7,
            cfg=cfg,
            global_step=700,
        )
        self.assertTrue(guard['accept_checkpoint'])
        self.assertFalse(guard['rollback_triggered'])

    def test_blend_teacher_guidance_boosts_real_mix_on_disagreement(self):
        wm_probs = torch.tensor([[0.80, 0.20], [0.10, 0.90]], dtype=torch.float32)
        real_probs = torch.tensor([[0.30, 0.70], [0.15, 0.85]], dtype=torch.float32)
        wm_best_idx = torch.tensor([0, 1], dtype=torch.long)
        real_best_idx = torch.tensor([1, 1], dtype=torch.long)
        teacher_probs, real_mix, agreement = blend_teacher_guidance(
            wm_probs,
            real_probs,
            wm_best_idx,
            real_best_idx,
            base_mix=0.30,
            disagreement_boost=0.20,
        )
        self.assertAlmostEqual(float(agreement.item()), 0.5, places=6)
        self.assertAlmostEqual(float(real_mix[0].item()), 0.50, places=6)
        self.assertAlmostEqual(float(real_mix[1].item()), 0.30, places=6)
        self.assertGreater(float(teacher_probs[0, 1].item()), float(wm_probs[0, 1].item()))
        self.assertGreater(float(teacher_probs[1, 1].item()), float(teacher_probs[1, 0].item()))

    def test_guidance_from_costs_respects_mask(self):
        costs = torch.tensor([[1.0, 0.5, 0.2]], dtype=torch.float32)
        mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
        probs, best_idx, best_cost, rank_idx, masked_costs = guidance_from_costs(costs, mask, tau=0.3, topk=2)
        self.assertEqual(int(best_idx.item()), 2)
        self.assertAlmostEqual(float(best_cost.item()), 0.2, places=6)
        self.assertEqual(int(rank_idx[0, 0].item()), 2)
        self.assertGreater(float(masked_costs[0, 1].item()), 1e8)
        self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
