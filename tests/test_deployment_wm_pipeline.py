from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts._shared import save_checkpoint
from scripts.train_deployment_actor import compute_wm_aux_loss, load_deployment_wm_model, select_planner_targets
from scripts.train_deployment_wm import build_candidate_conditioned_dataset, split_episode_indices
from src.models.mlp import MLP


class DeploymentWMPipelineTests(unittest.TestCase):
    def test_build_candidate_conditioned_dataset_uses_all_candidates(self):
        replay = [
            {
                'macro_obs': np.array([1.0, 2.0], dtype=np.float32),
                'candidate_xs': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                'candidate_latencies': np.array([4.0, 2.0], dtype=np.float32),
                'candidate_latency_norms': np.log1p(np.array([4.0, 2.0], dtype=np.float32)),
                'best_candidate_idx': 1,
            },
            {
                'macro_obs': np.array([3.0, 4.0], dtype=np.float32),
                'candidate_xs': np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                'candidate_latencies': np.array([5.0], dtype=np.float32),
                'candidate_latency_norms': np.log1p(np.array([5.0], dtype=np.float32)),
                'best_candidate_idx': 0,
            },
        ]
        dataset = build_candidate_conditioned_dataset(replay)
        self.assertEqual(dataset['x'].shape, (3, 5))
        self.assertEqual(dataset['episode_slices'], [(0, 2), (2, 3)])
        self.assertEqual(dataset['best_candidate_idx'], [1, 0])

    def test_split_episode_indices_is_deterministic(self):
        train_a, holdout_a = split_episode_indices(10, 0.2, 123)
        train_b, holdout_b = split_episode_indices(10, 0.2, 123)
        self.assertEqual(train_a, train_b)
        self.assertEqual(holdout_a, holdout_b)
        self.assertEqual(len(train_a) + len(holdout_a), 10)
        self.assertGreaterEqual(len(holdout_a), 1)

    def test_load_deployment_wm_model_uses_checkpoint_meta(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'wm.pt'
            model = MLP(5, 1, hidden=7)
            save_checkpoint(model, path, {'obs_dim': 2, 'num_outputs': 3, 'target': 'log1p_latency'})
            loaded, meta = load_deployment_wm_model(path, torch.device('cpu'))
            out = loaded(torch.zeros(1, 5))
            self.assertEqual(tuple(out.shape), (1, 1))
            self.assertEqual(meta['obs_dim'], 2)
            self.assertEqual(meta['num_outputs'], 3)
            self.assertEqual(meta['target'], 'log1p_latency')

    def test_select_planner_targets_prefers_lowest_predicted_latency(self):
        candidates = np.array([
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ], dtype=np.float32)
        predicted = np.array([
            [8.0, 5.0, 6.0],
            [9.0, 7.0, 4.0],
        ], dtype=np.float32)
        replay_label = np.array([
            [0.0, 1.0],
            [1.0, 0.0],
        ], dtype=np.float32)
        planner_target, stats = select_planner_targets(candidates, predicted, replay_label)
        np.testing.assert_allclose(planner_target[0], np.array([0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(planner_target[1], np.array([0.0, 1.0], dtype=np.float32))
        self.assertAlmostEqual(stats['label_agreement'], 0.5)
        self.assertAlmostEqual(stats['source_actor_rate'], 0.5)

    def test_compute_wm_aux_loss_off_returns_zero(self):
        logits = torch.tensor([[0.3, -0.2]], dtype=torch.float32)
        planner_target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        predicted_gain = torch.tensor([[0.5]], dtype=torch.float32)
        bce = nn.BCEWithLogitsLoss(reduction='none')
        wm_bc_loss, wm_aux_loss, gain_weight = compute_wm_aux_loss(
            logits,
            planner_target,
            predicted_gain,
            bce,
            wm_loss_weight=0.7,
            wm_gain_floor=0.02,
            progress=1.0,
            wm_mode='off',
        )
        self.assertEqual(float(wm_bc_loss.item()), 0.0)
        self.assertEqual(float(wm_aux_loss.item()), 0.0)
        self.assertEqual(float(gain_weight.item()), 0.0)


if __name__ == '__main__':
    unittest.main()
