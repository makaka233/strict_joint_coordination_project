from __future__ import annotations
import unittest

import numpy as np

from src.env.core import build_scenario, generate_macro_obs, generate_scheduler_tasks, init_workload_process


def base_cfg() -> dict:
    return {
        'seed': 13,
        'num_nodes': 4,
        'service_stages': [2, 2, 3],
        'stage_memory_base': 8.0,
        'stage_storage_base': 10.0,
        'node_memory': 48.0,
        'node_storage': 64.0,
        'node_compute': 120.0,
        'bandwidth': 32.0,
        'scheduler_num_tasks': 24,
        'macro_demand_base': 14.0,
        'macro_demand_volatility': 5.0,
        'compute_base': 18.0,
        'data_base': 6.0,
        'max_replicas': 3,
    }


class StochasticUserArrivalEnvTests(unittest.TestCase):
    def test_fixed_window_mode_keeps_task_count_constant(self):
        cfg = base_cfg()
        scn = build_scenario(cfg)
        rng = np.random.default_rng(int(cfg['seed']))
        counts = []
        for _ in range(5):
            macro = generate_macro_obs(scn, cfg, rng)
            tasks = generate_scheduler_tasks(macro, scn, cfg, rng)
            counts.append(len(tasks))
        self.assertEqual(counts, [int(cfg['scheduler_num_tasks'])] * 5)

    def test_user_arrival_mode_varies_task_count_with_bounded_swings(self):
        cfg = base_cfg()
        cfg.update({
            'task_generation_mode': 'user_arrival',
            'arrival_process': 'bernoulli',
            'users_per_node': 32,
            'slots_per_window': 8,
            'window_load_min_factor': 0.88,
            'window_load_max_factor': 1.12,
            'window_load_revert': 0.45,
            'window_load_sigma': 0.02,
            'window_load_wave_amplitude': 0.05,
            'window_load_wave_period': 8,
            'node_load_min_factor': 0.92,
            'node_load_max_factor': 1.10,
            'node_load_revert': 0.50,
            'node_load_sigma': 0.015,
            'node_load_wave_amplitude': 0.04,
            'macro_stage_noise_sigma': 0.04,
            'bernoulli_arrival_prob_cap': 0.10,
        })
        scn = build_scenario(cfg)
        rng = np.random.default_rng(int(cfg['seed']))
        workload_state = init_workload_process(scn, cfg, rng)
        counts = []
        macro_loads = []
        for _ in range(40):
            macro = generate_macro_obs(scn, cfg, rng, workload_state=workload_state)
            tasks = generate_scheduler_tasks(macro, scn, cfg, rng, workload_state=workload_state)
            counts.append(len(tasks))
            service_mass = 0.0
            offset = 0
            for service_stages in scn.service_stages:
                service_mass += float(np.mean(macro[-1, offset:offset + service_stages]))
                offset += service_stages
            macro_loads.append(service_mass)
        self.assertGreater(max(counts) - min(counts), 4)
        self.assertTrue(all(8 <= c <= 48 for c in counts))
        self.assertLess(max(np.abs(np.diff(counts))), 20)
        self.assertGreater(max(macro_loads) - min(macro_loads), 1.0)
        self.assertTrue(18.0 <= float(np.mean(counts)) <= 30.0)


if __name__ == '__main__':
    unittest.main()
