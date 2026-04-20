from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse, csv
import numpy as np

from src.common.config import load_yaml
from src.common.seed import set_seed
from src.common.device import resolve_device, describe_device
from src.common.plotter import plot_lines
from src.env.core import build_scenario, generate_macro_obs, flatten_macro_obs, stage_count, greedy_direct_deployment, evaluate_deployment_with_scheduler, make_scheduler_obs
from scripts._shared import load_scheduler_policy, load_deployment_policy


def quantile(vals, q):
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-config', required=True)
    ap.add_argument('--scheduler-checkpoint', required=True)
    ap.add_argument('--deployment-checkpoint', required=True)
    ap.add_argument('--episodes', type=int, default=64)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--gpu-id', type=int, default=2)
    args = ap.parse_args()
    env_cfg = load_yaml(args.env_config)
    set_seed(int(env_cfg['seed']))
    device = resolve_device(args.device, args.gpu_id)
    print({'stage': 'joint_eval', 'device': describe_device(device)})
    scn = build_scenario(env_cfg)
    rng = np.random.default_rng(int(env_cfg['seed']) + 7777)
    dummy_macro = generate_macro_obs(scn, env_cfg, rng)
    dummy_dep = greedy_direct_deployment(dummy_macro, scn, int(env_cfg['max_replicas']), env_cfg, rng)
    t = {'origin': 0, 'service': 0, 'stage_compute': [1.0] * scn.service_stages[0], 'stage_data': [1.0] * scn.service_stages[0]}
    sched_obs_dim = make_scheduler_obs(t, 0, 0, dummy_dep, dummy_macro, scn, np.zeros(scn.num_nodes, dtype=np.float32)).shape[0]
    dep_obs_dim = flatten_macro_obs(dummy_macro).shape[0]
    dep_num_outputs = stage_count(scn) * scn.num_nodes
    scheduler = load_scheduler_policy(args.scheduler_checkpoint, sched_obs_dim, scn.num_nodes, device=device)
    deployer = load_deployment_policy(args.deployment_checkpoint, dep_obs_dim, dep_num_outputs, device=device)
    rows = []
    prev_dep = dummy_dep.copy()
    for ep in range(args.episodes):
        macro = generate_macro_obs(scn, env_cfg, rng)
        dep = deployer.act(flatten_macro_obs(macro), scn, int(env_cfg['max_replicas']))
        out = evaluate_deployment_with_scheduler(macro, dep, scn, env_cfg, lambda obs, mask, task, local_stage, prev: scheduler.act(obs, mask))
        rows.append({
            'episode': ep,
            'total_reward': out['total_reward'],
            'mean_window_latency': out['mean_window_latency'],
            'num_tasks': out['num_tasks'],
            'deployment_density': float(dep.mean()),
            'deployment_switch_proxy': float(np.mean(np.abs(dep - prev_dep))),
        })
        prev_dep = dep
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot_lines(out_path, 'episode', ['mean_window_latency'], 'outputs/figures/stage7_joint_eval_latency.png', 'Stage7 joint eval latency', ma_window=3)
    plot_lines(out_path, 'episode', ['total_reward'], 'outputs/figures/stage7_joint_eval_reward.png', 'Stage7 joint eval reward', ma_window=3)
    lats = [r['mean_window_latency'] for r in rows]
    print({'mean_latency': float(np.mean(lats)), 'p90_latency': quantile(lats, 0.90), 'worst_latency': float(np.max(lats))})


if __name__ == '__main__':
    main()
