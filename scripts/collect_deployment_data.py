from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import torch
import numpy as np
from src.common.config import load_yaml
from src.common.seed import set_seed
from src.common.metric_logger import MetricLogger
from src.common.device import resolve_device, describe_device
from src.env.core import build_scenario, generate_macro_obs, greedy_direct_deployment, repair_deployment, evaluate_deployment_with_scheduler, flatten_macro_obs, stage_count
from scripts._shared import load_scheduler_policy


def random_feasible(scn, cfg, rng):
    s_count = stage_count(scn)
    raw = (rng.random((s_count, scn.num_nodes)) < 0.35).astype(np.float32)
    scores = rng.random((s_count, scn.num_nodes)).astype(np.float32)
    return repair_deployment(raw, scn, int(cfg['max_replicas']), scores=scores)


def mutate(x, scn, cfg, rng):
    y = x.copy()
    for _ in range(rng.integers(1, 4)):
        s = int(rng.integers(0, y.shape[0]))
        n = int(rng.integers(0, y.shape[1]))
        y[s, n] = 1.0 - y[s, n]
    return repair_deployment(y, scn, int(cfg['max_replicas']), scores=rng.random(y.shape).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-config', required=True)
    ap.add_argument('--scheduler-checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', type=int, default=1200)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--gpu-id', type=int, default=2)
    args = ap.parse_args()
    env_cfg = load_yaml(args.env_config)
    set_seed(int(env_cfg['seed']))
    rng = np.random.default_rng(int(env_cfg['seed']))
    device = resolve_device(args.device, args.gpu_id)
    print({'stage': 'collect_deployment_data', 'device': describe_device(device)})
    scn = build_scenario(env_cfg)
    dummy_macro = generate_macro_obs(scn, env_cfg, rng)
    dummy_dep = greedy_direct_deployment(dummy_macro, scn, int(env_cfg['max_replicas']))
    from src.env.core import make_scheduler_obs
    t = {'origin': 0, 'service': 0, 'stage_compute': [1.0]*scn.service_stages[0], 'stage_data':[1.0]*scn.service_stages[0]}
    obs_dim = make_scheduler_obs(t, 0, 0, dummy_dep, dummy_macro, scn, np.zeros(scn.num_nodes, dtype=np.float32)).shape[0]
    scheduler = load_scheduler_policy(args.scheduler_checkpoint, obs_dim, scn.num_nodes, device=device)
    replay = []
    log = MetricLogger('outputs/metrics/stage4_deployment_collect.csv', 'outputs/logs/stage4_deployment_collect.jsonl')
    candidate_count = int(load_yaml(PROJECT_ROOT / 'configs/deployment/actor_large.yaml').get('candidate_count', 8)) if (PROJECT_ROOT / 'configs/deployment/actor_large.yaml').exists() else 8
    for ep in range(args.episodes):
        macro = generate_macro_obs(scn, env_cfg, rng)
        current = greedy_direct_deployment(macro, scn, int(env_cfg['max_replicas']))
        candidates = [current]
        # richer candidate pool: heuristic / mutate / random / mutate-random
        heuristic = greedy_direct_deployment(macro, scn, int(env_cfg['max_replicas']))
        candidates.append(heuristic)
        for _ in range(max(1, candidate_count // 4)):
            candidates.append(mutate(current, scn, env_cfg, rng))
            candidates.append(mutate(heuristic, scn, env_cfg, rng))
            rnd = random_feasible(scn, env_cfg, rng)
            candidates.append(rnd)
            candidates.append(mutate(rnd, scn, env_cfg, rng))
        # dedupe and trim
        uniq = []
        seen = set()
        for cand in candidates:
            key = tuple(cand.reshape(-1).astype(int).tolist())
            if key not in seen:
                seen.add(key)
                uniq.append(cand)
        candidates = uniq[:candidate_count]
        vals = []
        rewards = []
        for cand in candidates:
            out = evaluate_deployment_with_scheduler(macro, cand, scn, env_cfg, lambda obs, mask, task, local_stage, prev: scheduler.act(obs, mask))
            vals.append(out['mean_window_latency'])
            rewards.append(out['total_reward'])
        best_idx = int(np.argmin(vals))
        current_latency = float(vals[0])
        best_latency = float(vals[best_idx])
        current_reward = float(rewards[0])
        best_reward = float(rewards[best_idx])
        rel_gain = float((current_latency - best_latency) / max(current_latency, 1e-6))
        replay.append({
            'macro_obs': flatten_macro_obs(macro),
            'x_label': candidates[best_idx].astype(np.float32),
            'current_x': current.astype(np.float32),
            'best_latency': best_latency,
            'best_latency_norm': float(np.log1p(best_latency)),
            'current_latency': current_latency,
            'current_latency_norm': float(np.log1p(current_latency)),
            'best_reward': best_reward,
            'current_reward': current_reward,
            'relative_gain': rel_gain,
            'switch_cost_est': float(np.mean(np.abs(candidates[best_idx] - current))),
            'candidate_count': len(candidates),
        })
        if (ep + 1) % 10 == 0:
            row = {
                'global_step': ep + 1,
                'episode': ep + 1,
                'window_latency_mean': float(np.mean(vals)),
                'best_candidate_latency': best_latency,
                'best_candidate_latency_norm': float(np.log1p(best_latency)),
                'current_latency': current_latency,
                'current_latency_norm': float(np.log1p(current_latency)),
                'relative_gain': rel_gain,
                'switch_cost_est': float(np.mean(np.abs(candidates[best_idx] - current))),
                'candidate_count': len(candidates),
            }
            log.log(row)
            print(f'[collect-deployment] episode={ep+1} best_latency={best_latency:.3f} gain={rel_gain:.4f}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(replay, args.out)
    log.close()
    print(f'Saved deployment replay to {args.out}')

if __name__ == '__main__':
    main()
