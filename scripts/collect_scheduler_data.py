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
from src.env.core import (
    build_scenario, generate_macro_obs, generate_scheduler_tasks, init_workload_process, make_scheduler_obs,
    scheduler_action_mask, greedy_scheduler_action, greedy_direct_deployment,
    random_feasible_deployment, mutate_deployment, scheduler_action_costs,
)


def choose_deployment(macro, scn, env_cfg, collect_cfg, rng):
    max_replicas = int(env_cfg['max_replicas'])
    p_h = float(collect_cfg.get('heuristic_prob', 0.40))
    p_r = float(collect_cfg.get('random_prob', 0.30))
    p_m = float(collect_cfg.get('mutate_prob', 0.30))
    probs = np.array([p_h, p_r, p_m], dtype=np.float64)
    probs = probs / probs.sum()
    source = int(rng.choice(np.arange(3), p=probs))
    if source == 0:
        dep = greedy_direct_deployment(macro, scn, max_replicas, env_cfg, rng)
        dep_src = 'heuristic'
    elif source == 1:
        dep = random_feasible_deployment(macro, scn, max_replicas, rng, env_cfg)
        dep_src = 'random'
    else:
        base = greedy_direct_deployment(macro, scn, max_replicas, env_cfg, rng)
        dep = mutate_deployment(base, macro, scn, max_replicas, rng, float(collect_cfg.get('mutation_prob', 0.22)), env_cfg)
        dep_src = 'mutate'
    return dep, dep_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-config', required=True)
    ap.add_argument('--collect-config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    env_cfg = load_yaml(args.env_config)
    collect_cfg = load_yaml(args.collect_config)
    set_seed(int(env_cfg['seed']))
    rng = np.random.default_rng(int(env_cfg['seed']))
    scn = build_scenario(env_cfg)
    workload_state = init_workload_process(scn, env_cfg, rng)
    replay = []
    logger = MetricLogger('outputs/metrics/stage1_scheduler_collect.csv', 'outputs/logs/stage1_scheduler_collect.jsonl')
    step = 0
    valid_action_counts = []
    source_counts = {'heuristic': 0, 'random': 0, 'mutate': 0}
    mean_best_costs, mean_gap_costs = [], []
    for ep in range(int(collect_cfg['episodes'])):
        macro = generate_macro_obs(scn, env_cfg, rng, workload_state=workload_state)
        dep, dep_src = choose_deployment(macro, scn, env_cfg, collect_cfg, rng)
        source_counts[dep_src] += 1
        tasks = generate_scheduler_tasks(macro, scn, env_cfg, rng, workload_state=workload_state)
        est = np.zeros(scn.num_nodes, dtype=np.float32)
        for t in tasks:
            prev = t['origin']
            for local_stage in range(scn.service_stages[t['service']]):
                obs = make_scheduler_obs(t, local_stage, prev, dep, macro, scn, est)
                mask = scheduler_action_mask(t, local_stage, dep, scn)
                costs = scheduler_action_costs(mask, t, local_stage, prev, scn, est)
                action = greedy_scheduler_action(obs, mask, t, local_stage, prev, scn, est)
                valid = np.where(np.isfinite(costs))[0]
                valid_count = int(len(valid))
                valid_action_counts.append(valid_count)
                best_cost = float(costs[valid].min()) if valid_count else 0.0
                second_cost = float(np.partition(costs[valid], 1)[1]) if valid_count > 1 else best_cost
                gap = float(second_cost - best_cost)
                log_costs = np.zeros_like(costs, dtype=np.float32)
                norm_costs = np.zeros_like(costs, dtype=np.float32)
                if valid_count:
                    log_costs[valid] = np.log1p(costs[valid]).astype(np.float32)
                    cmin = float(costs[valid].min())
                    cmax = float(costs[valid].max())
                    denom = max(cmax - cmin, 1e-6)
                    norm_costs[valid] = ((costs[valid] - cmin) / denom).astype(np.float32)
                replay.append({
                    'obs': obs.astype(np.float32),
                    'mask': mask.astype(np.float32),
                    'action': int(action),
                    'action_costs': costs.astype(np.float32),
                    'action_log_costs': log_costs.astype(np.float32),
                    'action_costs_norm': norm_costs.astype(np.float32),
                    'best_cost': float(best_cost),
                    'gap_cost': float(gap),
                    'valid_action_count': valid_count,
                    'deployment_source': dep_src,
                })
                est[action] += 1.0
                prev = action
                step += 1
                mean_best_costs.append(best_cost)
                mean_gap_costs.append(gap)
                if step % int(collect_cfg.get('log_every_steps', 200)) == 0:
                    arr = np.array(valid_action_counts[-200:], dtype=np.float32) if valid_action_counts else np.zeros(1, dtype=np.float32)
                    logger.log({
                        'global_step': step,
                        'episode': ep,
                        'mean_est_load': float(est.mean()),
                        'mean_best_cost': float(np.mean(mean_best_costs[-200:])),
                        'mean_gap_cost': float(np.mean(mean_gap_costs[-200:])),
                        'valid_action_count_mean': float(arr.mean()),
                        'single_action_ratio': float(np.mean(arr <= 1.0)),
                        'two_action_ratio': float(np.mean(arr == 2.0)),
                        'three_plus_action_ratio': float(np.mean(arr >= 3.0)),
                        'heuristic_ratio': source_counts['heuristic'] / max(ep + 1, 1),
                        'random_ratio': source_counts['random'] / max(ep + 1, 1),
                        'mutate_ratio': source_counts['mutate'] / max(ep + 1, 1),
                    })
        if (ep + 1) % 50 == 0:
            arr = np.array(valid_action_counts, dtype=np.float32) if valid_action_counts else np.zeros(1, dtype=np.float32)
            logger.log({
                'global_step': step,
                'episode': ep + 1,
                'replay_size': len(replay),
                'valid_action_count_mean': float(arr.mean()),
                'valid_action_count_p50': float(np.percentile(arr, 50)),
                'valid_action_count_p90': float(np.percentile(arr, 90)),
                'single_action_ratio': float(np.mean(arr <= 1.0)),
                'two_action_ratio': float(np.mean(arr == 2.0)),
                'three_plus_action_ratio': float(np.mean(arr >= 3.0)),
                'heuristic_ratio': source_counts['heuristic'] / (ep + 1),
                'random_ratio': source_counts['random'] / (ep + 1),
                'mutate_ratio': source_counts['mutate'] / (ep + 1),
                'mean_best_cost_window': float(np.mean(mean_best_costs[-1000:])) if mean_best_costs else 0.0,
                'mean_gap_cost_window': float(np.mean(mean_gap_costs[-1000:])) if mean_gap_costs else 0.0,
            })
            print(f'[collect-scheduler] episode={ep+1} replay={len(replay)} mean_valid={arr.mean():.2f} src(h/r/m)=({source_counts["heuristic"]}/{source_counts["random"]}/{source_counts["mutate"]})')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(replay, args.out)
    logger.close()
    print(f'Saved scheduler replay to {args.out} with {collect_cfg["episodes"]} episodes.')

if __name__ == '__main__':
    main()
