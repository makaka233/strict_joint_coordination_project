from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse, math, copy
import numpy as np
import torch
from torch import nn

from src.common.config import load_yaml
from src.common.seed import set_seed
from src.common.metric_logger import MetricLogger
from src.common.plotter import plot_lines
from src.common.device import resolve_device, describe_device
from src.env.core import (
    build_scenario, generate_macro_obs, flatten_macro_obs, stage_count,
    generate_scheduler_tasks, make_scheduler_obs, scheduler_action_mask,
    scheduler_action_costs, scheduler_target_probs,
    greedy_direct_deployment, random_feasible_deployment, mutate_deployment,
    evaluate_deployment_with_scheduler,
)
from src.agents.scheduler.policy import SchedulerPolicy
from src.agents.deployment.policy import DeploymentPolicy
from scripts._shared import load_scheduler_policy, load_deployment_policy, save_checkpoint


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.quantile(arr, q))


def eval_joint(dep_pol, sched_pol, scn, env_cfg, episodes: int, seed_list: list[int]):
    rows = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        for _ in range(episodes):
            macro = generate_macro_obs(scn, env_cfg, rng)
            dep = dep_pol.act(flatten_macro_obs(macro), scn, int(env_cfg['max_replicas']))
            out = evaluate_deployment_with_scheduler(
                macro, dep, scn, env_cfg,
                lambda obs, mask, task, local_stage, prev: sched_pol.act(obs, mask)
            )
            rows.append({
                'latency': float(out['mean_window_latency']),
                'reward': float(out['total_reward']),
                'density': float(dep.mean()),
            })
    lats = [r['latency'] for r in rows]
    rews = [r['reward'] for r in rows]
    return {
        'mean_latency': float(np.mean(lats)) if lats else math.inf,
        'p90_latency': quantile(lats, 0.90),
        'worst_latency': float(np.max(lats)) if lats else math.inf,
        'mean_reward': float(np.mean(rews)) if rews else -math.inf,
        'density': float(np.mean([r['density'] for r in rows])) if rows else 0.0,
        'n': len(rows),
    }


def collect_scheduler_samples(dep_pol, sched_pol, scn, env_cfg, episodes: int, seed: int, temp: float):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(episodes):
        macro = generate_macro_obs(scn, env_cfg, rng)
        dep = dep_pol.act(flatten_macro_obs(macro), scn, int(env_cfg['max_replicas']))
        tasks = generate_scheduler_tasks(macro, scn, env_cfg, rng)
        est_node_loads = np.zeros(scn.num_nodes, dtype=np.float32)
        for task in tasks:
            prev = task['origin']
            for local_stage in range(scn.service_stages[task['service']]):
                obs = make_scheduler_obs(task, local_stage, prev, dep, macro, scn, est_node_loads)
                mask = scheduler_action_mask(task, local_stage, dep, scn)
                if mask.sum() < 1:
                    continue
                act = int(sched_pol.act(obs, mask))
                costs = scheduler_action_costs(mask, task, local_stage, prev, scn, est_node_loads)
                target = scheduler_target_probs(costs, mask, temperature=temp)
                finite = np.isfinite(costs)
                best = float(np.min(costs[finite])) if np.any(finite) else 0.0
                chosen = float(costs[act]) if np.isfinite(costs[act]) else best
                regret = max(0.0, chosen - best)
                samples.append({
                    'obs': obs.astype(np.float32),
                    'mask': mask.astype(np.float32),
                    'costs': np.nan_to_num(costs, nan=1e9, posinf=1e9).astype(np.float32),
                    'target_probs': target.astype(np.float32),
                    'best_cost': best,
                    'chosen_cost': chosen,
                    'regret': regret,
                    'valid_action_count': int(mask.sum()),
                })
                if mask[act] < 0.5:
                    act = int(np.argmax(mask))
                est_node_loads[act] += 1.0
                prev = act
    return samples


def collect_deployment_samples(dep_pol, sched_pol, scn, env_cfg, episodes: int, seed: int, candidate_count: int):
    rng = np.random.default_rng(seed)
    samples = []
    max_replicas = int(env_cfg['max_replicas'])
    for _ in range(episodes):
        macro = generate_macro_obs(scn, env_cfg, rng)
        obs = flatten_macro_obs(macro)
        current = dep_pol.act(obs, scn, max_replicas)
        cands = [current]
        heuristic = greedy_direct_deployment(macro, scn, max_replicas, env_cfg, rng)
        cands.append(heuristic)
        while len(cands) < candidate_count:
            if rng.random() < 0.50:
                base = current if rng.random() < 0.6 else heuristic
                cand = mutate_deployment(base, macro, scn, max_replicas, rng, 0.22, env_cfg)
            else:
                cand = random_feasible_deployment(macro, scn, max_replicas, rng, env_cfg)
            if not any(np.array_equal(cand, x) for x in cands):
                cands.append(cand)
        outs = [evaluate_deployment_with_scheduler(macro, x, scn, env_cfg, lambda o, m, t, ls, p: sched_pol.act(o, m)) for x in cands]
        lats = [o['mean_window_latency'] for o in outs]
        best_i = int(np.argmin(lats))
        best_x = cands[best_i]
        current_lat = float(lats[0])
        best_lat = float(lats[best_i])
        rel_gain = max(0.0, (current_lat - best_lat) / max(current_lat, 1e-6))
        switch = float(np.mean(np.abs(best_x - current)))
        samples.append({
            'macro_obs': obs.astype(np.float32),
            'x_label': best_x.reshape(-1).astype(np.float32),
            'current_x': current.reshape(-1).astype(np.float32),
            'relative_gain': rel_gain,
            'switch_cost_est': switch,
            'current_latency': current_lat,
            'best_candidate_latency': best_lat,
        })
    return samples


def train_scheduler_inner(samples, pol, cfg, device):
    if not samples:
        return {'scheduler_actor_loss': 0.0, 'scheduler_entropy': 0.0, 'scheduler_regret': 0.0}
    obs = torch.from_numpy(np.stack([s['obs'] for s in samples]).astype('float32'))
    mask = torch.from_numpy(np.stack([s['mask'] for s in samples]).astype('float32'))
    target = torch.from_numpy(np.stack([s['target_probs'] for s in samples]).astype('float32'))
    costs = torch.from_numpy(np.stack([s['costs'] for s in samples]).astype('float32'))
    regret = torch.from_numpy(np.array([s['regret'] for s in samples], dtype='float32')).unsqueeze(1)
    pol.model.train()
    opt = torch.optim.Adam(pol.model.parameters(), lr=float(cfg['scheduler_lr']))
    bs = int(cfg['scheduler_batch_size'])
    steps = int(cfg['scheduler_inner_steps'])
    entropy_coef = float(cfg.get('scheduler_entropy_coef', 0.004))
    regret_coef = float(cfg.get('scheduler_regret_coef', 0.6))
    last = {}
    for _ in range(steps):
        idx = torch.randint(0, len(samples), (min(bs, len(samples)),))
        ob = obs[idx].to(device)
        mb = mask[idx].to(device)
        tb = target[idx].to(device)
        cb = costs[idx].to(device)
        rb = regret[idx].to(device)
        logits = pol.model(ob)
        masked = logits + (mb - 1.0) * 1e9
        probs = torch.softmax(masked, dim=1)
        plan_loss = -(tb * torch.log(probs + 1e-8)).sum(dim=1).mean()
        expected_cost = (probs * cb).sum(dim=1, keepdim=True)
        best_cost = torch.min(cb + (1.0 - mb) * 1e9, dim=1, keepdim=True).values
        relative_cost = ((expected_cost - best_cost) * (1.0 + regret_coef * rb)).mean()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        rank_idx = torch.argsort(cb + (1.0 - mb) * 1e9, dim=1)
        best_i = rank_idx[:, 0]
        second_i = rank_idx[:, 1] if cb.shape[1] > 1 else best_i
        best_logit = logits.gather(1, best_i.unsqueeze(1))
        second_logit = logits.gather(1, second_i.unsqueeze(1))
        rank_loss = torch.relu(0.10 - (best_logit - second_logit)).mean()
        loss = plan_loss + relative_cost + 0.4 * rank_loss - entropy_coef * entropy
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.model.parameters(), 1.0); opt.step()
        last = {
            'scheduler_actor_loss': float(loss.item()),
            'scheduler_entropy': float(entropy.item()),
            'scheduler_regret': float(rb.mean().item()),
            'scheduler_plan_loss': float(plan_loss.item()),
            'scheduler_relative_cost': float(relative_cost.item()),
        }
    return last


def train_deployment_inner(samples, dep_pol, cfg, device):
    if not samples:
        return {'deployment_actor_loss': 0.0, 'deployment_bc_loss': 0.0, 'deployment_switch_cost': 0.0}
    obs = torch.from_numpy(np.stack([s['macro_obs'] for s in samples]).astype('float32'))
    xlab = torch.from_numpy(np.stack([s['x_label'] for s in samples]).astype('float32'))
    cur = torch.from_numpy(np.stack([s['current_x'] for s in samples]).astype('float32'))
    gain = torch.tensor([s['relative_gain'] for s in samples], dtype=torch.float32).unsqueeze(1)
    swt = torch.tensor([s['switch_cost_est'] for s in samples], dtype=torch.float32).unsqueeze(1)
    dep_pol.model.train()
    opt = torch.optim.Adam(dep_pol.model.parameters(), lr=float(cfg['deployment_lr']))
    bce = nn.BCEWithLogitsLoss(reduction='none')
    bs = int(cfg['deployment_batch_size'])
    steps = int(cfg['deployment_inner_steps'])
    stay_coef = float(cfg.get('deployment_stay_coef', 0.2))
    switch_coef = float(cfg.get('deployment_switch_coef', 0.05))
    last = {}
    for _ in range(steps):
        idx = torch.randint(0, len(samples), (min(bs, len(samples)),))
        ob = obs[idx].to(device)
        xb = xlab[idx].to(device)
        xc = cur[idx].to(device)
        gb = gain[idx].to(device)
        swb = swt[idx].to(device)
        logits = dep_pol.model(ob)
        probs = torch.sigmoid(logits)
        bc_per = bce(logits, xb).mean(dim=1, keepdim=True)
        bc_loss = ((0.5 + gb) * bc_per).mean()
        stay_loss = (probs - xc).abs().mean()
        pred_switch = (probs - xc).abs().mean(dim=1, keepdim=True)
        loss = bc_loss + stay_coef * (1.0 - gb.mean()) * stay_loss + switch_coef * pred_switch.mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(dep_pol.model.parameters(), 2.0); opt.step()
        last = {
            'deployment_actor_loss': float(loss.item()),
            'deployment_bc_loss': float(bc_loss.item()),
            'deployment_switch_cost': float(pred_switch.mean().item()),
            'deployment_gain': float(gb.mean().item()),
            'deployment_target_switch': float(swb.mean().item()),
        }
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-config', required=True)
    ap.add_argument('--joint-config', required=True)
    ap.add_argument('--scheduler-checkpoint', required=True)
    ap.add_argument('--deployment-checkpoint', required=True)
    ap.add_argument('--out-dir', default='outputs/joint_stage7')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--gpu-id', type=int, default=2)
    args = ap.parse_args()

    env_cfg = load_yaml(args.env_config)
    cfg = load_yaml(args.joint_config)
    set_seed(int(env_cfg['seed']))
    torch.set_num_threads(1)
    device = resolve_device(args.device, args.gpu_id)
    print({'stage': 'joint_stage7', 'device': describe_device(device)})
    scn = build_scenario(env_cfg)
    rng = np.random.default_rng(int(env_cfg['seed']) + 17007)
    dummy_macro = generate_macro_obs(scn, env_cfg, rng)
    dummy_dep = greedy_direct_deployment(dummy_macro, scn, int(env_cfg['max_replicas']), env_cfg, rng)
    t = {'origin': 0, 'service': 0, 'stage_compute': [1.0] * scn.service_stages[0], 'stage_data': [1.0] * scn.service_stages[0]}
    from src.env.core import make_scheduler_obs
    sched_obs_dim = make_scheduler_obs(t, 0, 0, dummy_dep, dummy_macro, scn, np.zeros(scn.num_nodes, dtype=np.float32)).shape[0]
    dep_obs_dim = flatten_macro_obs(dummy_macro).shape[0]
    dep_num_outputs = stage_count(scn) * scn.num_nodes

    sched_pol = load_scheduler_policy(args.scheduler_checkpoint, sched_obs_dim, scn.num_nodes, device=device)
    dep_pol = load_deployment_policy(args.deployment_checkpoint, dep_obs_dim, dep_num_outputs, device=device)
    best_sched_state = copy.deepcopy(sched_pol.model.state_dict())
    best_dep_state = copy.deepcopy(dep_pol.model.state_dict())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(out_dir / 'stage7_joint_train.csv', out_dir / 'stage7_joint_train.jsonl')

    seed_base = int(cfg.get('fixed_seed_base', int(env_cfg['seed']) + 50000))
    seed_count = int(cfg.get('fixed_seed_count', 10))
    seed_bank = [seed_base + 101 * i for i in range(seed_count)]
    eval_eps = int(cfg.get('eval_episodes', 20))
    def score(ev):
        return float(cfg.get('accept_mean_weight', 1.0)) * ev['mean_latency'] + float(cfg.get('accept_p90_weight', 0.25)) * ev['p90_latency'] + float(cfg.get('accept_worst_weight', 0.05)) * ev['worst_latency']

    best_eval = eval_joint(dep_pol, sched_pol, scn, env_cfg, eval_eps, seed_bank)
    best_score = score(best_eval)
    logger.log({'cycle': 0, 'phase': 'baseline', **best_eval, 'joint_score': best_score, 'accepted_update': 1.0})
    save_checkpoint(sched_pol.model, out_dir / 'scheduler_joint_best.pt', {'joint_score': best_score, 'obs_dim': sched_obs_dim, 'num_nodes': scn.num_nodes})
    save_checkpoint(dep_pol.model, out_dir / 'deployment_joint_best.pt', {'joint_score': best_score, 'obs_dim': dep_obs_dim, 'num_outputs': dep_num_outputs})

    cycles = int(cfg['cycles'])
    for cycle in range(1, cycles + 1):
        sched_samples = collect_scheduler_samples(dep_pol, sched_pol, scn, env_cfg, int(cfg['rollout_episodes']), seed_base + cycle * 17, float(cfg.get('scheduler_temp', 0.35)))
        sched_stats = train_scheduler_inner(sched_samples, sched_pol, cfg, device)
        dep_samples = collect_deployment_samples(dep_pol, sched_pol, scn, env_cfg, int(cfg['rollout_episodes']), seed_base + cycle * 29, int(cfg.get('candidate_count', 8)))
        dep_stats = train_deployment_inner(dep_samples, dep_pol, cfg, device)
        ev = eval_joint(dep_pol, sched_pol, scn, env_cfg, eval_eps, seed_bank)
        joint_score = score(ev)
        accepted = float(joint_score < best_score)
        if accepted:
            best_score = joint_score
            best_sched_state = copy.deepcopy(sched_pol.model.state_dict())
            best_dep_state = copy.deepcopy(dep_pol.model.state_dict())
            save_checkpoint(sched_pol.model, out_dir / 'scheduler_joint_best.pt', {'joint_score': best_score, 'obs_dim': sched_obs_dim, 'num_nodes': scn.num_nodes})
            save_checkpoint(dep_pol.model, out_dir / 'deployment_joint_best.pt', {'joint_score': best_score, 'obs_dim': dep_obs_dim, 'num_outputs': dep_num_outputs})
        else:
            sched_pol.model.load_state_dict(best_sched_state)
            dep_pol.model.load_state_dict(best_dep_state)
        row = {
            'cycle': cycle,
            **sched_stats,
            **dep_stats,
            **ev,
            'joint_score': joint_score,
            'accepted_update': accepted,
            'sched_sample_count': len(sched_samples),
            'dep_sample_count': len(dep_samples),
        }
        logger.log(row)
        print(row)
        plot_lines(out_dir / 'stage7_joint_train.csv', 'cycle', ['mean_latency', 'p90_latency', 'worst_latency'], out_dir / 'stage7_joint_latency.png', 'Stage7 joint latency', ma_window=0)
        plot_lines(out_dir / 'stage7_joint_train.csv', 'cycle', ['scheduler_actor_loss', 'deployment_actor_loss'], out_dir / 'stage7_joint_losses.png', 'Stage7 joint losses', ma_window=0)
    logger.close()


if __name__ == '__main__':
    main()
