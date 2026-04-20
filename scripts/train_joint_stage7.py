from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import copy
import math

import numpy as np
import torch
from torch import nn

from src.common.config import load_yaml
from src.common.seed import set_seed
from src.common.metric_logger import MetricLogger
from src.common.plotter import plot_lines
from src.common.device import resolve_device, describe_device
from src.env.core import (
    build_scenario, flatten_macro_obs, generate_macro_obs, stage_count,
    generate_scheduler_tasks, greedy_direct_deployment, make_scheduler_obs,
    mutate_deployment, random_feasible_deployment, repair_deployment,
    scheduler_action_costs, scheduler_action_mask, scheduler_target_probs,
)
from src.agents.scheduler.policy import SchedulerPolicy
from src.agents.deployment.policy import DeploymentPolicy
from scripts._shared import load_deployment_policy, load_scheduler_policy, save_checkpoint


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.quantile(arr, q))


def compute_joint_score(ev: dict, cfg: dict) -> float:
    score = (
        float(cfg.get('accept_mean_weight', 1.0)) * float(ev['mean_latency'])
        + float(cfg.get('accept_p90_weight', 0.25)) * float(ev['p90_latency'])
        + float(cfg.get('accept_worst_weight', 0.05)) * float(ev['worst_latency'])
        - float(cfg.get('accept_reward_weight', 0.0)) * float(ev.get('mean_reward', 0.0))
    )
    return float(score)


def materialize_deployment_vector(raw_vec: np.ndarray, score_vec: np.ndarray, scn, max_replicas: int) -> np.ndarray:
    raw = np.asarray(raw_vec, dtype=np.float32).reshape(stage_count(scn), scn.num_nodes)
    scores = np.asarray(score_vec, dtype=np.float32).reshape(stage_count(scn), scn.num_nodes)
    repaired = repair_deployment(raw, scn, max_replicas, scores=scores)
    return repaired.reshape(-1).astype(np.float32)


def clone_scheduler_policy(base_pol: SchedulerPolicy, obs_dim: int, num_nodes: int, device: torch.device) -> SchedulerPolicy:
    hidden = int(base_pol.model.net[0].out_features)
    clone = SchedulerPolicy(obs_dim, num_nodes, hidden=hidden, device=device)
    clone.model.load_state_dict(copy.deepcopy(base_pol.model.state_dict()))
    clone.model.to(device)
    return clone


def model_anchor_penalty(model: torch.nn.Module, anchor_state: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if not anchor_state:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        ref = anchor_state[name].to(param.device)
        penalty = penalty + torch.mean((param - ref) ** 2)
    return penalty


def policy_deployment_probs(dep_pol: DeploymentPolicy, obs_vec: np.ndarray) -> np.ndarray:
    dep_pol.model.eval()
    device = next(dep_pol.model.parameters()).device
    with torch.no_grad():
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        logits = dep_pol.model(obs_tensor)[0]
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs.astype(np.float32)


def rollout_macro_with_scheduler(
    macro_obs: np.ndarray,
    deployment: np.ndarray,
    sched_pol: SchedulerPolicy,
    scn,
    env_cfg: dict,
    seed_offset: int = 0,
):
    seed = int(env_cfg.get('seed', 0)) + int(seed_offset) + int(macro_obs.sum() * 10) % 100000
    rng = np.random.default_rng(seed)
    tasks = generate_scheduler_tasks(macro_obs, scn, env_cfg, rng)
    comp_assignments = []
    flows = []
    est_node_loads = np.zeros(scn.num_nodes, dtype=np.float32)
    per_task_latency = []
    for task in tasks:
        prev = task['origin']
        task_stage_meta = []
        for local_stage in range(scn.service_stages[task['service']]):
            obs = make_scheduler_obs(task, local_stage, prev, deployment, macro_obs, scn, est_node_loads)
            mask = scheduler_action_mask(task, local_stage, deployment, scn)
            action = int(sched_pol.act(obs, mask))
            if mask[action] < 0.5:
                action = int(np.argmax(mask))
            c = float(task['stage_compute'][local_stage])
            d = float(task['stage_data'][local_stage])
            comp_assignments.append((action, c))
            flow_idx = None
            if prev != action:
                flow_idx = len(flows)
                flows.append(((prev, action), d))
            task_stage_meta.append((len(comp_assignments) - 1, flow_idx))
            est_node_loads[action] += 1.0
            prev = action
        per_task_latency.append(task_stage_meta)
    from src.kkt.solver import kkt_bandwidth_latencies, kkt_compute_latencies
    comp_lats = kkt_compute_latencies(scn.node_compute, comp_assignments)
    flow_lats = kkt_bandwidth_latencies(scn.bandwidth, flows)
    latencies = []
    for task_meta in per_task_latency:
        total = 0.0
        for comp_idx, flow_idx in task_meta:
            total += comp_lats[comp_idx]
            if flow_idx is not None:
                total += flow_lats[flow_idx]
        latencies.append(total)
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    total_reward = -float(sum(latencies))
    return {
        'total_reward': total_reward,
        'mean_window_latency': mean_lat * len(latencies),
        'mean_task_latency': mean_lat,
        'num_tasks': len(latencies),
    }


def eval_joint(dep_pol, sched_pol, scn, env_cfg, episodes: int, seed_list: list[int]):
    rows = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        for _ in range(episodes):
            macro = generate_macro_obs(scn, env_cfg, rng)
            dep = dep_pol.act(flatten_macro_obs(macro), scn, int(env_cfg['max_replicas']))
            out = rollout_macro_with_scheduler(macro, dep, sched_pol, scn, env_cfg, seed_offset=0)
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
        samples.extend(collect_scheduler_samples_for_macro(macro, dep, sched_pol, scn, env_cfg, temp, seed_offset=0))
    return samples


def collect_scheduler_samples_for_macro(
    macro_obs: np.ndarray,
    deployment: np.ndarray,
    sched_pol: SchedulerPolicy,
    scn,
    env_cfg: dict,
    temp: float,
    seed_offset: int = 0,
):
    seed = int(env_cfg.get('seed', 0)) + int(seed_offset) + int(macro_obs.sum() * 10) % 100000
    rng = np.random.default_rng(seed)
    samples = []
    tasks = generate_scheduler_tasks(macro_obs, scn, env_cfg, rng)
    est_node_loads = np.zeros(scn.num_nodes, dtype=np.float32)
    for task in tasks:
        prev = task['origin']
        for local_stage in range(scn.service_stages[task['service']]):
            obs = make_scheduler_obs(task, local_stage, prev, deployment, macro_obs, scn, est_node_loads)
            mask = scheduler_action_mask(task, local_stage, deployment, scn)
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


def train_scheduler_inner(
    samples,
    pol,
    cfg,
    device,
    lr_scale: float = 1.0,
    anchor_state: dict[str, torch.Tensor] | None = None,
    steps_key: str = 'scheduler_inner_steps',
    batch_key: str = 'scheduler_batch_size',
    lr_key: str = 'scheduler_lr',
    anchor_coef_key: str = 'scheduler_anchor_coef',
):
    if not samples:
        return {
            'scheduler_actor_loss': 0.0,
            'scheduler_entropy': 0.0,
            'scheduler_regret': 0.0,
            'scheduler_anchor_loss': 0.0,
        }
    obs = torch.from_numpy(np.stack([s['obs'] for s in samples]).astype('float32'))
    mask = torch.from_numpy(np.stack([s['mask'] for s in samples]).astype('float32'))
    target = torch.from_numpy(np.stack([s['target_probs'] for s in samples]).astype('float32'))
    costs = torch.from_numpy(np.stack([s['costs'] for s in samples]).astype('float32'))
    regret = torch.from_numpy(np.array([s['regret'] for s in samples], dtype='float32')).unsqueeze(1)
    valid_count = torch.from_numpy(np.array([s['valid_action_count'] for s in samples], dtype='float32'))
    pol.model.train()
    lr = float(cfg.get(lr_key, cfg.get('scheduler_lr', 1e-4))) * float(lr_scale)
    opt = torch.optim.Adam(pol.model.parameters(), lr=lr)
    bs = int(cfg.get(batch_key, cfg.get('scheduler_batch_size', 128)))
    steps = int(cfg.get(steps_key, cfg.get('scheduler_inner_steps', 80)))
    entropy_coef = float(cfg.get('scheduler_entropy_coef', 0.004))
    regret_coef = float(cfg.get('scheduler_regret_coef', 0.6))
    anchor_coef = float(cfg.get(anchor_coef_key, cfg.get('scheduler_anchor_coef', 0.0)))
    last = {}
    for _ in range(steps):
        idx = torch.randint(0, len(samples), (min(bs, len(samples)),))
        ob = obs[idx].to(device)
        mb = mask[idx].to(device)
        tb = target[idx].to(device)
        cb = costs[idx].to(device)
        rb = regret[idx].to(device)
        vcb = valid_count[idx].to(device)
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
        valid_pair = (vcb > 1).float().unsqueeze(1)
        rank_loss = (valid_pair * torch.relu(0.10 - (best_logit - second_logit))).mean()
        anchor_loss = model_anchor_penalty(pol.model, anchor_state) if anchor_coef > 0.0 else torch.tensor(0.0, device=device)
        loss = plan_loss + relative_cost + 0.4 * rank_loss - entropy_coef * entropy + anchor_coef * anchor_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.model.parameters(), 1.0)
        opt.step()
        last = {
            'scheduler_actor_loss': float(loss.item()),
            'scheduler_entropy': float(entropy.item()),
            'scheduler_regret': float(rb.mean().item()),
            'scheduler_plan_loss': float(plan_loss.item()),
            'scheduler_relative_cost': float(relative_cost.item()),
            'scheduler_anchor_loss': float(anchor_loss.item()),
        }
    return last


def build_deployment_candidate_pool(
    macro_obs: np.ndarray,
    current_x: np.ndarray,
    dep_pol: DeploymentPolicy,
    scn,
    env_cfg: dict,
    candidate_count: int,
    rng: np.random.Generator,
):
    max_replicas = int(env_cfg['max_replicas'])
    obs = flatten_macro_obs(macro_obs)
    probs = policy_deployment_probs(dep_pol, obs)
    current_flat = np.asarray(current_x, dtype=np.float32).reshape(-1)
    heuristic = greedy_direct_deployment(macro_obs, scn, max_replicas, env_cfg, rng).reshape(-1).astype(np.float32)
    actor_greedy = materialize_deployment_vector((probs > 0.5).astype(np.float32), probs, scn, max_replicas)
    candidates: list[np.ndarray] = []

    def add_candidate(vec: np.ndarray):
        flat = np.asarray(vec, dtype=np.float32).reshape(-1)
        if not any(np.array_equal(flat, existing) for existing in candidates):
            candidates.append(flat)

    add_candidate(current_flat)
    add_candidate(heuristic)
    add_candidate(actor_greedy)

    mutation_prob = float(env_cfg.get('deployment_stage7_mutation_prob', 0.22))
    attempts = 0
    max_attempts = max(16, candidate_count * 8)
    while len(candidates) < candidate_count and attempts < max_attempts:
        attempts += 1
        draw = rng.random()
        if draw < 0.35:
            sampled = (rng.random(probs.shape[0]) < probs).astype(np.float32)
            cand = materialize_deployment_vector(sampled, probs, scn, max_replicas)
        elif draw < 0.75:
            base = candidates[int(rng.integers(0, len(candidates)))].reshape(stage_count(scn), scn.num_nodes)
            cand = mutate_deployment(base, macro_obs, scn, max_replicas, rng, mutation_prob, env_cfg).reshape(-1).astype(np.float32)
        else:
            cand = random_feasible_deployment(macro_obs, scn, max_replicas, rng, env_cfg).reshape(-1).astype(np.float32)
        add_candidate(cand)
    return np.stack(candidates).astype(np.float32)


def select_response_candidate_indices(frozen_latencies: np.ndarray, response_topk: int) -> list[int]:
    order = list(np.argsort(frozen_latencies).astype(int))
    chosen: list[int] = []
    if 0 not in chosen:
        chosen.append(0)
    for idx in order:
        if idx not in chosen:
            chosen.append(int(idx))
        if len(chosen) >= max(1, response_topk):
            break
    return chosen


def compute_candidate_objective(
    frozen_latency: float,
    response_latency: float,
    switch_cost: float,
    cfg: dict,
) -> float:
    return float(
        float(cfg.get('deployment_frozen_latency_weight', 0.25)) * float(frozen_latency)
        + float(cfg.get('deployment_response_latency_weight', 0.75)) * float(response_latency)
        + float(cfg.get('deployment_switch_penalty', 0.05)) * float(switch_cost)
    )


def collect_deployment_response_samples(
    dep_pol: DeploymentPolicy,
    sched_pol: SchedulerPolicy,
    scn,
    env_cfg: dict,
    cfg: dict,
    device: torch.device,
    episodes: int,
    seed: int,
    sched_obs_dim: int,
):
    rng = np.random.default_rng(seed)
    samples = []
    label_gains = []
    reward_gains = []
    response_gains = []
    chosen_indices = []
    max_replicas = int(env_cfg['max_replicas'])
    candidate_count = int(cfg.get('deployment_candidate_count', cfg.get('candidate_count', 8)))
    response_topk = int(cfg.get('deployment_response_topk', min(3, candidate_count)))
    train_seed_offset = int(cfg.get('deployment_response_train_seed_offset', 1100))
    eval_seed_offset = int(cfg.get('deployment_response_eval_seed_offset', 2200))
    scheduler_temp = float(cfg.get('scheduler_temp', 0.35))
    for episode_idx in range(episodes):
        macro = generate_macro_obs(scn, env_cfg, rng)
        obs = flatten_macro_obs(macro).astype(np.float32)
        current = dep_pol.act(obs, scn, max_replicas).reshape(-1).astype(np.float32)
        candidates = build_deployment_candidate_pool(macro, current, dep_pol, scn, env_cfg, candidate_count, rng)
        frozen_latencies = []
        frozen_rewards = []
        for cand in candidates:
            dep = cand.reshape(stage_count(scn), scn.num_nodes)
            out = rollout_macro_with_scheduler(macro, dep, sched_pol, scn, env_cfg, seed_offset=0)
            frozen_latencies.append(float(out['mean_window_latency']))
            frozen_rewards.append(float(out['total_reward']))
        frozen_latencies_np = np.asarray(frozen_latencies, dtype=np.float32)
        selected = select_response_candidate_indices(frozen_latencies_np, response_topk)
        current_response_latency = float(frozen_latencies_np[0])
        current_response_reward = float(frozen_rewards[0])
        candidate_records = {}
        for cand_idx in selected:
            cand = candidates[cand_idx].reshape(stage_count(scn), scn.num_nodes)
            response_sched = clone_scheduler_policy(sched_pol, sched_obs_dim, scn.num_nodes, device)
            response_samples = collect_scheduler_samples_for_macro(
                macro,
                cand,
                response_sched,
                scn,
                env_cfg,
                scheduler_temp,
                seed_offset=train_seed_offset + episode_idx * 17 + cand_idx,
            )
            response_train_stats = train_scheduler_inner(
                response_samples,
                response_sched,
                cfg,
                device,
                lr_scale=float(cfg.get('response_scheduler_lr_scale', 0.35)),
                anchor_state=copy.deepcopy(sched_pol.model.state_dict()),
                steps_key='response_scheduler_inner_steps',
                batch_key='response_scheduler_batch_size',
                lr_key='scheduler_lr',
                anchor_coef_key='response_scheduler_anchor_coef',
            )
            response_eval = rollout_macro_with_scheduler(
                macro,
                cand,
                response_sched,
                scn,
                env_cfg,
                seed_offset=eval_seed_offset + episode_idx * 19 + cand_idx,
            )
            candidate_records[cand_idx] = {
                'frozen_latency': float(frozen_latencies_np[cand_idx]),
                'frozen_reward': float(frozen_rewards[cand_idx]),
                'response_latency': float(response_eval['mean_window_latency']),
                'response_reward': float(response_eval['total_reward']),
                'train_stats': response_train_stats,
            }
        if 0 in candidate_records:
            current_response_latency = float(candidate_records[0]['response_latency'])
            current_response_reward = float(candidate_records[0]['response_reward'])
        best_idx = 0
        best_obj = compute_candidate_objective(
            float(frozen_latencies_np[0]),
            current_response_latency,
            0.0,
            cfg,
        )
        for cand_idx, record in candidate_records.items():
            switch_cost = float(np.mean(np.abs(candidates[cand_idx] - current)))
            cand_obj = compute_candidate_objective(
                record['frozen_latency'],
                record['response_latency'],
                switch_cost,
                cfg,
            )
            if cand_obj < best_obj:
                best_obj = cand_obj
                best_idx = int(cand_idx)
        best_x = candidates[best_idx]
        best_record = candidate_records.get(best_idx, {
            'response_latency': float(frozen_latencies_np[best_idx]),
            'response_reward': float(frozen_rewards[best_idx]),
            'frozen_latency': float(frozen_latencies_np[best_idx]),
            'frozen_reward': float(frozen_rewards[best_idx]),
            'train_stats': {},
        })
        current_obj = compute_candidate_objective(
            float(frozen_latencies_np[0]),
            current_response_latency,
            0.0,
            cfg,
        )
        relative_gain = max(0.0, (current_obj - best_obj) / max(current_obj, 1e-6))
        reward_gain = float(best_record['response_reward'] - current_response_reward)
        switch_cost = float(np.mean(np.abs(best_x - current)))
        samples.append({
            'macro_obs': obs.astype(np.float32),
            'x_label': best_x.astype(np.float32),
            'current_x': current.astype(np.float32),
            'relative_gain': relative_gain,
            'switch_cost_est': switch_cost,
            'current_latency': float(frozen_latencies_np[0]),
            'best_candidate_latency': float(best_record['response_latency']),
            'current_reward': current_response_reward,
            'best_candidate_reward': float(best_record['response_reward']),
        })
        label_gains.append(relative_gain)
        reward_gains.append(reward_gain)
        response_gains.append(max(0.0, (current_response_latency - float(best_record['response_latency'])) / max(current_response_latency, 1e-6)))
        chosen_indices.append(float(best_idx))
    stats = {
        'deployment_label_relative_gain': float(np.mean(label_gains)) if label_gains else 0.0,
        'deployment_label_reward_gain': float(np.mean(reward_gains)) if reward_gains else 0.0,
        'deployment_response_gain': float(np.mean(response_gains)) if response_gains else 0.0,
        'deployment_candidate_choice_mean': float(np.mean(chosen_indices)) if chosen_indices else 0.0,
    }
    return samples, stats


def train_deployment_inner(
    samples,
    dep_pol,
    cfg,
    device,
    lr_scale: float = 1.0,
    anchor_state: dict[str, torch.Tensor] | None = None,
):
    if not samples:
        return {
            'deployment_actor_loss': 0.0,
            'deployment_bc_loss': 0.0,
            'deployment_switch_cost': 0.0,
            'deployment_anchor_loss': 0.0,
        }
    obs = torch.from_numpy(np.stack([s['macro_obs'] for s in samples]).astype('float32'))
    xlab = torch.from_numpy(np.stack([s['x_label'] for s in samples]).astype('float32'))
    cur = torch.from_numpy(np.stack([s['current_x'] for s in samples]).astype('float32'))
    gain = torch.tensor([s['relative_gain'] for s in samples], dtype=torch.float32).unsqueeze(1)
    swt = torch.tensor([s['switch_cost_est'] for s in samples], dtype=torch.float32).unsqueeze(1)
    dep_pol.model.train()
    lr = float(cfg.get('deployment_lr', 1e-4)) * float(lr_scale)
    opt = torch.optim.Adam(dep_pol.model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    bs = int(cfg.get('deployment_batch_size', 64))
    steps = int(cfg.get('deployment_inner_steps', 20))
    stay_coef = float(cfg.get('deployment_stay_coef', 0.2))
    switch_coef = float(cfg.get('deployment_switch_coef', 0.05))
    anchor_coef = float(cfg.get('deployment_anchor_coef', 0.0))
    gain_floor = float(cfg.get('deployment_gain_floor', 0.0))
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
        gain_weight = torch.where(gb > gain_floor, 0.5 + gb, torch.full_like(gb, 0.25))
        bc_loss = (gain_weight * bc_per).mean()
        stay_loss = (probs - xc).abs().mean()
        pred_switch = (probs - xc).abs().mean(dim=1, keepdim=True)
        anchor_loss = model_anchor_penalty(dep_pol.model, anchor_state) if anchor_coef > 0.0 else torch.tensor(0.0, device=device)
        loss = (
            bc_loss
            + stay_coef * (1.0 - gb.mean()) * stay_loss
            + switch_coef * pred_switch.mean()
            + anchor_coef * anchor_loss
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dep_pol.model.parameters(), 2.0)
        opt.step()
        last = {
            'deployment_actor_loss': float(loss.item()),
            'deployment_bc_loss': float(bc_loss.item()),
            'deployment_switch_cost': float(pred_switch.mean().item()),
            'deployment_gain': float(gb.mean().item()),
            'deployment_target_switch': float(swb.mean().item()),
            'deployment_anchor_loss': float(anchor_loss.item()),
        }
    return last


def maybe_save_best(
    out_dir: Path,
    sched_pol: SchedulerPolicy,
    dep_pol: DeploymentPolicy,
    joint_score: float,
    ev: dict,
    sched_obs_dim: int,
    dep_obs_dim: int,
    dep_num_outputs: int,
    scn,
):
    save_checkpoint(
        sched_pol.model,
        out_dir / 'scheduler_joint_best.pt',
        {
            'joint_score': joint_score,
            'mean_latency': float(ev['mean_latency']),
            'mean_reward': float(ev['mean_reward']),
            'obs_dim': sched_obs_dim,
            'num_nodes': scn.num_nodes,
        },
    )
    save_checkpoint(
        dep_pol.model,
        out_dir / 'deployment_joint_best.pt',
        {
            'joint_score': joint_score,
            'mean_latency': float(ev['mean_latency']),
            'mean_reward': float(ev['mean_reward']),
            'obs_dim': dep_obs_dim,
            'num_outputs': dep_num_outputs,
        },
    )


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
    sched_obs_dim = make_scheduler_obs(t, 0, 0, dummy_dep, dummy_macro, scn, np.zeros(scn.num_nodes, dtype=np.float32)).shape[0]
    dep_obs_dim = flatten_macro_obs(dummy_macro).shape[0]
    dep_num_outputs = stage_count(scn) * scn.num_nodes

    sched_pol = load_scheduler_policy(args.scheduler_checkpoint, sched_obs_dim, scn.num_nodes, device=device)
    dep_pol = load_deployment_policy(args.deployment_checkpoint, dep_obs_dim, dep_num_outputs, device=device)
    initial_sched_state = copy.deepcopy(sched_pol.model.state_dict())
    initial_dep_state = copy.deepcopy(dep_pol.model.state_dict())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(out_dir / 'stage7_joint_train.csv', out_dir / 'stage7_joint_train.jsonl')

    seed_base = int(cfg.get('fixed_seed_base', int(env_cfg['seed']) + 50000))
    seed_count = int(cfg.get('fixed_seed_count', 10))
    seed_bank = [seed_base + 101 * i for i in range(seed_count)]
    eval_eps = int(cfg.get('eval_episodes', 20))
    accept_gain_floor = float(cfg.get('accept_gain_floor', 0.0))
    baseline_eval = eval_joint(dep_pol, sched_pol, scn, env_cfg, eval_eps, seed_bank)
    current_eval = dict(baseline_eval)
    best_eval = dict(baseline_eval)
    current_score = compute_joint_score(current_eval, cfg)
    best_score = current_score
    current_sched_state = copy.deepcopy(sched_pol.model.state_dict())
    current_dep_state = copy.deepcopy(dep_pol.model.state_dict())
    joint_step = 0

    baseline_row = {
        'joint_step': joint_step,
        'outer_cycle': 0,
        'phase': 'baseline',
        'update_accepted': 1.0,
        'joint_score': current_score,
        'best_joint_score': best_score,
        'baseline_mean_latency': float(baseline_eval['mean_latency']),
        'baseline_mean_reward': float(baseline_eval['mean_reward']),
        'mean_latency_delta_vs_baseline': 0.0,
        'mean_reward_delta_vs_baseline': 0.0,
        **current_eval,
    }
    logger.log(baseline_row)
    maybe_save_best(out_dir, sched_pol, dep_pol, best_score, best_eval, sched_obs_dim, dep_obs_dim, dep_num_outputs, scn)

    cycles = int(cfg['cycles'])
    scheduler_blocks = int(cfg.get('scheduler_blocks_per_cycle', 1))
    scheduler_rollout_episodes = int(cfg.get('scheduler_rollout_episodes', cfg.get('rollout_episodes', 12)))
    deployment_rollout_episodes = int(cfg.get('deployment_rollout_episodes', max(4, scheduler_rollout_episodes // 2)))
    deployment_update_interval = int(cfg.get('deployment_update_interval', 1))

    for cycle in range(1, cycles + 1):
        for block in range(scheduler_blocks):
            sched_samples = collect_scheduler_samples(
                dep_pol,
                sched_pol,
                scn,
                env_cfg,
                scheduler_rollout_episodes,
                seed_base + cycle * 17 + block * 7,
                float(cfg.get('scheduler_temp', 0.35)),
            )
            sched_stats = train_scheduler_inner(
                sched_samples,
                sched_pol,
                cfg,
                device,
                lr_scale=float(cfg.get('scheduler_lr_scale', 1.0)),
                anchor_state=initial_sched_state,
            )
            trial_eval = eval_joint(dep_pol, sched_pol, scn, env_cfg, eval_eps, seed_bank)
            trial_score = compute_joint_score(trial_eval, cfg)
            improved = float(trial_score + accept_gain_floor < current_score)
            if improved:
                current_score = trial_score
                current_eval = dict(trial_eval)
                current_sched_state = copy.deepcopy(sched_pol.model.state_dict())
                if trial_score < best_score:
                    best_score = trial_score
                    best_eval = dict(trial_eval)
                    maybe_save_best(out_dir, sched_pol, dep_pol, best_score, best_eval, sched_obs_dim, dep_obs_dim, dep_num_outputs, scn)
            else:
                sched_pol.model.load_state_dict(current_sched_state)
            joint_step += 1
            row = {
                'joint_step': joint_step,
                'outer_cycle': cycle,
                'phase': 'scheduler_block',
                'phase_index': block,
                'update_accepted': improved,
                'joint_score': trial_score,
                'best_joint_score': best_score,
                'baseline_mean_latency': float(baseline_eval['mean_latency']),
                'baseline_mean_reward': float(baseline_eval['mean_reward']),
                'mean_latency_delta_vs_baseline': float(trial_eval['mean_latency'] - baseline_eval['mean_latency']),
                'mean_reward_delta_vs_baseline': float(trial_eval['mean_reward'] - baseline_eval['mean_reward']),
                'sched_sample_count': len(sched_samples),
                'dep_sample_count': 0,
                **sched_stats,
                **trial_eval,
            }
            logger.log(row)
            print(row)

        if cycle % deployment_update_interval == 0:
            dep_samples, dep_collect_stats = collect_deployment_response_samples(
                dep_pol,
                sched_pol,
                scn,
                env_cfg,
                cfg,
                device,
                deployment_rollout_episodes,
                seed_base + cycle * 29,
                sched_obs_dim,
            )
            dep_stats = train_deployment_inner(
                dep_samples,
                dep_pol,
                cfg,
                device,
                lr_scale=float(cfg.get('deployment_lr_scale', 1.0)),
                anchor_state=initial_dep_state,
            )
            trial_eval = eval_joint(dep_pol, sched_pol, scn, env_cfg, eval_eps, seed_bank)
            trial_score = compute_joint_score(trial_eval, cfg)
            improved = float(trial_score + accept_gain_floor < current_score)
            if improved:
                current_score = trial_score
                current_eval = dict(trial_eval)
                current_dep_state = copy.deepcopy(dep_pol.model.state_dict())
                if trial_score < best_score:
                    best_score = trial_score
                    best_eval = dict(trial_eval)
                    maybe_save_best(out_dir, sched_pol, dep_pol, best_score, best_eval, sched_obs_dim, dep_obs_dim, dep_num_outputs, scn)
            else:
                dep_pol.model.load_state_dict(current_dep_state)
            joint_step += 1
            row = {
                'joint_step': joint_step,
                'outer_cycle': cycle,
                'phase': 'deployment_update',
                'phase_index': 0,
                'update_accepted': improved,
                'joint_score': trial_score,
                'best_joint_score': best_score,
                'baseline_mean_latency': float(baseline_eval['mean_latency']),
                'baseline_mean_reward': float(baseline_eval['mean_reward']),
                'mean_latency_delta_vs_baseline': float(trial_eval['mean_latency'] - baseline_eval['mean_latency']),
                'mean_reward_delta_vs_baseline': float(trial_eval['mean_reward'] - baseline_eval['mean_reward']),
                'sched_sample_count': 0,
                'dep_sample_count': len(dep_samples),
                **dep_collect_stats,
                **dep_stats,
                **trial_eval,
            }
            logger.log(row)
            print(row)

        joint_step += 1
        cycle_row = {
            'joint_step': joint_step,
            'outer_cycle': cycle,
            'phase': 'cycle_summary',
            'phase_index': 0,
            'update_accepted': 1.0,
            'joint_score': current_score,
            'best_joint_score': best_score,
            'baseline_mean_latency': float(baseline_eval['mean_latency']),
            'baseline_mean_reward': float(baseline_eval['mean_reward']),
            'mean_latency_delta_vs_baseline': float(current_eval['mean_latency'] - baseline_eval['mean_latency']),
            'mean_reward_delta_vs_baseline': float(current_eval['mean_reward'] - baseline_eval['mean_reward']),
            **current_eval,
        }
        logger.log(cycle_row)
        print(cycle_row)
        plot_lines(out_dir / 'stage7_joint_train.csv', 'joint_step', ['mean_latency'], out_dir / 'stage7_joint_latency.png', 'Stage7 joint mean latency', ma_window=0)
        plot_lines(out_dir / 'stage7_joint_train.csv', 'joint_step', ['mean_reward'], out_dir / 'stage7_joint_reward.png', 'Stage7 joint mean reward', ma_window=0)
        plot_lines(out_dir / 'stage7_joint_train.csv', 'joint_step', ['scheduler_actor_loss', 'deployment_actor_loss'], out_dir / 'stage7_joint_losses.png', 'Stage7 joint losses', ma_window=0)
        plot_lines(out_dir / 'stage7_joint_train.csv', 'joint_step', ['joint_score', 'best_joint_score'], out_dir / 'stage7_joint_score.png', 'Stage7 joint score', ma_window=0)

    logger.close()


if __name__ == '__main__':
    main()
