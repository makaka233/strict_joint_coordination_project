from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse, math
import numpy as np
import torch
from torch import nn
from src.common.config import load_yaml
from src.common.seed import set_seed
from src.common.metric_logger import MetricLogger
from src.common.plotter import plot_lines
from src.common.device import resolve_device, describe_device
from src.env.core import (
    build_scenario, generate_macro_obs, stage_count, greedy_direct_deployment,
    random_feasible_deployment, mutate_deployment, evaluate_deployment_with_scheduler,
)
from src.agents.scheduler.policy import SchedulerPolicy
from src.models.mlp import MLP
from scripts._shared import save_checkpoint


def masked_probs(logits: torch.Tensor, mask: torch.Tensor):
    masked = logits + (mask - 1.0) * 1e9
    probs = torch.softmax(masked, dim=1)
    return masked, probs


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def choose_eval_deployment(macro, scn, env_cfg, cfg, rng):
    max_replicas = int(env_cfg['max_replicas'])
    p_h = float(cfg.get('eval_heuristic_prob', 0.40))
    p_r = float(cfg.get('eval_random_prob', 0.30))
    p_m = float(cfg.get('eval_mutate_prob', 0.30))
    probs = np.array([p_h, p_r, p_m], dtype=np.float64)
    probs = probs / probs.sum()
    source = int(rng.choice(np.arange(3), p=probs))
    if source == 0:
        return greedy_direct_deployment(macro, scn, max_replicas, env_cfg, rng)
    if source == 1:
        return random_feasible_deployment(macro, scn, max_replicas, rng, env_cfg)
    base = greedy_direct_deployment(macro, scn, max_replicas, env_cfg, rng)
    return mutate_deployment(base, macro, scn, max_replicas, rng, float(cfg.get('eval_mutation_prob', 0.20)), env_cfg)


def eval_policy_fixed(policy, scn, env_cfg, cfg, episodes, seed_list):
    vals, rewards = [], []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        local_vals, local_rewards = [], []
        for _ in range(episodes):
            macro = generate_macro_obs(scn, env_cfg, rng)
            dep = choose_eval_deployment(macro, scn, env_cfg, cfg, rng)
            out = evaluate_deployment_with_scheduler(macro, dep, scn, env_cfg, lambda obs, mask, task, local_stage, prev: policy.act(obs, mask))
            local_vals.append(out['mean_window_latency'])
            local_rewards.append(out['total_reward'])
        vals.append(float(np.mean(local_vals)))
        rewards.append(float(np.mean(local_rewards)))
    return float(np.mean(vals)), float(np.mean(rewards))


def phase_name(step: int, total_steps: int, a_frac: float, b_frac: float) -> str:
    a_end = int(total_steps * a_frac)
    b_end = int(total_steps * (a_frac + b_frac))
    if step < a_end:
        return 'A'
    if step < b_end:
        return 'B'
    return 'C'


def phase_weights(phase: str, cfg: dict):
    return {
        'plan_weight': float(cfg.get(f'{phase}_plan_weight', cfg.get('plan_weight', 1.0))),
        'rank_weight': float(cfg.get(f'{phase}_rank_weight', cfg.get('rank_weight', 0.5))),
        'relative_weight': float(cfg.get(f'{phase}_relative_weight', cfg.get('relative_weight', 1.0))),
        'entropy_coef': float(cfg.get(f'{phase}_entropy_coef', cfg.get('entropy_coef', 0.01))),
        'entropy_floor': float(cfg.get(f'{phase}_entropy_floor', cfg.get('entropy_floor', 0.12))),
        'entropy_floor_coef': float(cfg.get(f'{phase}_entropy_floor_coef', cfg.get('entropy_floor_coef', 0.08))),
        'planner_tau': float(cfg.get(f'{phase}_planner_tau', cfg.get('planner_tau', 0.30))),
        'rank_margin': float(cfg.get(f'{phase}_rank_margin', cfg.get('rank_margin', 0.08))),
        'gumbel_noise': float(cfg.get(f'{phase}_gumbel_noise', cfg.get('gumbel_noise', 0.0))),
    }


def maybe_gumbel(logits: torch.Tensor, scale: float):
    if scale <= 0.0:
        return logits
    u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    g = -torch.log(-torch.log(u))
    return logits + scale * g


def load_wm_model(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get('meta', {})
    obs_dim = int(meta.get('obs_dim'))
    num_nodes = int(meta.get('num_nodes'))
    hidden = int(meta.get('hidden', 128))
    cost_scale = float(meta.get('cost_scale', 1.0))
    model = MLP(obs_dim + num_nodes, 2, hidden=hidden).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, cost_scale, num_nodes


def wm_predict_costs(wm_model: torch.nn.Module, xb: torch.Tensor, mb: torch.Tensor, cost_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    B, D = xb.shape
    N = mb.shape[1]
    eye = torch.eye(N, device=xb.device).unsqueeze(0).expand(B, N, N)
    obs_rep = xb.unsqueeze(1).expand(B, N, D)
    inp = torch.cat([obs_rep, eye], dim=2).reshape(B * N, D + N)
    pred = wm_model(inp).reshape(B, N, 2)
    pred_cost = torch.expm1(pred[:, :, 0].clamp(min=-20.0, max=20.0) * cost_scale)
    pred_gap = torch.relu(pred[:, :, 1])
    pred_cost = pred_cost + (1.0 - mb) * 1e9
    return pred_cost, pred_gap


def planner_distribution(pred_cost: torch.Tensor, mask: torch.Tensor, tau: float):
    logits = -pred_cost / max(tau, 1e-6)
    logits = logits + (mask - 1.0) * 1e9
    probs = torch.softmax(logits, dim=1)
    return logits, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-config', required=True)
    ap.add_argument('--actor-config', required=True)
    ap.add_argument('--wm-checkpoint', required=True)
    ap.add_argument('--replay', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--gpu-id', type=int, default=2)
    args = ap.parse_args()
    env_cfg = load_yaml(args.env_config)
    cfg = load_yaml(args.actor_config)
    set_seed(int(env_cfg['seed']))
    torch.set_num_threads(1)
    device = resolve_device(args.device, args.gpu_id)
    print({'stage': 'scheduler_actor', 'device': describe_device(device)})
    scn = build_scenario(env_cfg)
    replay = torch.load(args.replay, weights_only=False)
    obs_np = np.stack([r['obs'] for r in replay]).astype('float32')
    mask_np = np.stack([r['mask'] for r in replay]).astype('float32')
    real_cost_np = np.stack([r['action_costs'] for r in replay]).astype('float32')
    valid_count_np = np.array([r.get('valid_action_count', int(np.isfinite(r['action_costs']).sum())) for r in replay], dtype='float32')
    obs = torch.from_numpy(obs_np)
    mask = torch.from_numpy(mask_np)
    real_costs = torch.from_numpy(real_cost_np)
    valid_count = torch.from_numpy(valid_count_np)
    pol = SchedulerPolicy(obs.shape[1], scn.num_nodes, hidden=int(cfg['hidden']), device=device)
    ema_decay = float(cfg.get('ema_decay', 0.996))
    ema_pol = SchedulerPolicy(obs.shape[1], scn.num_nodes, hidden=int(cfg['hidden']), device=device)
    ema_pol.model.load_state_dict(pol.model.state_dict())
    for p in ema_pol.model.parameters():
        p.requires_grad_(False)
    wm_model, cost_scale, _ = load_wm_model(args.wm_checkpoint, device)
    opt = torch.optim.Adam(pol.model.parameters(), lr=float(cfg['lr']))
    logger = MetricLogger('outputs/metrics/stage3_scheduler_actor.csv', 'outputs/logs/stage3_scheduler_actor.jsonl')
    best_score = math.inf
    best_latency = math.inf
    global_step = 0
    bs = int(cfg['batch_size'])
    n = len(replay)
    total_steps = int(cfg['epochs']) * int(cfg['steps_per_epoch'])
    log_every = int(cfg.get('log_every_steps', 20))
    short_eval_every = int(cfg.get('short_eval_every_steps', 200))
    plot_every = int(cfg.get('plot_every_steps', 200))
    fixed_eval_seed_base = int(cfg.get('fixed_eval_seed_base', int(env_cfg['seed']) + 10000))
    fixed_eval_seed_count = int(cfg.get('fixed_eval_seed_count', 6))
    fixed_eval_episodes = int(cfg.get('eval_episodes', 16))
    fixed_eval_seeds = [fixed_eval_seed_base + 97 * i for i in range(fixed_eval_seed_count)]
    anchor_cfg = dict(cfg)
    anchor_cfg['eval_heuristic_prob'] = float(cfg.get('anchor_eval_heuristic_prob', 0.20))
    anchor_cfg['eval_random_prob'] = float(cfg.get('anchor_eval_random_prob', 0.40))
    anchor_cfg['eval_mutate_prob'] = float(cfg.get('anchor_eval_mutate_prob', 0.40))
    anchor_cfg['eval_mutation_prob'] = float(cfg.get('anchor_eval_mutation_prob', 0.28))
    anchor_seed_base = int(cfg.get('anchor_eval_seed_base', fixed_eval_seed_base + 5000))
    anchor_seed_count = int(cfg.get('anchor_eval_seed_count', 4))
    anchor_eval_episodes = int(cfg.get('anchor_eval_episodes', max(8, fixed_eval_episodes // 2)))
    anchor_eval_seeds = [anchor_seed_base + 131 * i for i in range(anchor_seed_count)]
    score_alpha = float(cfg.get('eval_score_alpha', 0.55))
    patience_evals = int(cfg.get('patience_evals', 18))
    min_steps_before_early_stop = int(cfg.get('min_steps_before_early_stop', 8000))
    no_improve_evals = 0
    phase_a_frac = float(cfg.get('phase_a_frac', 0.08))
    phase_b_frac = float(cfg.get('phase_b_frac', 0.70))
    for epoch in range(int(cfg['epochs'])):
        for step in range(int(cfg['steps_per_epoch'])):
            idx = torch.randint(0, n, (bs,))
            xb = obs[idx].to(device)
            mb = mask[idx].to(device)
            rb = real_costs[idx].to(device)
            vcb = valid_count[idx].to(device)
            logits = pol.model(xb)
            phase = phase_name(global_step, total_steps, phase_a_frac, phase_b_frac)
            w = phase_weights(phase, cfg)
            logits_masked, probs = masked_probs(logits, mb)
            perturbed = maybe_gumbel(logits_masked, w['gumbel_noise'])
            _, probs = masked_probs(perturbed, mb)
            log_probs = torch.log_softmax(perturbed + (mb - 1.0) * 1e9, dim=1)
            with torch.no_grad():
                pred_cost, pred_gap = wm_predict_costs(wm_model, xb, mb, cost_scale)
                _, plan_probs = planner_distribution(pred_cost, mb, w['planner_tau'])
                best_idx = torch.argmin(pred_cost, dim=1)
                best_cost = pred_cost.gather(1, best_idx[:, None]).squeeze(1)
                topk = min(int(cfg.get('rank_topk', 4)), pred_cost.shape[1])
                rank_idx = torch.topk(-pred_cost, k=topk, dim=1).indices
            plan_loss = -(plan_probs * log_probs).sum(dim=1).mean()
            expected_pred_cost_per = (probs * pred_cost).sum(dim=1)
            relative_cost = (expected_pred_cost_per - best_cost).mean()
            pair_losses = []
            best_logits = perturbed.gather(1, best_idx[:, None]).squeeze(1)
            for j in range(1, rank_idx.shape[1]):
                comp_idx = rank_idx[:, j]
                comp_logits = perturbed.gather(1, comp_idx[:, None]).squeeze(1)
                comp_cost = pred_cost.gather(1, comp_idx[:, None]).squeeze(1)
                adaptive_margin = torch.clamp((comp_cost - best_cost) + w['rank_margin'], min=w['rank_margin'])
                valid_pair = (vcb > j).float()
                pair_losses.append(valid_pair * torch.relu(adaptive_margin - (best_logits - comp_logits)))
            if pair_losses:
                pair_stack = torch.stack(pair_losses, dim=1)
                denom = torch.stack([(vcb > j).float() for j in range(1, rank_idx.shape[1])], dim=1).sum(dim=1).clamp_min(1.0)
                rank_loss = (pair_stack.sum(dim=1) / denom).mean()
            else:
                rank_loss = torch.tensor(0.0, device=device)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            entropy_pen = torch.relu(torch.tensor(w['entropy_floor'], device=device) - entropy)
            actor_loss = (
                w['plan_weight'] * plan_loss
                + w['rank_weight'] * rank_loss
                + w['relative_weight'] * relative_cost
                + w['entropy_floor_coef'] * entropy_pen
                - w['entropy_coef'] * entropy
            )
            # log-only oracle metrics on true costs
            valid_real = rb + (1.0 - mb) * 1e9
            best_real = torch.min(valid_real, dim=1).values
            expected_real = (probs * valid_real).sum(dim=1)
            critic_loss = nn.functional.mse_loss(expected_real, best_real)
            opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.model.parameters(), float(cfg.get('grad_clip', 1.0)))
            opt.step()
            update_ema(ema_pol.model, pol.model, ema_decay)
            global_step += 1
            row = {
                'global_step': global_step,
                'epoch': epoch,
                'step_in_epoch': step,
                'phase': phase,
                'actor_loss': float(actor_loss.item()),
                'critic_loss': float(critic_loss.item()),
                'imag_return': float(-expected_pred_cost_per.mean().item() * 1000.0),
                'entropy': float(entropy.item()),
                'plan_loss': float(plan_loss.item()),
                'expected_cost': float(expected_pred_cost_per.mean().item()),
                'relative_cost': float(relative_cost.item()),
                'margin_loss': float(rank_loss.item()),
                'planner_entropy': float((-(plan_probs * torch.log(plan_probs + 1e-8)).sum(dim=1).mean()).item()),
                'actor_vs_planner_agreement': float((probs.argmax(dim=1) == best_idx).float().mean().item()),
                'valid_action_count_mean': float(vcb.mean().item()),
                'real_expected_cost': float(expected_real.mean().item()),
                'real_best_cost': float(best_real.mean().item()),
                'lr': float(opt.param_groups[0]['lr']),
                'plan_weight': w['plan_weight'],
                'rank_weight': w['rank_weight'],
                'relative_weight': w['relative_weight'],
                'entropy_coef': w['entropy_coef'],
            }
            logger.log(row)
            if global_step % log_every == 0:
                print(row)
            if global_step % short_eval_every == 0:
                eval_lat, eval_reward = eval_policy_fixed(ema_pol, scn, env_cfg, cfg, episodes=fixed_eval_episodes, seed_list=fixed_eval_seeds)
                anchor_lat, anchor_reward = eval_policy_fixed(ema_pol, scn, env_cfg, anchor_cfg, episodes=anchor_eval_episodes, seed_list=anchor_eval_seeds)
                eval_score = score_alpha * eval_lat + (1.0 - score_alpha) * anchor_lat
                eval_row = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'step_in_epoch': step,
                    'eval_latency_short': eval_lat,
                    'eval_reward_short': eval_reward,
                    'anchor_eval_latency': anchor_lat,
                    'anchor_eval_reward': anchor_reward,
                    'eval_score': eval_score,
                    'best_eval_latency_so_far': min(best_latency, eval_lat),
                    'best_eval_score_so_far': min(best_score, eval_score),
                    'is_best_checkpoint': float(eval_score < best_score),
                }
                logger.log(eval_row)
                if eval_score + 1e-9 < best_score:
                    best_score = eval_score
                    best_latency = eval_lat
                    no_improve_evals = 0
                    save_checkpoint(ema_pol.model, args.out, {'best_eval_latency': best_latency, 'best_eval_score': best_score, 'obs_dim': obs.shape[1], 'num_nodes': scn.num_nodes, 'ema_decay': ema_decay})
                    print({'best_eval_latency': best_latency, 'best_eval_score': best_score, 'global_step': global_step})
                else:
                    no_improve_evals += 1
                if global_step >= min_steps_before_early_stop and no_improve_evals >= patience_evals and phase == 'C':
                    print({'early_stop': True, 'global_step': global_step, 'best_eval_latency': best_latency, 'best_eval_score': best_score})
                    logger.flush()
                    plot_lines('outputs/metrics/stage3_scheduler_actor.csv', 'global_step', ['actor_loss', 'plan_loss', 'expected_cost', 'relative_cost', 'margin_loss', 'entropy'], 'outputs/figures/stage3_scheduler_actor_train.png', 'Stage3 scheduler actor train', ma_window=50)
                    plot_lines('outputs/metrics/stage3_scheduler_actor.csv', 'global_step', ['eval_latency_short', 'anchor_eval_latency'], 'outputs/figures/stage3_scheduler_actor_eval_latency.png', 'Stage3 scheduler eval latency', ma_window=5)
                    plot_lines('outputs/metrics/stage3_scheduler_actor.csv', 'global_step', ['imag_return', 'eval_reward_short', 'anchor_eval_reward'], 'outputs/figures/stage3_scheduler_actor_reward.png', 'Stage3 scheduler reward', ma_window=5)
                    print({'best_eval_latency': best_latency, 'best_eval_score': best_score})
                    return
            if global_step % plot_every == 0:
                plot_lines('outputs/metrics/stage3_scheduler_actor.csv', 'global_step', ['actor_loss', 'plan_loss', 'expected_cost', 'relative_cost', 'margin_loss', 'entropy'], 'outputs/figures/stage3_scheduler_actor_train.png', 'Stage3 scheduler actor train', ma_window=25)
                plot_lines('outputs/metrics/stage3_scheduler_actor.csv', 'global_step', ['eval_latency_short', 'anchor_eval_latency'], 'outputs/figures/stage3_scheduler_actor_eval_latency.png', 'Stage3 scheduler eval latency', ma_window=5)
                plot_lines('outputs/metrics/stage3_scheduler_actor.csv', 'global_step', ['imag_return', 'eval_reward_short', 'anchor_eval_reward'], 'outputs/figures/stage3_scheduler_actor_reward.png', 'Stage3 scheduler reward', ma_window=5)
    logger.flush()
    print({'best_eval_latency': best_latency, 'best_eval_score': best_score})

if __name__ == '__main__':
    main()
