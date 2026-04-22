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
from src.env.core import build_scenario, generate_macro_obs, greedy_direct_deployment, init_workload_process, evaluate_deployment_with_scheduler, flatten_macro_obs, stage_count, make_scheduler_obs, repair_deployment
from src.agents.deployment.policy import DeploymentPolicy
from src.models.mlp import MLP
from scripts._shared import save_checkpoint, load_scheduler_policy


def eval_policy(dep_pol, sched_pol, scn, env_cfg, episodes=16, seed=777):
    rng = np.random.default_rng(seed)
    workload_state = init_workload_process(scn, env_cfg, rng)
    vals = []
    rewards = []
    for _ in range(episodes):
        macro = generate_macro_obs(scn, env_cfg, rng, workload_state=workload_state)
        obs = flatten_macro_obs(macro)
        x = dep_pol.act(obs, scn, int(env_cfg['max_replicas']))
        out = evaluate_deployment_with_scheduler(macro, x, scn, env_cfg, lambda obs, mask, task, local_stage, prev: sched_pol.act(obs, mask), workload_state=workload_state)
        vals.append(out['mean_window_latency'])
        rewards.append(out['total_reward'])
    return float(np.mean(vals)), float(np.mean(rewards))


def load_deployment_wm_model(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get('meta', {})
    obs_dim = int(meta.get('obs_dim'))
    num_outputs = int(meta.get('num_outputs'))
    hidden = int(meta.get('hidden', 128))
    model = MLP(obs_dim + num_outputs, 1, hidden=hidden).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, {
        'obs_dim': obs_dim,
        'num_outputs': num_outputs,
        'target': meta.get('target', ''),
        'hidden': hidden,
    }


def materialize_deployment_vector(raw_vec: np.ndarray, score_vec: np.ndarray, scn, max_replicas: int) -> np.ndarray:
    raw = np.asarray(raw_vec, dtype=np.float32).reshape(stage_count(scn), scn.num_nodes)
    scores = np.asarray(score_vec, dtype=np.float32).reshape(stage_count(scn), scn.num_nodes)
    repaired = repair_deployment(raw, scn, max_replicas, scores=scores)
    return repaired.reshape(-1).astype(np.float32)


def build_planner_candidate_pool(current_x: np.ndarray, label_x: np.ndarray, probs: np.ndarray, scn, max_replicas: int, sample_count: int, rng: np.random.Generator):
    batch, num_outputs = probs.shape
    total_candidates = 3 + sample_count
    candidates = np.zeros((batch, total_candidates, num_outputs), dtype=np.float32)
    for i in range(batch):
        cur = np.asarray(current_x[i], dtype=np.float32).reshape(-1)
        lab = np.asarray(label_x[i], dtype=np.float32).reshape(-1)
        prob = np.asarray(probs[i], dtype=np.float32).reshape(-1)
        candidates[i, 0] = materialize_deployment_vector(cur, cur, scn, max_replicas)
        candidates[i, 1] = materialize_deployment_vector(lab, lab, scn, max_replicas)
        candidates[i, 2] = materialize_deployment_vector((prob > 0.5).astype(np.float32), prob, scn, max_replicas)
        for j in range(sample_count):
            sampled = (rng.random(num_outputs) < prob).astype(np.float32)
            candidates[i, 3 + j] = materialize_deployment_vector(sampled, prob, scn, max_replicas)
    return candidates


def predict_candidate_latencies(wm_model: torch.nn.Module, macro_obs: np.ndarray, candidates: np.ndarray, device: torch.device) -> np.ndarray:
    batch, cand_count, num_outputs = candidates.shape
    obs_dim = macro_obs.shape[1]
    obs_rep = np.repeat(macro_obs[:, None, :], cand_count, axis=1).reshape(batch * cand_count, obs_dim)
    cand_flat = candidates.reshape(batch * cand_count, num_outputs)
    inp = torch.from_numpy(np.concatenate([obs_rep, cand_flat], axis=1).astype(np.float32)).to(device)
    with torch.no_grad():
        pred_log = wm_model(inp).reshape(batch, cand_count)
        pred_raw = torch.expm1(pred_log.clamp(min=0.0, max=20.0))
    return pred_raw.cpu().numpy()


def select_planner_targets(candidates: np.ndarray, predicted_latency: np.ndarray, replay_label: np.ndarray):
    best_idx = np.argmin(predicted_latency, axis=1)
    rows = np.arange(candidates.shape[0])
    planner_target = candidates[rows, best_idx].astype(np.float32)
    current_pred = predicted_latency[:, 0]
    best_pred = predicted_latency[rows, best_idx]
    relative_gain = np.maximum(0.0, (current_pred - best_pred) / np.maximum(current_pred, 1e-6)).astype(np.float32)
    label_agreement = float(np.mean(np.all(np.isclose(planner_target, replay_label, atol=1e-6), axis=1)))
    source_actor_rate = float(np.mean(best_idx >= 2))
    return planner_target, {
        'best_idx': best_idx.astype(np.int64),
        'current_pred': current_pred.astype(np.float32),
        'best_pred': best_pred.astype(np.float32),
        'relative_gain': relative_gain,
        'label_agreement': label_agreement,
        'source_actor_rate': source_actor_rate,
    }


def compute_original_actor_terms(logits: torch.Tensor, xb: torch.Tensor, xc: torch.Tensor, gb: torch.Tensor, swt: torch.Tensor, bce: nn.Module, entropy_floor: float, entropy_coef: float, stay_coef: float, progress: float):
    probs = torch.sigmoid(logits)
    bc_per = bce(logits, xb).mean(dim=1, keepdim=True)
    bc_weight = 0.5 + gb
    bc_loss = (bc_weight * bc_per).mean()
    entropy = -(probs * torch.log(probs + 1e-8) + (1-probs)*torch.log(1-probs + 1e-8)).mean()
    entropy_penalty = torch.relu(torch.tensor(entropy_floor, device=logits.device) - entropy)
    stay_loss = (probs - xc).abs().mean()
    bc_scale = 1.2 - 0.7 * progress
    policy_scale = 0.2 + 0.8 * progress
    actor_loss = bc_scale * bc_loss + policy_scale * (stay_coef * (1.0 - gb.mean()) * stay_loss) + entropy_coef * entropy_penalty + 0.05 * swt.mean()
    critic_loss = bc_loss.detach() * 100.0 + swt.mean().detach() * 10.0
    return {
        'probs': probs,
        'bc_loss': bc_loss,
        'entropy': entropy,
        'entropy_penalty': entropy_penalty,
        'stay_loss': stay_loss,
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
    }


def compute_wm_aux_loss(logits: torch.Tensor, planner_target: torch.Tensor | None, predicted_gain: torch.Tensor | None, bce: nn.Module, wm_loss_weight: float, wm_gain_floor: float, progress: float, wm_mode: str):
    zero = torch.tensor(0.0, device=logits.device)
    if wm_mode != 'planner' or planner_target is None or predicted_gain is None:
        return zero, zero, zero
    wm_bc_per = bce(logits, planner_target).mean(dim=1, keepdim=True)
    gain_weight = torch.where(predicted_gain > wm_gain_floor, predicted_gain, torch.zeros_like(predicted_gain))
    wm_bc_loss = wm_bc_per.mean()
    wm_aux_loss = wm_loss_weight * progress * (gain_weight * wm_bc_per).mean()
    return wm_bc_loss, wm_aux_loss, gain_weight.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-config', required=True)
    ap.add_argument('--actor-config', required=True)
    ap.add_argument('--wm-checkpoint', required=True)
    ap.add_argument('--scheduler-checkpoint', required=True)
    ap.add_argument('--replay', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--deployment-wm-mode', choices=['off', 'planner'], default='planner')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--gpu-id', type=int, default=2)
    args = ap.parse_args()
    env_cfg = load_yaml(args.env_config)
    cfg = load_yaml(args.actor_config)
    set_seed(int(env_cfg['seed']))
    torch.set_num_threads(1)
    device = resolve_device(args.device, args.gpu_id)
    print({'stage': 'deployment_actor', 'device': describe_device(device), 'deployment_wm_mode': args.deployment_wm_mode})
    scn = build_scenario(env_cfg)
    replay = torch.load(args.replay, weights_only=False)
    obs = torch.from_numpy(np.stack([r['macro_obs'] for r in replay]).astype('float32'))
    x_label = torch.from_numpy(np.stack([r['x_label'].reshape(-1) for r in replay]).astype('float32'))
    current = torch.from_numpy(np.stack([r['current_x'].reshape(-1) for r in replay]).astype('float32'))
    rel_gain = torch.tensor([r.get('relative_gain', 0.0) for r in replay], dtype=torch.float32).unsqueeze(1)
    switch_target = torch.tensor([r.get('switch_cost_est', 0.0) for r in replay], dtype=torch.float32).unsqueeze(1)
    num_outputs = x_label.shape[1]
    dep_pol = DeploymentPolicy(obs.shape[1], num_outputs, hidden=int(cfg['hidden']), device=device)
    max_replicas = int(env_cfg['max_replicas'])
    planner_sample_count = int(cfg.get('wm_candidate_samples', 3))
    wm_loss_weight = float(cfg.get('wm_loss_weight', 0.70))
    wm_gain_floor = float(cfg.get('wm_gain_floor', 0.02))
    dummy_rng = np.random.default_rng(int(env_cfg['seed']) + 1)
    dummy_workload_state = init_workload_process(scn, env_cfg, dummy_rng)
    dummy_macro = generate_macro_obs(scn, env_cfg, dummy_rng, workload_state=dummy_workload_state)
    dummy_dep = greedy_direct_deployment(dummy_macro, scn, max_replicas)
    t = {'origin': 0, 'service': 0, 'stage_compute': [1.0] * scn.service_stages[0], 'stage_data': [1.0] * scn.service_stages[0]}
    sched_obs_dim = make_scheduler_obs(t, 0, 0, dummy_dep, dummy_macro, scn, np.zeros(scn.num_nodes, dtype=np.float32)).shape[0]
    sched_pol = load_scheduler_policy(args.scheduler_checkpoint, sched_obs_dim, scn.num_nodes, device=device)
    wm_model = None
    if args.deployment_wm_mode == 'planner':
        wm_model, wm_meta = load_deployment_wm_model(args.wm_checkpoint, device)
        if wm_meta['obs_dim'] != obs.shape[1] or wm_meta['num_outputs'] != num_outputs:
            raise ValueError(f'deployment WM shape mismatch: expected obs_dim={obs.shape[1]}, num_outputs={num_outputs}, got {wm_meta}')
    opt = torch.optim.Adam(dep_pol.model.parameters(), lr=float(cfg['lr']))
    logger = MetricLogger('outputs/metrics/stage6_deployment_actor.csv', 'outputs/logs/stage6_deployment_actor.jsonl')
    bce = nn.BCEWithLogitsLoss(reduction='none')
    best = math.inf
    global_step = 0
    bs = int(cfg['batch_size'])
    n = len(replay)
    entropy_floor = float(cfg.get('entropy_floor', 0.02))
    entropy_coef = float(cfg.get('entropy_coef', 0.002))
    stay_coef = float(cfg.get('stay_coef', 0.2))
    bc_anchor_steps = int(cfg.get('bc_anchor_steps', max(1, int(cfg['epochs']) * int(cfg['steps_per_epoch']) // 3)))
    planner_rng = np.random.default_rng(int(env_cfg['seed']) + 6060)
    for epoch in range(int(cfg['epochs'])):
        for step in range(int(cfg['steps_per_epoch'])):
            idx = torch.randint(0, n, (bs,))
            ob = obs[idx].to(device)
            xb = x_label[idx].to(device)
            xc = current[idx].to(device)
            gb = rel_gain[idx].to(device).clamp(0.0, 1.0)
            swt = switch_target[idx].to(device)
            logits = dep_pol.model(ob)
            progress = min(1.0, global_step / max(1, bc_anchor_steps))
            base_terms = compute_original_actor_terms(logits, xb, xc, gb, swt, bce, entropy_floor, entropy_coef, stay_coef, progress)
            actor_loss = base_terms['actor_loss']
            critic_loss = base_terms['critic_loss']
            wm_bc_loss = torch.tensor(0.0, device=device)
            wm_aux_loss = torch.tensor(0.0, device=device)
            wm_gain_weight_mean = torch.tensor(0.0, device=device)
            wm_pred_current_latency = 0.0
            wm_pred_best_latency = 0.0
            wm_pred_relative_gain = 0.0
            planner_label_agreement = 0.0
            planner_source_actor_rate = 0.0
            if args.deployment_wm_mode == 'planner':
                candidate_pool = build_planner_candidate_pool(
                    xc.detach().cpu().numpy(),
                    xb.detach().cpu().numpy(),
                    base_terms['probs'].detach().cpu().numpy(),
                    scn,
                    max_replicas,
                    planner_sample_count,
                    planner_rng,
                )
                pred_latency = predict_candidate_latencies(wm_model, ob.detach().cpu().numpy(), candidate_pool, device)
                planner_target_np, planner_stats = select_planner_targets(candidate_pool, pred_latency, xb.detach().cpu().numpy())
                planner_target = torch.from_numpy(planner_target_np).to(device)
                pred_gain = torch.from_numpy(planner_stats['relative_gain']).to(device).unsqueeze(1)
                wm_bc_loss, wm_aux_loss, wm_gain_weight_mean = compute_wm_aux_loss(
                    logits,
                    planner_target,
                    pred_gain,
                    bce,
                    wm_loss_weight,
                    wm_gain_floor,
                    progress,
                    args.deployment_wm_mode,
                )
                actor_loss = actor_loss + wm_aux_loss
                wm_pred_current_latency = float(np.mean(planner_stats['current_pred']))
                wm_pred_best_latency = float(np.mean(planner_stats['best_pred']))
                wm_pred_relative_gain = float(np.mean(planner_stats['relative_gain']))
                planner_label_agreement = float(planner_stats['label_agreement'])
                planner_source_actor_rate = float(planner_stats['source_actor_rate'])
            opt.zero_grad()
            actor_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(dep_pol.model.parameters(), 5.0).item())
            opt.step()
            global_step += 1
            row = {
                'global_step': global_step,
                'epoch': epoch,
                'step_in_epoch': step,
                'deployment_wm_mode': args.deployment_wm_mode,
                'actor_loss': float(actor_loss.item()),
                'critic_loss': float(critic_loss.item()),
                'imag_return': float((gb.mean() - base_terms['stay_loss']).item()),
                'entropy': float(base_terms['entropy'].item()),
                'entropy_penalty': float(base_terms['entropy_penalty'].item()),
                'bc_loss': float(base_terms['bc_loss'].item()),
                'switch_cost': float(base_terms['stay_loss'].item()),
                'relative_gain_mean': float(gb.mean().item()),
                'target_switch_cost': float(swt.mean().item()),
                'wm_bc_loss': float(wm_bc_loss.item()),
                'wm_aux_loss': float(wm_aux_loss.item()),
                'wm_gain_weight_mean': float(wm_gain_weight_mean.item()),
                'wm_pred_current_latency': wm_pred_current_latency,
                'wm_pred_best_latency': wm_pred_best_latency,
                'wm_pred_relative_gain': wm_pred_relative_gain,
                'planner_label_agreement': planner_label_agreement,
                'planner_source_actor_rate': planner_source_actor_rate,
                'grad_norm': grad_norm,
                'lr': float(cfg['lr']),
            }
            logger.log(row)
            if global_step % int(cfg['log_every_steps']) == 0:
                print(row)
            if global_step % int(cfg['short_eval_every_steps']) == 0:
                eval_lat, eval_reward = eval_policy(dep_pol, sched_pol, scn, env_cfg, episodes=int(cfg.get('short_eval_episodes', 10)), seed=int(env_cfg['seed']) + global_step)
                eval_row = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'step_in_epoch': step,
                    'eval_latency_short': eval_lat,
                    'eval_reward_short': eval_reward,
                    'best_eval_latency_so_far': min(best, eval_lat),
                    'is_best_checkpoint': float(eval_lat < best),
                }
                logger.log(eval_row)
                if eval_lat < best:
                    best = eval_lat
                    save_checkpoint(dep_pol.model, args.out, {
                        'best_eval_latency': best,
                        'obs_dim': obs.shape[1],
                        'num_outputs': num_outputs,
                        'deployment_wm_mode': args.deployment_wm_mode,
                    })
                    print({'best_eval_latency': best, 'global_step': global_step})
            if global_step % int(cfg['plot_every_steps']) == 0:
                plot_lines('outputs/metrics/stage6_deployment_actor.csv', 'global_step', ['actor_loss', 'bc_loss', 'wm_bc_loss', 'switch_cost'], 'outputs/figures/stage6_deployment_actor_train.png', 'Stage6 deployment actor train', ma_window=50)
                plot_lines('outputs/metrics/stage6_deployment_actor.csv', 'global_step', ['eval_latency_short'], 'outputs/figures/stage6_deployment_actor_eval_latency.png', 'Stage6 deployment eval latency', ma_window=5)
                plot_lines('outputs/metrics/stage6_deployment_actor.csv', 'global_step', ['eval_reward_short'], 'outputs/figures/stage6_deployment_actor_reward.png', 'Stage6 deployment reward', ma_window=5)
    logger.close()
    print({'best_eval_latency': best})


if __name__ == '__main__':
    main()
