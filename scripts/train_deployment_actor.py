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
from src.env.core import build_scenario, generate_macro_obs, greedy_direct_deployment, evaluate_deployment_with_scheduler, flatten_macro_obs, stage_count, make_scheduler_obs
from src.agents.deployment.policy import DeploymentPolicy
from scripts._shared import save_checkpoint, load_scheduler_policy


def eval_policy(dep_pol, sched_pol, scn, env_cfg, episodes=16, seed=777):
    rng = np.random.default_rng(seed)
    vals = []
    rewards = []
    for _ in range(episodes):
        macro = generate_macro_obs(scn, env_cfg, rng)
        obs = flatten_macro_obs(macro)
        x = dep_pol.act(obs, scn, int(env_cfg['max_replicas']))
        out = evaluate_deployment_with_scheduler(macro, x, scn, env_cfg, lambda obs, mask, task, local_stage, prev: sched_pol.act(obs, mask))
        vals.append(out['mean_window_latency'])
        rewards.append(out['total_reward'])
    return float(np.mean(vals)), float(np.mean(rewards))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-config', required=True)
    ap.add_argument('--actor-config', required=True)
    ap.add_argument('--wm-checkpoint', required=True)
    ap.add_argument('--scheduler-checkpoint', required=True)
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
    print({'stage': 'deployment_actor', 'device': describe_device(device)})
    scn = build_scenario(env_cfg)
    replay = torch.load(args.replay, weights_only=False)
    obs = torch.from_numpy(np.stack([r['macro_obs'] for r in replay]).astype('float32'))
    x_label = torch.from_numpy(np.stack([r['x_label'].reshape(-1) for r in replay]).astype('float32'))
    current = torch.from_numpy(np.stack([r['current_x'].reshape(-1) for r in replay]).astype('float32'))
    rel_gain = torch.tensor([r.get('relative_gain', 0.0) for r in replay], dtype=torch.float32).unsqueeze(1)
    switch_target = torch.tensor([r.get('switch_cost_est', 0.0) for r in replay], dtype=torch.float32).unsqueeze(1)
    num_outputs = x_label.shape[1]
    dep_pol = DeploymentPolicy(obs.shape[1], num_outputs, hidden=int(cfg['hidden']), device=device)
    dummy_macro = generate_macro_obs(scn, env_cfg, np.random.default_rng(int(env_cfg['seed'])+1))
    dummy_dep = greedy_direct_deployment(dummy_macro, scn, int(env_cfg['max_replicas']))
    t = {'origin': 0, 'service': 0, 'stage_compute': [1.0]*scn.service_stages[0], 'stage_data':[1.0]*scn.service_stages[0]}
    sched_obs_dim = make_scheduler_obs(t, 0, 0, dummy_dep, dummy_macro, scn, np.zeros(scn.num_nodes, dtype=np.float32)).shape[0]
    sched_pol = load_scheduler_policy(args.scheduler_checkpoint, sched_obs_dim, scn.num_nodes, device=device)
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
    for epoch in range(int(cfg['epochs'])):
        for step in range(int(cfg['steps_per_epoch'])):
            idx = torch.randint(0, n, (bs,))
            ob = obs[idx].to(device)
            xb = x_label[idx].to(device)
            xc = current[idx].to(device)
            gb = rel_gain[idx].to(device).clamp(0.0, 1.0)
            swt = switch_target[idx].to(device)
            logits = dep_pol.model(ob)
            probs = torch.sigmoid(logits)
            # weighted behavior cloning: stronger when candidate clearly beats current
            bc_per = bce(logits, xb).mean(dim=1, keepdim=True)
            bc_weight = 0.5 + gb
            bc_loss = (bc_weight * bc_per).mean()
            entropy = -(probs * torch.log(probs + 1e-8) + (1-probs)*torch.log(1-probs + 1e-8)).mean()
            entropy_penalty = torch.relu(torch.tensor(entropy_floor, device=device) - entropy)
            stay_loss = (probs - xc).abs().mean()
            # optimistic improvement proxy: move toward label if there is gain, otherwise stay close to current
            progress = min(1.0, global_step / max(1, bc_anchor_steps))
            bc_scale = 1.2 - 0.7 * progress
            policy_scale = 0.2 + 0.8 * progress
            actor_loss = bc_scale * bc_loss + policy_scale * (stay_coef * (1.0 - gb.mean()) * stay_loss) + entropy_coef * entropy_penalty + 0.05 * swt.mean()
            critic_loss = bc_loss.detach() * 100.0 + swt.mean().detach() * 10.0
            opt.zero_grad()
            actor_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(dep_pol.model.parameters(), 5.0).item())
            opt.step()
            global_step += 1
            row = {
                'global_step': global_step,
                'epoch': epoch,
                'step_in_epoch': step,
                'actor_loss': float(actor_loss.item()),
                'critic_loss': float(critic_loss.item()),
                'imag_return': float((gb.mean() - stay_loss).item()),
                'entropy': float(entropy.item()),
                'entropy_penalty': float(entropy_penalty.item()),
                'bc_loss': float(bc_loss.item()),
                'switch_cost': float(stay_loss.item()),
                'relative_gain_mean': float(gb.mean().item()),
                'target_switch_cost': float(swt.mean().item()),
                'grad_norm': grad_norm,
                'lr': float(cfg['lr'])
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
                    'is_best_checkpoint': float(eval_lat < best)
                }
                logger.log(eval_row)
                if eval_lat < best:
                    best = eval_lat
                    save_checkpoint(dep_pol.model, args.out, {'best_eval_latency': best, 'obs_dim': obs.shape[1], 'num_outputs': num_outputs})
                    print({'best_eval_latency': best, 'global_step': global_step})
            if global_step % int(cfg['plot_every_steps']) == 0:
                plot_lines('outputs/metrics/stage6_deployment_actor.csv', 'global_step', ['actor_loss', 'bc_loss', 'switch_cost'], 'outputs/figures/stage6_deployment_actor_train.png', 'Stage6 deployment actor train', ma_window=50)
                plot_lines('outputs/metrics/stage6_deployment_actor.csv', 'global_step', ['eval_latency_short'], 'outputs/figures/stage6_deployment_actor_eval_latency.png', 'Stage6 deployment eval latency', ma_window=5)
                plot_lines('outputs/metrics/stage6_deployment_actor.csv', 'global_step', ['imag_return', 'eval_reward_short'], 'outputs/figures/stage6_deployment_actor_reward.png', 'Stage6 deployment reward', ma_window=5)
    logger.close()
    print({'best_eval_latency': best})

if __name__ == '__main__':
    main()
