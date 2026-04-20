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
from src.models.mlp import MLP
from scripts._shared import save_checkpoint


def expand_action_conditioned_dataset(replay, num_nodes: int):
    xs = []
    ys = []
    gaps = []
    for r in replay:
        obs = np.asarray(r['obs'], dtype=np.float32)
        mask = np.asarray(r['mask'], dtype=np.float32)
        costs = np.asarray(r['action_costs'], dtype=np.float32)
        best = float(r.get('best_cost', 0.0))
        gap = float(r.get('gap_cost', 0.0))
        valid = np.where(np.isfinite(costs) & (mask > 0.5))[0]
        if len(valid) == 0:
            continue
        for a in valid:
            one_hot = np.zeros(num_nodes, dtype=np.float32)
            one_hot[int(a)] = 1.0
            xs.append(np.concatenate([obs, one_hot], axis=0))
            ys.append(float(np.log1p(float(costs[a]))))
            gaps.append(gap)
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32).reshape(-1, 1)
    g = np.asarray(gaps, dtype=np.float32).reshape(-1, 1)
    return x, y, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-config', required=True)
    ap.add_argument('--wm-config', required=True)
    ap.add_argument('--replay', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--gpu-id', type=int, default=2)
    args = ap.parse_args()
    env_cfg = load_yaml(args.env_config)
    cfg = load_yaml(args.wm_config)
    set_seed(int(env_cfg['seed']))
    torch.set_num_threads(1)
    device = resolve_device(args.device, args.gpu_id)
    print({'stage': 'scheduler_wm', 'device': describe_device(device)})
    replay = torch.load(args.replay, weights_only=False)
    num_nodes = int(env_cfg['num_nodes'])
    x_np, y_np, gap_np = expand_action_conditioned_dataset(replay, num_nodes)
    scale = max(float(np.quantile(y_np, 0.99)), 1e-6)
    gap_scale = max(float(np.quantile(gap_np, 0.95)), 1e-6)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy((y_np / scale).astype(np.float32))
    gap = torch.from_numpy(np.clip(gap_np / gap_scale, 0.0, 5.0).astype(np.float32))
    model = MLP(x.shape[1], 2, hidden=int(cfg['hidden'])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
    logger = MetricLogger('outputs/metrics/stage2_scheduler_wm.csv', 'outputs/logs/stage2_scheduler_wm.jsonl')
    best = math.inf
    global_step = 0
    bs = int(cfg['batch_size'])
    n = len(x)
    for epoch in range(int(cfg['epochs'])):
        for step in range(int(cfg['steps_per_epoch'])):
            idx = torch.randint(0, n, (bs,))
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            gb = gap[idx].to(device)
            pred = model(xb)
            cost_loss = nn.functional.mse_loss(pred[:, :1], yb)
            gap_loss = nn.functional.mse_loss(pred[:, 1:], gb)
            cont_loss = torch.tensor(0.0, device=device)
            kl = torch.tensor(0.0, device=device)
            loss = cost_loss + 0.25 * gap_loss
            opt.zero_grad(); loss.backward(); opt.step()
            global_step += 1
            row = {
                'global_step': global_step,
                'epoch': epoch,
                'step_in_epoch': step,
                'loss': float(loss.item()),
                'recon_loss': float(cost_loss.item()),
                'reward_loss': float(gap_loss.item()),
                'cont_loss': float(cont_loss.item()),
                'kl': float(kl.item()),
                'lr': float(opt.param_groups[0]['lr'])
            }
            logger.log(row)
            if loss.item() < best:
                best = loss.item()
                save_checkpoint(model, args.out, {
                    'best_loss': best,
                    'obs_dim': int(x.shape[1] - num_nodes),
                    'num_nodes': num_nodes,
                    'cost_scale': scale,
                    'gap_scale': gap_scale,
                })
            if global_step % int(cfg['log_every_steps']) == 0:
                print(row)
            if global_step % int(cfg['plot_every_steps']) == 0:
                plot_lines('outputs/metrics/stage2_scheduler_wm.csv', 'global_step', ['loss', 'recon_loss', 'reward_loss'], 'outputs/figures/stage2_scheduler_wm_loss.png', 'Stage2 scheduler WM', ma_window=25)
    logger.close()
    print({'best_loss': best})

if __name__ == '__main__':
    main()
