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
    print({'stage': 'deployment_wm', 'device': describe_device(device)})
    replay = torch.load(args.replay, weights_only=False)
    obs = torch.from_numpy(np.stack([r['macro_obs'] for r in replay]).astype('float32'))
    x_label = torch.from_numpy(np.stack([r['x_label'].reshape(-1) for r in replay]).astype('float32'))
    # normalized target to stabilize world model
    y = torch.tensor([[r.get('best_latency_norm', float(np.log1p(r['best_latency'])))] for r in replay], dtype=torch.float32)
    x = torch.cat([obs, x_label], dim=1)
    model = MLP(x.shape[1], 1, hidden=int(cfg['hidden'])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
    logger = MetricLogger('outputs/metrics/stage5_deployment_wm.csv', 'outputs/logs/stage5_deployment_wm.jsonl')
    best = math.inf
    global_step = 0
    bs = int(cfg['batch_size'])
    n = len(replay)
    for epoch in range(int(cfg['epochs'])):
        for step in range(int(cfg['steps_per_epoch'])):
            idx = torch.randint(0, n, (bs,))
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            pred = model(xb)
            recon_loss = nn.functional.mse_loss(pred, yb)
            reward_loss = recon_loss
            cont_loss = torch.tensor(0.0, device=device)
            kl = torch.tensor(0.0, device=device)
            loss = recon_loss + reward_loss
            opt.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
            opt.step()
            global_step += 1
            row = {
                'global_step': global_step,
                'epoch': epoch,
                'step_in_epoch': step,
                'loss': float(loss.item()),
                'recon_loss': float(recon_loss.item()),
                'reward_loss': float(reward_loss.item()),
                'cont_loss': float(cont_loss.item()),
                'kl': float(kl.item()),
                'target_mean': float(yb.mean().item()),
                'pred_mean': float(pred.mean().item()),
                'grad_norm': grad_norm,
                'lr': float(cfg['lr'])
            }
            logger.log(row)
            if loss.item() < best:
                best = loss.item()
                save_checkpoint(model, args.out, {'best_loss': best, 'target': 'log1p_latency'})
            if global_step % int(cfg['log_every_steps']) == 0:
                print(row)
            if global_step % int(cfg['plot_every_steps']) == 0:
                plot_lines('outputs/metrics/stage5_deployment_wm.csv', 'global_step', ['loss', 'recon_loss', 'reward_loss'], 'outputs/figures/stage5_deployment_wm_loss.png', 'Stage5 deployment WM', ma_window=25)
    logger.close()
    print({'best_loss': best})

if __name__ == '__main__':
    main()
