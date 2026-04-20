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


def build_candidate_conditioned_dataset(replay):
    xs = []
    y_log = []
    y_raw = []
    episode_slices = []
    best_candidate_idx = []
    obs_dim = 0
    num_outputs = 0
    cursor = 0
    for row in replay:
        obs = np.asarray(row['macro_obs'], dtype=np.float32).reshape(-1)
        cand_xs = row.get('candidate_xs')
        cand_log = row.get('candidate_latency_norms')
        cand_raw = row.get('candidate_latencies')
        if cand_xs is None or cand_log is None or cand_raw is None:
            best_x = np.asarray(row['x_label'], dtype=np.float32).reshape(-1)
            best_raw = float(row.get('best_latency', np.expm1(float(row['best_latency_norm']))))
            cand_xs = best_x.reshape(1, -1)
            cand_raw = np.asarray([best_raw], dtype=np.float32)
            cand_log = np.log1p(cand_raw).astype(np.float32)
            best_idx = 0
        else:
            cand_xs = np.asarray(cand_xs, dtype=np.float32)
            cand_log = np.asarray(cand_log, dtype=np.float32).reshape(-1)
            cand_raw = np.asarray(cand_raw, dtype=np.float32).reshape(-1)
            best_idx = int(row.get('best_candidate_idx', int(np.argmin(cand_raw))))
        obs_dim = int(obs.shape[0])
        num_outputs = int(cand_xs.shape[1])
        start = cursor
        for cand, cand_log_v, cand_raw_v in zip(cand_xs, cand_log, cand_raw):
            xs.append(np.concatenate([obs, cand.astype(np.float32)], axis=0))
            y_log.append(float(cand_log_v))
            y_raw.append(float(cand_raw_v))
            cursor += 1
        episode_slices.append((start, cursor))
        best_candidate_idx.append(best_idx)
    return {
        'x': np.asarray(xs, dtype=np.float32),
        'y_log': np.asarray(y_log, dtype=np.float32).reshape(-1, 1),
        'y_raw': np.asarray(y_raw, dtype=np.float32).reshape(-1, 1),
        'episode_slices': episode_slices,
        'best_candidate_idx': best_candidate_idx,
        'obs_dim': obs_dim,
        'num_outputs': num_outputs,
    }


def split_episode_indices(num_episodes: int, holdout_fraction: float, seed: int):
    if num_episodes <= 1 or holdout_fraction <= 0.0:
        return list(range(num_episodes)), []
    holdout_count = min(num_episodes - 1, max(1, int(round(num_episodes * holdout_fraction))))
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_episodes).tolist()
    holdout = sorted(order[:holdout_count])
    train = sorted(order[holdout_count:])
    return train, holdout


def predict_dataset(model: torch.nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 2048):
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            preds.append(model(xb).cpu())
    return torch.cat(preds, dim=0).numpy().reshape(-1)


def pairwise_ranking_accuracy(preds: np.ndarray, truth: np.ndarray) -> float:
    if len(preds) <= 1:
        return 1.0
    correct = 0
    total = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            true_diff = truth[i] - truth[j]
            if abs(true_diff) <= 1e-8:
                continue
            pred_diff = preds[i] - preds[j]
            correct += float((true_diff < 0 and pred_diff < 0) or (true_diff > 0 and pred_diff > 0))
            total += 1
    return float(correct / total) if total else 1.0


def evaluate_predictions(pred_log: np.ndarray, dataset: dict, episode_indices: list[int]):
    if not episode_indices:
        return {
            'mse': 0.0,
            'mae': 0.0,
            'log_mse': 0.0,
            'log_mae': 0.0,
            'ranking_accuracy': 0.0,
            'best_candidate_hit_rate': 0.0,
            'sample_count': 0,
            'episode_count': 0,
        }
    pred_log = np.asarray(pred_log, dtype=np.float32).reshape(-1)
    pred_raw = np.expm1(np.clip(pred_log, a_min=0.0, a_max=20.0)).astype(np.float32)
    true_log = dataset['y_log'].reshape(-1)
    true_raw = dataset['y_raw'].reshape(-1)
    sample_indices = []
    ranking_scores = []
    hit_scores = []
    for ep in episode_indices:
        start, end = dataset['episode_slices'][ep]
        sample_indices.extend(range(start, end))
        pred_ep = pred_raw[start:end]
        true_ep = true_raw[start:end]
        ranking_scores.append(pairwise_ranking_accuracy(pred_ep, true_ep))
        hit_scores.append(float(int(np.argmin(pred_ep)) == int(dataset['best_candidate_idx'][ep])))
    idx = np.asarray(sample_indices, dtype=np.int64)
    return {
        'mse': float(np.mean((pred_raw[idx] - true_raw[idx]) ** 2)),
        'mae': float(np.mean(np.abs(pred_raw[idx] - true_raw[idx]))),
        'log_mse': float(np.mean((pred_log[idx] - true_log[idx]) ** 2)),
        'log_mae': float(np.mean(np.abs(pred_log[idx] - true_log[idx]))),
        'ranking_accuracy': float(np.mean(ranking_scores)) if ranking_scores else 0.0,
        'best_candidate_hit_rate': float(np.mean(hit_scores)) if hit_scores else 0.0,
        'sample_count': int(len(idx)),
        'episode_count': int(len(episode_indices)),
    }


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
    dataset = build_candidate_conditioned_dataset(replay)
    x = torch.from_numpy(dataset['x'])
    y_log = torch.from_numpy(dataset['y_log'])
    train_eps, holdout_eps = split_episode_indices(
        len(dataset['episode_slices']),
        float(cfg.get('holdout_fraction', 0.20)),
        int(cfg.get('holdout_seed', int(env_cfg['seed']) + 22040)),
    )
    train_sample_idx = np.concatenate([
        np.arange(*dataset['episode_slices'][ep], dtype=np.int64)
        for ep in train_eps
    ]) if train_eps else np.arange(len(dataset['x']), dtype=np.int64)
    train_sample_idx_t = torch.from_numpy(train_sample_idx)
    model = MLP(x.shape[1], 1, hidden=int(cfg['hidden'])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
    logger = MetricLogger('outputs/metrics/stage5_deployment_wm.csv', 'outputs/logs/stage5_deployment_wm.jsonl')
    best = math.inf
    global_step = 0
    bs = int(cfg['batch_size'])
    grad_clip = float(cfg.get('grad_clip', 5.0))
    eval_every = int(cfg.get('eval_every_steps', cfg.get('plot_every_steps', 180)))
    for epoch in range(int(cfg['epochs'])):
        for step in range(int(cfg['steps_per_epoch'])):
            rand_pos = torch.randint(0, len(train_sample_idx_t), (bs,))
            idx = train_sample_idx_t[rand_pos]
            xb = x[idx].to(device)
            yb = y_log[idx].to(device)
            pred = model(xb)
            recon_loss = nn.functional.mse_loss(pred, yb)
            loss = recon_loss
            opt.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
            opt.step()
            global_step += 1
            row = {
                'global_step': global_step,
                'epoch': epoch,
                'step_in_epoch': step,
                'loss': float(loss.item()),
                'recon_loss': float(recon_loss.item()),
                'target_mean': float(yb.mean().item()),
                'pred_mean': float(pred.mean().item()),
                'grad_norm': grad_norm,
                'lr': float(cfg['lr']),
            }
            logger.log(row)
            if global_step % int(cfg['log_every_steps']) == 0:
                print(row)
            if global_step % eval_every == 0:
                pred_log_all = predict_dataset(model, x, device)
                train_metrics = evaluate_predictions(pred_log_all, dataset, train_eps)
                holdout_metrics = evaluate_predictions(pred_log_all, dataset, holdout_eps)
                eval_row = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'step_in_epoch': step,
                    'train_mse': train_metrics['mse'],
                    'train_mae': train_metrics['mae'],
                    'train_log_mse': train_metrics['log_mse'],
                    'train_log_mae': train_metrics['log_mae'],
                    'train_ranking_accuracy': train_metrics['ranking_accuracy'],
                    'train_best_candidate_hit_rate': train_metrics['best_candidate_hit_rate'],
                    'holdout_mse': holdout_metrics['mse'],
                    'holdout_mae': holdout_metrics['mae'],
                    'holdout_log_mse': holdout_metrics['log_mse'],
                    'holdout_log_mae': holdout_metrics['log_mae'],
                    'holdout_ranking_accuracy': holdout_metrics['ranking_accuracy'],
                    'holdout_best_candidate_hit_rate': holdout_metrics['best_candidate_hit_rate'],
                    'train_episode_count': train_metrics['episode_count'],
                    'holdout_episode_count': holdout_metrics['episode_count'],
                }
                logger.log(eval_row)
                select_metric = eval_row['holdout_mse'] if holdout_eps else eval_row['train_mse']
                if select_metric < best:
                    best = select_metric
                    save_checkpoint(model, args.out, {
                        'best_loss': best,
                        'obs_dim': dataset['obs_dim'],
                        'num_outputs': dataset['num_outputs'],
                        'target': 'log1p_latency',
                    })
                print({k: eval_row[k] for k in ['global_step', 'train_mse', 'holdout_mse', 'train_ranking_accuracy', 'holdout_ranking_accuracy', 'holdout_best_candidate_hit_rate']})
            if global_step % int(cfg['plot_every_steps']) == 0:
                plot_lines('outputs/metrics/stage5_deployment_wm.csv', 'global_step', ['loss', 'train_mse', 'holdout_mse'], 'outputs/figures/stage5_deployment_wm_loss.png', 'Stage5 deployment WM', ma_window=25)
                plot_lines('outputs/metrics/stage5_deployment_wm.csv', 'global_step', ['train_ranking_accuracy', 'holdout_ranking_accuracy', 'holdout_best_candidate_hit_rate'], 'outputs/figures/stage5_deployment_wm_ranking.png', 'Stage5 deployment WM ranking', ma_window=10)
    logger.close()
    print({'best_loss': best})


if __name__ == '__main__':
    main()
