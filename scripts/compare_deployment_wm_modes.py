from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse, csv, json, subprocess
import torch


def _run(cmd):
    print('[RUN]', ' '.join(map(str, cmd)))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _load_json(path: str | Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_best_eval_latency(path: str | Path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return float(ckpt.get('meta', {}).get('best_eval_latency', 0.0))


def _relative_improvement(metric: str, baseline: float, planner: float) -> float:
    denom = max(abs(baseline), 1e-6)
    lower_better = {'best_stage6_short_eval_latency', 'mean_latency', 'p90_latency', 'worst_latency', 'deployment_switch_proxy'}
    if metric in lower_better:
        return float((baseline - planner) / denom)
    if metric == 'mean_reward':
        return float((planner - baseline) / denom)
    return float((planner - baseline) / denom)


def maybe_run(cmd, out_path: Path, reuse_existing: bool):
    if reuse_existing and out_path.exists():
        print({'reuse_existing': str(out_path)})
        return
    _run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--python', default='python')
    ap.add_argument('--env-config', required=True)
    ap.add_argument('--actor-config', required=True)
    ap.add_argument('--wm-checkpoint', required=True)
    ap.add_argument('--scheduler-checkpoint', required=True)
    ap.add_argument('--replay', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--eval-episodes', type=int, default=64)
    ap.add_argument('--seed-base', type=int, default=52000)
    ap.add_argument('--seed-count', type=int, default=4)
    ap.add_argument('--reuse-existing', action='store_true')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--gpu-id', type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_ckpt = out_dir / 'deployment_actor_off.pt'
    planner_ckpt = out_dir / 'deployment_actor_planner.pt'
    baseline_eval = out_dir / 'full_eval_off.csv'
    planner_eval = out_dir / 'full_eval_planner.csv'
    baseline_summary = out_dir / 'summary_off.json'
    planner_summary = out_dir / 'summary_planner.json'

    common_dev = ['--device', args.device, '--gpu-id', str(args.gpu_id)]
    train_common = [
        '--env-config', args.env_config,
        '--actor-config', args.actor_config,
        '--wm-checkpoint', args.wm_checkpoint,
        '--scheduler-checkpoint', args.scheduler_checkpoint,
        '--replay', args.replay,
        *common_dev,
    ]

    maybe_run([
        args.python, 'scripts/train_deployment_actor.py',
        *train_common,
        '--deployment-wm-mode', 'off',
        '--out', str(baseline_ckpt),
    ], baseline_ckpt, args.reuse_existing)

    maybe_run([
        args.python, 'scripts/train_deployment_actor.py',
        *train_common,
        '--deployment-wm-mode', 'planner',
        '--out', str(planner_ckpt),
    ], planner_ckpt, args.reuse_existing)

    maybe_run([
        args.python, 'scripts/run_full_system.py',
        '--env-config', args.env_config,
        '--scheduler-checkpoint', args.scheduler_checkpoint,
        '--deployment-checkpoint', str(baseline_ckpt),
        '--episodes', str(args.eval_episodes),
        '--seed-base', str(args.seed_base),
        '--seed-count', str(args.seed_count),
        '--out', str(baseline_eval),
        '--summary-out', str(baseline_summary),
        *common_dev,
    ], baseline_summary, args.reuse_existing)

    maybe_run([
        args.python, 'scripts/run_full_system.py',
        '--env-config', args.env_config,
        '--scheduler-checkpoint', args.scheduler_checkpoint,
        '--deployment-checkpoint', str(planner_ckpt),
        '--episodes', str(args.eval_episodes),
        '--seed-base', str(args.seed_base),
        '--seed-count', str(args.seed_count),
        '--out', str(planner_eval),
        '--summary-out', str(planner_summary),
        *common_dev,
    ], planner_summary, args.reuse_existing)

    baseline = _load_json(baseline_summary)
    planner = _load_json(planner_summary)
    baseline['best_stage6_short_eval_latency'] = _load_best_eval_latency(baseline_ckpt)
    planner['best_stage6_short_eval_latency'] = _load_best_eval_latency(planner_ckpt)

    metrics = [
        'best_stage6_short_eval_latency',
        'mean_latency',
        'p90_latency',
        'worst_latency',
        'mean_reward',
        'deployment_density',
        'deployment_switch_proxy',
    ]
    rows = []
    comparison = {}
    for metric in metrics:
        b = float(baseline.get(metric, 0.0))
        p = float(planner.get(metric, 0.0))
        delta = p - b
        rel = _relative_improvement(metric, b, p)
        row = {
            'metric': metric,
            'baseline': b,
            'planner': p,
            'absolute_delta': delta,
            'relative_improvement': rel,
        }
        rows.append(row)
        comparison[metric] = row

    csv_path = out_dir / 'comparison_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / 'comparison_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': baseline,
            'planner': planner,
            'comparison': comparison,
            'artifacts': {
                'baseline_checkpoint': str(baseline_ckpt),
                'planner_checkpoint': str(planner_ckpt),
                'baseline_eval_csv': str(baseline_eval),
                'planner_eval_csv': str(planner_eval),
            },
        }, f, ensure_ascii=False, indent=2)

    print({
        'baseline_checkpoint': str(baseline_ckpt),
        'planner_checkpoint': str(planner_ckpt),
        'comparison_summary_csv': str(csv_path),
        'comparison_summary_json': str(json_path),
    })


if __name__ == '__main__':
    main()
