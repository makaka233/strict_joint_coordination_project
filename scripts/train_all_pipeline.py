from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse, subprocess, shutil
from pathlib import Path
from src.common.config import PROJECT_ROOT


def _run(cmd):
    print('[RUN]', ' '.join(map(str, cmd)))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--python', default='python')
    ap.add_argument('--train-env-config', required=True)
    ap.add_argument('--eval-env-config', required=True)
    ap.add_argument('--scheduler-collect-config', required=True)
    ap.add_argument('--scheduler-wm-config', required=True)
    ap.add_argument('--scheduler-actor-config', required=True)
    ap.add_argument('--deployment-wm-config', required=True)
    ap.add_argument('--deployment-actor-config', required=True)
    ap.add_argument('--deployment-data-episodes', type=int, default=1200)
    ap.add_argument('--eval-episodes', type=int, default=64)
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--reuse-existing', action='store_true')
    ap.add_argument('--joint-stage7-config', default='')
    ap.add_argument('--skip-joint-stage7', action='store_true')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--gpu-id', type=int, default=2)
    args = ap.parse_args()
    common_dev = ['--device', args.device, '--gpu-id', str(args.gpu_id)]
    if args.clean:
        for p in ['outputs/scheduler_data', 'outputs/scheduler_wm', 'outputs/scheduler_actor', 'outputs/deployment_data', 'outputs/deployment_wm', 'outputs/deployment_actor', 'outputs/joint_stage7', 'outputs/logs', 'outputs/metrics', 'outputs/figures']:
            pp = PROJECT_ROOT / p
            if pp.exists():
                shutil.rmtree(pp)
    print('='*80); print('Stage 1/7 - Collecting scheduler data'); print('='*80)
    if not args.reuse_existing or not (PROJECT_ROOT / 'outputs/scheduler_data/replay.pt').exists():
        _run([args.python, 'scripts/collect_scheduler_data.py', '--env-config', args.train_env_config, '--collect-config', args.scheduler_collect_config, '--out', 'outputs/scheduler_data/replay.pt'])
    print('='*80); print('Stage 2/7 - Training scheduler world model'); print('='*80)
    if not args.reuse_existing or not (PROJECT_ROOT / 'outputs/scheduler_wm/best.pt').exists():
        _run([args.python, 'scripts/train_scheduler_wm.py', '--env-config', args.train_env_config, '--wm-config', args.scheduler_wm_config, '--replay', 'outputs/scheduler_data/replay.pt', '--out', 'outputs/scheduler_wm/best.pt', *common_dev])
    print('='*80); print('Stage 3/7 - Training scheduler actor'); print('='*80)
    if not args.reuse_existing or not (PROJECT_ROOT / 'outputs/scheduler_actor/best.pt').exists():
        _run([args.python, 'scripts/train_scheduler_actor.py', '--env-config', args.train_env_config, '--actor-config', args.scheduler_actor_config, '--wm-checkpoint', 'outputs/scheduler_wm/best.pt', '--replay', 'outputs/scheduler_data/replay.pt', '--out', 'outputs/scheduler_actor/best.pt', *common_dev])
    print('='*80); print('Stage 4/7 - Collecting deployment data'); print('='*80)
    if not args.reuse_existing or not (PROJECT_ROOT / 'outputs/deployment_data/replay.pt').exists():
        _run([args.python, 'scripts/collect_deployment_data.py', '--env-config', args.train_env_config, '--scheduler-checkpoint', 'outputs/scheduler_actor/best.pt', '--out', 'outputs/deployment_data/replay.pt', '--episodes', str(args.deployment_data_episodes), *common_dev])
    print('='*80); print('Stage 5/7 - Training deployment world model'); print('='*80)
    if not args.reuse_existing or not (PROJECT_ROOT / 'outputs/deployment_wm/best.pt').exists():
        _run([args.python, 'scripts/train_deployment_wm.py', '--env-config', args.train_env_config, '--wm-config', args.deployment_wm_config, '--replay', 'outputs/deployment_data/replay.pt', '--out', 'outputs/deployment_wm/best.pt', *common_dev])
    print('='*80); print('Stage 6/7 - Training deployment actor'); print('='*80)
    if not args.reuse_existing or not (PROJECT_ROOT / 'outputs/deployment_actor/best.pt').exists():
        _run([args.python, 'scripts/train_deployment_actor.py', '--env-config', args.train_env_config, '--actor-config', args.deployment_actor_config, '--wm-checkpoint', 'outputs/deployment_wm/best.pt', '--scheduler-checkpoint', 'outputs/scheduler_actor/best.pt', '--replay', 'outputs/deployment_data/replay.pt', '--out', 'outputs/deployment_actor/best.pt', *common_dev])
    sched_ckpt = 'outputs/scheduler_actor/best.pt'
    dep_ckpt = 'outputs/deployment_actor/best.pt'
    if args.joint_stage7_config and not args.skip_joint_stage7:
        print('='*80); print('Stage 7/8 - Joint coordination fine-tuning'); print('='*80)
        _run([args.python, 'scripts/train_joint_stage7.py', '--env-config', args.train_env_config, '--joint-config', args.joint_stage7_config, '--scheduler-checkpoint', sched_ckpt, '--deployment-checkpoint', dep_ckpt, '--out-dir', 'outputs/joint_stage7', *common_dev])
        joint_sched = PROJECT_ROOT / 'outputs/joint_stage7/scheduler_joint_best.pt'
        joint_dep = PROJECT_ROOT / 'outputs/joint_stage7/deployment_joint_best.pt'
        if joint_sched.exists():
            sched_ckpt = 'outputs/joint_stage7/scheduler_joint_best.pt'
        if joint_dep.exists():
            dep_ckpt = 'outputs/joint_stage7/deployment_joint_best.pt'
        stage_lbl = 'Stage 8/8 - Final full-system evaluation'
    else:
        stage_lbl = 'Stage 7/7 - Final full-system evaluation'
    print('='*80); print(stage_lbl); print('='*80)
    _run([args.python, 'scripts/eval_joint_stage7.py', '--env-config', args.eval_env_config, '--scheduler-checkpoint', sched_ckpt, '--deployment-checkpoint', dep_ckpt, '--episodes', str(args.eval_episodes), '--out', 'outputs/eval/full_system_eval.csv', *common_dev])

if __name__ == '__main__':
    main()
