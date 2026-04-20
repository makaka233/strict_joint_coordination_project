# Stage 7 Joint Coordination Fine-Tuning

This project extends the verified WM-driven scheduler baseline with a new **Stage 7** coordination fine-tuning phase.

## What is added
- `configs/joint/stage7_validated.yaml`
- `configs/joint/stage7_large.yaml`
- `scripts/train_joint_stage7.py`
- `scripts/eval_joint_stage7.py`
- `scripts/train_all_pipeline.py` integration via `--joint-stage7-config`

## Design
Stage 7 alternates updates with **deployment frozen / scheduler updated** and **scheduler frozen / deployment updated**. Each cycle is accepted only if a fixed seed-bank joint evaluation improves a weighted latency score.

## Outputs
Stage 7 writes:
- `outputs/joint_stage7/stage7_joint_train.csv`
- `outputs/joint_stage7/stage7_joint_train.jsonl`
- `outputs/joint_stage7/scheduler_joint_best.pt`
- `outputs/joint_stage7/deployment_joint_best.pt`
- `outputs/figures/stage7_joint_latency.png`
- `outputs/figures/stage7_joint_losses.png`

## Recommended usage
Validated:
```powershell
python .\scripts\train_all_pipeline.py `
  --train-env-config .\configs\env\validated.yaml `
  --eval-env-config .\configs\env\validated.yaml `
  --scheduler-collect-config .\configs\scheduler\collect_validated.yaml `
  --scheduler-wm-config .\configs\scheduler\wm_validated.yaml `
  --scheduler-actor-config .\configs\scheduler\actor_validated.yaml `
  --deployment-wm-config .\configs\deployment\wm_validated.yaml `
  --deployment-actor-config .\configs\deployment\actor_validated.yaml `
  --joint-stage7-config .\configs\joint\stage7_validated.yaml `
  --deployment-data-episodes 60 `
  --eval-episodes 20 `
  --gpu-id 2 --device auto --clean
```

Large:
```powershell
python .\scripts\train_all_pipeline.py `
  --train-env-config .\configs\env\large_train.yaml `
  --eval-env-config .\configs\env\large_train.yaml `
  --scheduler-collect-config .\configs\scheduler\collect_large.yaml `
  --scheduler-wm-config .\configs\scheduler\wm_large.yaml `
  --scheduler-actor-config .\configs\scheduler\actor_large.yaml `
  --deployment-wm-config .\configs\deployment\wm_large.yaml `
  --deployment-actor-config .\configs\deployment\actor_large.yaml `
  --joint-stage7-config .\configs\joint\stage7_large.yaml `
  --deployment-data-episodes 1200 `
  --eval-episodes 64 `
  --gpu-id 2 --device auto --clean
```
