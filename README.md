# WM-driven Scheduler + Richer Deployment Generation

This variant makes the scheduler use the trained scheduler WM for online action scoring and planner distillation, instead of directly imitating replay baseline labels. It also broadens deployment generation so scheduler states more often expose 2-4 legal candidate nodes.

# Improved hierarchical deployment+scheduler project

This package contains the improved codebase only. It does **not** include external validation artifacts.

## What changed in this scheduler-focused revision
- scheduler data collection is no longer tied to a single deployment generator
  - heuristic deployment
  - random feasible deployment
  - mutated deployment near the heuristic solution
- scheduler replay now stores latency-aligned supervision:
  - per-action immediate cost vector
  - normalized per-action cost vector
  - soft target probabilities derived from action costs
  - best-vs-second-best action gap
- scheduler world model now predicts normalized chosen-action cost and gap instead of a dummy zero target
- scheduler actor now optimizes a latency-aligned objective instead of pure behavior cloning
  - decaying BC weight
  - soft imitation from cost-derived target probabilities
  - expected normalized action-cost minimization
  - entropy floor penalty to avoid early collapse
  - mixed-deployment short evaluation for better generalization tracking

## Existing project features
- GPU device selection (`--device`, `--gpu-id`)
- step-level logging to `outputs/logs/*.jsonl` and `outputs/metrics/*.csv`
- automatic figure generation to `outputs/figures/*.png`
- improved sparse-eval plotting with markers and moving averages
- deployment replay stores normalized latency targets and relative gains
- deployment world model trains on normalized `log1p(latency)` targets
- deployment actor uses a more conservative objective

## Recommended large-scale run on GPU 2

### Windows PowerShell
```powershell
python .\scripts\train_all_pipeline.py `
  --train-env-config .\configs\env\large_train.yaml `
  --eval-env-config .\configs\env\large_train.yaml `
  --scheduler-collect-config .\configs\scheduler\collect_large.yaml `
  --scheduler-wm-config .\configs\scheduler\wm_large.yaml `
  --scheduler-actor-config .\configs\scheduler\actor_large.yaml `
  --deployment-wm-config .\configs\deployment\wm_large.yaml `
  --deployment-actor-config .\configs\deployment\actor_large.yaml `
  --deployment-data-episodes 600 `
  --eval-episodes 16 `
  --gpu-id 2 `
  --device auto `
  --clean
```

## Practical notes
- This revision changes **scheduler-related parts only**. Deployment code is left structurally unchanged.
- The scheduler actor still saves the best checkpoint according to `eval_latency_short`.
- Start with `deployment-data-episodes 600` before moving to `1200`.
