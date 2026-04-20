from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path
import torch
from src.common.config import load_yaml, PROJECT_ROOT
from src.common.seed import set_seed
from src.env.core import build_scenario, stage_count
from src.agents.scheduler.policy import SchedulerPolicy
from src.agents.deployment.policy import DeploymentPolicy


def project_path(rel: str) -> Path:
    return PROJECT_ROOT / rel


def save_checkpoint(model: torch.nn.Module, path: str | Path, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = meta or {}
    first_linear = None
    for k, v in model.state_dict().items():
        if k.endswith('weight') and v.ndim == 2:
            first_linear = int(v.shape[0])
            break
    if first_linear is not None:
        meta.setdefault('hidden', first_linear)
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, path)


def _infer_hidden(ckpt: dict, default: int = 128) -> int:
    meta = ckpt.get('meta', {}) if isinstance(ckpt, dict) else {}
    if 'hidden' in meta:
        return int(meta['hidden'])
    sd = ckpt['state_dict']
    for k, v in sd.items():
        if k.endswith('weight') and v.ndim == 2:
            return int(v.shape[0])
    return default


def load_scheduler_policy(ckpt_path: str | Path, obs_dim: int, num_nodes: int, hidden: int | None = None, device: str | torch.device = 'cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden = _infer_hidden(ckpt, hidden or 128)
    pol = SchedulerPolicy(obs_dim, num_nodes, hidden=hidden, device=device)
    pol.model.load_state_dict(ckpt['state_dict'])
    pol.model.to(device)
    return pol


def load_deployment_policy(ckpt_path: str | Path, obs_dim: int, num_outputs: int, hidden: int | None = None, device: str | torch.device = 'cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden = _infer_hidden(ckpt, hidden or 128)
    pol = DeploymentPolicy(obs_dim, num_outputs, hidden=hidden, device=device)
    pol.model.load_state_dict(ckpt['state_dict'])
    pol.model.to(device)
    return pol
