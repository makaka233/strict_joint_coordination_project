from __future__ import annotations
import numpy as np
import torch
from src.models.mlp import MLP
from src.env.core import stage_count, repair_deployment

class DeploymentPolicy:
    def __init__(self, obs_dim: int, num_outputs: int, hidden: int = 128, device: str | torch.device = 'cpu'):
        self.model = MLP(obs_dim, num_outputs, hidden=hidden)
        self.model.to(device)

    def act(self, obs: np.ndarray, scn, max_replicas: int) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = self.model(o)[0]
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        x = probs.reshape(stage_count(scn), scn.num_nodes)
        raw = (x > 0.5).astype(np.float32)
        return repair_deployment(raw, scn, max_replicas, scores=x)
