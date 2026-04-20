from __future__ import annotations
import numpy as np
import torch
from src.models.mlp import MLP

class SchedulerPolicy:
    def __init__(self, obs_dim: int, num_nodes: int, hidden: int = 128, device: str | torch.device = 'cpu'):
        self.model = MLP(obs_dim, num_nodes, hidden=hidden)
        self.model.to(device)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = self.model(o)[0]
            m = torch.tensor(mask, dtype=torch.float32, device=device)
            logits = logits + (m - 1.0) * 1e9
            return int(torch.argmax(logits).item())
