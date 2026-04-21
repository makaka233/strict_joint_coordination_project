from __future__ import annotations
import copy
from collections import deque
from typing import Any

import numpy as np


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.quantile(arr, q))


def compute_joint_score(ev: dict[str, Any], cfg: dict[str, Any]) -> float:
    score = (
        float(cfg.get('accept_mean_weight', 1.0)) * float(ev['mean_latency'])
        + float(cfg.get('accept_p90_weight', 0.25)) * float(ev['p90_latency'])
        + float(cfg.get('accept_worst_weight', 0.05)) * float(ev['worst_latency'])
        - float(cfg.get('accept_reward_weight', 0.0)) * float(ev.get('mean_reward', 0.0))
    )
    return float(score)


def resolve_cycle_value(cfg: dict[str, Any], key: str, progress: float, default: float | int, as_int: bool = False):
    start_key = f'{key}_start'
    end_key = f'{key}_end'
    if start_key in cfg or end_key in cfg:
        start_val = float(cfg.get(start_key, cfg.get(key, default)))
        end_val = float(cfg.get(end_key, cfg.get(key, start_val)))
        value = start_val + (end_val - start_val) * min(max(progress, 0.0), 1.0)
    else:
        value = float(cfg.get(key, default))
    if as_int:
        return max(1, int(round(value)))
    return float(value)


def average_scalar_dicts(dicts: list[dict[str, Any]], prefix: str = '') -> dict[str, float]:
    if not dicts:
        return {}
    keys = sorted({k for d in dicts for k in d.keys()})
    out = {}
    for key in keys:
        vals = []
        for d in dicts:
            v = d.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out[f'{prefix}{key}'] = float(np.mean(vals))
    return out


class SampleBuffer:
    def __init__(self, max_size: int):
        self.max_size = max(1, int(max_size))
        self._rows: deque[dict[str, Any]] = deque(maxlen=self.max_size)

    def extend(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._rows.append(copy.deepcopy(row))

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

