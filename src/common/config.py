from __future__ import annotations
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
