from __future__ import annotations
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt


def read_csv(path: str | Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _to_float(x):
    try:
        if x is None or x == '':
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _moving_average(vals, win: int = 25):
    out = []
    acc = 0.0
    buf = []
    for v in vals:
        buf.append(v)
        acc += v
        if len(buf) > win:
            acc -= buf.pop(0)
        out.append(acc / len(buf))
    return out


def plot_lines(csv_path: str | Path, x_key: str, y_keys: list[str], out_path: str | Path, title: str, ma_window: int = 0):
    rows = read_csv(csv_path)
    plt.figure(figsize=(8, 4.5))
    any_plotted = False
    for yk in y_keys:
        pts = []
        for r in rows:
            xv = _to_float(r.get(x_key))
            yv = _to_float(r.get(yk))
            if xv is None or yv is None:
                continue
            pts.append((xv, yv))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if len(xs) <= 128:
            plt.plot(xs, ys, marker='o', markersize=3, linewidth=1.2, label=yk)
        else:
            plt.plot(xs, ys, linewidth=1.0, alpha=0.55, label=yk)
        if ma_window and len(xs) >= max(5, ma_window):
            plt.plot(xs, _moving_average(ys, ma_window), linewidth=2.0, linestyle='--', label=f'{yk}_ma')
        any_plotted = True
    plt.xlabel(x_key)
    plt.title(title)
    if any_plotted and len(plt.gca().lines) > 1:
        plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
