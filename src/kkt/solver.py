from __future__ import annotations
from collections import defaultdict
import math


def kkt_compute_latencies(node_caps: list[float], assignments: list[tuple[int, float]]):
    by_node = defaultdict(list)
    for idx, (node, c) in enumerate(assignments):
        by_node[node].append((idx, float(c)))
    lat = [0.0] * len(assignments)
    for node, items in by_node.items():
        denom = sum(math.sqrt(max(c, 1e-8)) for _, c in items)
        if denom <= 0:
            continue
        cap = float(node_caps[node])
        for idx, c in items:
            f = cap * math.sqrt(max(c, 1e-8)) / denom
            lat[idx] = c / max(f, 1e-8)
    return lat


def kkt_bandwidth_latencies(link_caps: dict[tuple[int, int], float], flows: list[tuple[tuple[int, int], float]]):
    by_link = defaultdict(list)
    for idx, (lk, d) in enumerate(flows):
        by_link[lk].append((idx, float(d)))
    lat = [0.0] * len(flows)
    for lk, items in by_link.items():
        denom = sum(math.sqrt(max(d, 1e-8)) for _, d in items)
        if denom <= 0:
            continue
        cap = float(link_caps[lk])
        for idx, d in items:
            r = cap * math.sqrt(max(d, 1e-8)) / denom
            lat[idx] = d / max(r, 1e-8)
    return lat
