from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from src.kkt.solver import kkt_compute_latencies, kkt_bandwidth_latencies

@dataclass
class Scenario:
    num_nodes: int
    service_stages: list[int]
    stage_memory: list[float]
    stage_storage: list[float]
    node_memory: list[float]
    node_storage: list[float]
    node_compute: list[float]
    bandwidth: dict[tuple[int, int], float]
    stage_to_service: list[int]
    stage_local_idx: list[int]


def build_scenario(cfg: dict[str, Any]) -> Scenario:
    num_nodes = int(cfg['num_nodes'])
    service_stages = list(cfg['service_stages'])
    stage_memory = []
    stage_storage = []
    stage_to_service = []
    stage_local_idx = []
    rng = np.random.default_rng(int(cfg.get('seed', 0)))
    for i, js in enumerate(service_stages):
        for j in range(js):
            stage_to_service.append(i)
            stage_local_idx.append(j)
            stage_memory.append(float(cfg['stage_memory_base']) * (1.0 + 0.15 * ((i + j) % 3)))
            stage_storage.append(float(cfg['stage_storage_base']) * (1.0 + 0.2 * ((i + 2*j) % 3)))
    node_memory = [float(cfg['node_memory'])] * num_nodes
    node_storage = [float(cfg['node_storage'])] * num_nodes
    node_compute = [float(cfg['node_compute']) * (0.9 + 0.2 * rng.random()) for _ in range(num_nodes)]
    bandwidth = {}
    base_bw = float(cfg['bandwidth'])
    for a in range(num_nodes):
        for b in range(num_nodes):
            if a == b:
                continue
            bandwidth[(a, b)] = base_bw * (0.8 + 0.4 * rng.random())
    return Scenario(num_nodes, service_stages, stage_memory, stage_storage, node_memory, node_storage, node_compute, bandwidth, stage_to_service, stage_local_idx)


def stage_count(scn: Scenario) -> int:
    return len(scn.stage_memory)


def deployment_feasible(x: np.ndarray, scn: Scenario, max_replicas: int) -> bool:
    s_count, n_count = x.shape
    if n_count != scn.num_nodes or s_count != stage_count(scn):
        return False
    for s in range(s_count):
        replicas = int(x[s].sum())
        if replicas < 1 or replicas > max_replicas:
            return False
    mem = np.zeros(scn.num_nodes)
    sto = np.zeros(scn.num_nodes)
    for s in range(s_count):
        for n in range(scn.num_nodes):
            if x[s, n] > 0.5:
                mem[n] += scn.stage_memory[s]
                sto[n] += scn.stage_storage[s]
    return bool(np.all(mem <= np.array(scn.node_memory) + 1e-6) and np.all(sto <= np.array(scn.node_storage) + 1e-6))


def repair_deployment(x: np.ndarray, scn: Scenario, max_replicas: int, scores: np.ndarray | None = None) -> np.ndarray:
    x = (x > 0.5).astype(np.float32)
    s_count = stage_count(scn)
    if scores is None:
        scores = np.zeros_like(x)
    for s in range(s_count):
        if x[s].sum() < 1:
            order = np.argsort(scores[s])[::-1]
            x[s, order[0]] = 1.0
        while x[s].sum() > max_replicas:
            on = np.where(x[s] > 0.5)[0]
            off = on[np.argmin(scores[s, on])]
            x[s, off] = 0.0
    while True:
        mem = np.zeros(scn.num_nodes)
        sto = np.zeros(scn.num_nodes)
        for s in range(s_count):
            for n in range(scn.num_nodes):
                if x[s, n] > 0.5:
                    mem[n] += scn.stage_memory[s]
                    sto[n] += scn.stage_storage[s]
        bad = None
        for n in range(scn.num_nodes):
            if mem[n] > scn.node_memory[n] + 1e-6 or sto[n] > scn.node_storage[n] + 1e-6:
                bad = n
                break
        if bad is None:
            break
        cand = np.where(x[:, bad] > 0.5)[0]
        removable = [s for s in cand if x[s].sum() > 1]
        if not removable:
            break
        s = removable[int(np.argmin(scores[removable, bad]))]
        x[s, bad] = 0.0
    for s in range(s_count):
        if x[s].sum() < 1:
            ratios = [scn.stage_memory[s]/scn.node_memory[n] + scn.stage_storage[s]/scn.node_storage[n] for n in range(scn.num_nodes)]
            x[s, int(np.argmin(ratios))] = 1.0
    return x


def _cfg(cfg: dict[str, Any] | None, key: str, default: float | int):
    if cfg is None:
        return default
    return cfg.get(key, default)


def _stage_candidate_scores(s: int, macro_obs: np.ndarray, scn: Scenario, rem_mem: np.ndarray, rem_sto: np.ndarray, cfg: dict[str, Any] | None, rng: np.random.Generator | None) -> np.ndarray:
    demand = macro_obs[:-1, s].astype(np.float32)
    demand = demand / max(float(demand.max()), 1e-6)
    compute = np.asarray(scn.node_compute, dtype=np.float32)
    compute = compute / max(float(compute.max()), 1e-6)
    rem_ratio = rem_mem / np.maximum(np.asarray(scn.node_memory, dtype=np.float32), 1e-6)
    rem_ratio += rem_sto / np.maximum(np.asarray(scn.node_storage, dtype=np.float32), 1e-6)
    rem_ratio *= 0.5
    diversity = np.linspace(0.0, 1.0, scn.num_nodes, dtype=np.float32)
    noise = 0.0
    if rng is not None:
        noise = float(_cfg(cfg, 'deployment_noise', 0.05)) * rng.standard_normal(scn.num_nodes)
    return (
        float(_cfg(cfg, 'deployment_demand_weight', 1.0)) * demand
        + float(_cfg(cfg, 'deployment_compute_weight', 0.25)) * compute
        + float(_cfg(cfg, 'deployment_resource_weight', 0.35)) * rem_ratio
        + float(_cfg(cfg, 'deployment_diversity_bias', 0.10)) * diversity
        + noise
    )


def _desired_replicas(total_demand: np.ndarray, s: int, max_replicas: int, cfg: dict[str, Any] | None, rng: np.random.Generator | None) -> int:
    if max_replicas <= 1:
        return 1
    mean_d = max(float(total_demand.mean()), 1e-6)
    ratio = float(total_demand[s] / mean_d)
    target = int(_cfg(cfg, 'deployment_replica_min', 1))
    if ratio >= float(_cfg(cfg, 'deployment_replica_ratio_2', 1.05)):
        target = max(target, 2)
    if max_replicas >= 3 and ratio >= float(_cfg(cfg, 'deployment_replica_ratio_3', 1.40)):
        target = max(target, 3)
    if max_replicas >= 4 and ratio >= float(_cfg(cfg, 'deployment_replica_ratio_4', 1.80)):
        target = max(target, 4)
    extra_p = float(_cfg(cfg, 'deployment_extra_replica_prob', 0.12))
    if rng is not None and rng.random() < extra_p:
        target += 1
    return int(np.clip(target, 1, max_replicas))


def greedy_direct_deployment(macro_obs: np.ndarray, scn: Scenario, max_replicas: int, cfg: dict[str, Any] | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    total_demand = macro_obs[-1]
    x = np.zeros((stage_count(scn), scn.num_nodes), dtype=np.float32)
    node_mem = np.array(scn.node_memory, dtype=np.float32)
    node_sto = np.array(scn.node_storage, dtype=np.float32)
    scores = np.zeros_like(x)
    topk = int(_cfg(cfg, 'deployment_candidate_topk', max(3, max_replicas + 1)))
    topk = min(topk, scn.num_nodes)
    for s in np.argsort(total_demand)[::-1]:
        node_scores = _stage_candidate_scores(s, macro_obs, scn, node_mem, node_sto, cfg, rng)
        scores[s] = node_scores
        order = np.argsort(node_scores)[::-1]
        order = order[:topk]
        replicas = _desired_replicas(total_demand, s, max_replicas, cfg, rng)
        placed = 0
        for n in order:
            if node_mem[n] >= scn.stage_memory[s] and node_sto[n] >= scn.stage_storage[s]:
                x[s, n] = 1.0
                node_mem[n] -= scn.stage_memory[s]
                node_sto[n] -= scn.stage_storage[s]
                placed += 1
                if placed >= replicas:
                    break
        if placed == 0:
            fallback = int(np.argmax(node_mem + node_sto))
            x[s, fallback] = 1.0
            node_mem[fallback] = max(0.0, node_mem[fallback] - scn.stage_memory[s])
            node_sto[fallback] = max(0.0, node_sto[fallback] - scn.stage_storage[s])
    return repair_deployment(x, scn, max_replicas, scores=scores)


def random_feasible_deployment(macro_obs: np.ndarray, scn: Scenario, max_replicas: int, rng: np.random.Generator, cfg: dict[str, Any] | None = None) -> np.ndarray:
    s_count = stage_count(scn)
    x = np.zeros((s_count, scn.num_nodes), dtype=np.float32)
    rem_mem = np.array(scn.node_memory, dtype=np.float32)
    rem_sto = np.array(scn.node_storage, dtype=np.float32)
    total = macro_obs[-1]
    stage_order = np.argsort(total + 0.10 * rng.standard_normal(s_count))[::-1]
    scores = np.zeros_like(x)
    topk = min(scn.num_nodes, int(_cfg(cfg, 'deployment_candidate_topk', max(3, max_replicas + 1))))
    for s in stage_order:
        node_pref = _stage_candidate_scores(s, macro_obs, scn, rem_mem, rem_sto, cfg, rng)
        order = np.argsort(node_pref)[::-1][:topk]
        scores[s] = node_pref
        replicas = _desired_replicas(total, s, max_replicas, cfg, rng)
        replicas = int(np.clip(replicas + int(rng.random() < 0.35), 1, max_replicas))
        placed = 0
        for n in order:
            if rem_mem[n] >= scn.stage_memory[s] and rem_sto[n] >= scn.stage_storage[s]:
                x[s, n] = 1.0
                rem_mem[n] -= scn.stage_memory[s]
                rem_sto[n] -= scn.stage_storage[s]
                placed += 1
                if placed >= replicas:
                    break
        if placed == 0:
            fallback = int(np.argmax(rem_mem + rem_sto))
            x[s, fallback] = 1.0
    return repair_deployment(x, scn, max_replicas, scores=scores)


def mutate_deployment(base: np.ndarray, macro_obs: np.ndarray, scn: Scenario, max_replicas: int, rng: np.random.Generator, mutation_prob: float = 0.18, cfg: dict[str, Any] | None = None) -> np.ndarray:
    x = np.array(base, dtype=np.float32, copy=True)
    scores = macro_obs[:-1].T.astype(np.float32).copy()
    s_count = stage_count(scn)
    topk = min(scn.num_nodes, int(_cfg(cfg, 'deployment_candidate_topk', max(3, max_replicas + 1))))
    for s in range(s_count):
        if rng.random() > mutation_prob:
            continue
        demand_order = np.argsort(scores[s])[::-1][:topk]
        if rng.random() < 0.5:
            active = np.where(x[s] > 0.5)[0]
            inactive = [n for n in demand_order if x[s, n] < 0.5]
            if len(active) > 0 and len(inactive) > 0:
                off = int(active[rng.integers(0, len(active))])
                on = int(inactive[rng.integers(0, len(inactive))])
                x[s, off] = 0.0
                x[s, on] = 1.0
        else:
            cand = int(demand_order[rng.integers(0, len(demand_order))])
            x[s, cand] = 1.0
            if rng.random() < 0.35:
                active = np.where(x[s] > 0.5)[0]
                if len(active) > 1:
                    off = int(active[rng.integers(0, len(active))])
                    x[s, off] = 0.0
    return repair_deployment(x, scn, max_replicas, scores=scores)


def generate_macro_obs(scn: Scenario, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    s_count = stage_count(scn)
    obs = np.zeros((scn.num_nodes + 1, s_count), dtype=np.float32)
    base = float(cfg.get('macro_demand_base', 16.0))
    volatility = float(cfg.get('macro_demand_volatility', 8.0))
    node_hot = rng.integers(0, scn.num_nodes)
    for s in range(s_count):
        service = scn.stage_to_service[s]
        for n in range(scn.num_nodes):
            demand = max(0.0, base + rng.normal(0, volatility) + (8.0 if n == node_hot and service % 2 == 0 else 0.0))
            obs[n, s] = demand
    obs[-1] = obs[:-1].sum(axis=0)
    return obs


def flatten_macro_obs(obs: np.ndarray) -> np.ndarray:
    return obs.reshape(-1).astype(np.float32)


def generate_scheduler_tasks(macro_obs: np.ndarray, scn: Scenario, cfg: dict[str, Any], rng: np.random.Generator):
    num_tasks = int(cfg.get('scheduler_num_tasks', 32))
    s_count = stage_count(scn)
    weights = macro_obs[:-1].sum(axis=0)
    weights = np.maximum(weights, 1e-6)
    weights = weights / weights.sum()
    tasks = []
    for _ in range(num_tasks):
        s_global = int(rng.choice(np.arange(s_count), p=weights))
        service = scn.stage_to_service[s_global]
        origin_probs = macro_obs[:-1, s_global]
        if origin_probs.sum() <= 0:
            origin_probs = np.ones(scn.num_nodes)
        origin_probs = origin_probs / origin_probs.sum()
        origin = int(rng.choice(np.arange(scn.num_nodes), p=origin_probs))
        compute_base = float(cfg.get('compute_base', 18.0))
        data_base = float(cfg.get('data_base', 6.0))
        stage_count_local = scn.service_stages[service]
        task = {
            'origin': origin,
            'service': service,
            'stage_compute': [max(1.0, compute_base * (1.0 + 0.2*j + 0.1*service) * (0.8 + 0.4*rng.random())) for j in range(stage_count_local)],
            'stage_data': [max(0.5, data_base * (1.0 + 0.1*j + 0.1*service) * (0.8 + 0.4*rng.random())) for j in range(stage_count_local)],
        }
        tasks.append(task)
    return tasks


def make_scheduler_obs(task: dict, local_stage: int, prev_node: int, deployment: np.ndarray, macro_obs: np.ndarray, scn: Scenario, est_node_loads: np.ndarray) -> np.ndarray:
    s_offset = sum(scn.service_stages[:task['service']])
    s_global = s_offset + local_stage
    deployed = deployment[s_global].astype(np.float32)
    feat = []
    feat += [task['origin'], task['service'], local_stage, prev_node]
    feat += [task['stage_compute'][local_stage], task['stage_data'][local_stage]]
    feat += est_node_loads.tolist()
    feat += deployed.tolist()
    feat += macro_obs[:-1, s_global].tolist()
    return np.array(feat, dtype=np.float32)


def scheduler_action_mask(task: dict, local_stage: int, deployment: np.ndarray, scn: Scenario) -> np.ndarray:
    s_offset = sum(scn.service_stages[:task['service']])
    s_global = s_offset + local_stage
    return deployment[s_global].astype(np.float32)


def scheduler_action_costs(mask: np.ndarray, task: dict, local_stage: int, prev_node: int, scn: Scenario, est_node_loads: np.ndarray) -> np.ndarray:
    c = float(task['stage_compute'][local_stage])
    d = float(task['stage_data'][local_stage])
    costs = np.full(scn.num_nodes, np.inf, dtype=np.float32)
    for n in range(scn.num_nodes):
        if mask[n] < 0.5:
            continue
        trans = 0.0 if prev_node == n else d / scn.bandwidth[(prev_node, n)]
        eff_compute = max(scn.node_compute[n] / (1.0 + est_node_loads[n]), 1e-6)
        comp = c / eff_compute
        costs[n] = trans + comp
    return costs


def scheduler_target_probs(costs: np.ndarray, mask: np.ndarray, temperature: float = 0.35) -> np.ndarray:
    valid = np.isfinite(costs) & (mask > 0.5)
    out = np.zeros_like(costs, dtype=np.float32)
    if not np.any(valid):
        return out
    vals = costs[valid]
    vals = vals - vals.min()
    logits = -vals / max(temperature, 1e-6)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / max(probs.sum(), 1e-8)
    out[np.where(valid)[0]] = probs.astype(np.float32)
    return out


def greedy_scheduler_action(obs: np.ndarray, mask: np.ndarray, task: dict, local_stage: int, prev_node: int, scn: Scenario, est_node_loads: np.ndarray) -> int:
    costs = scheduler_action_costs(mask, task, local_stage, prev_node, scn, est_node_loads)
    valid = np.where(np.isfinite(costs))[0]
    if len(valid) == 0:
        return int(np.argmax(mask))
    return int(valid[np.argmin(costs[valid])])


def evaluate_deployment_with_scheduler(macro_obs: np.ndarray, deployment: np.ndarray, scn: Scenario, cfg: dict[str, Any], scheduler_policy) -> dict:
    rng = np.random.default_rng(int(cfg.get('seed', 0)) + int(macro_obs.sum() * 10) % 100000)
    tasks = generate_scheduler_tasks(macro_obs, scn, cfg, rng)
    comp_assignments = []
    flows = []
    est_node_loads = np.zeros(scn.num_nodes, dtype=np.float32)
    per_task_latency = []
    for task in tasks:
        prev = task['origin']
        task_stage_meta = []
        for local_stage in range(scn.service_stages[task['service']]):
            obs = make_scheduler_obs(task, local_stage, prev, deployment, macro_obs, scn, est_node_loads)
            mask = scheduler_action_mask(task, local_stage, deployment, scn)
            action = int(scheduler_policy(obs, mask, task, local_stage, prev))
            if mask[action] < 0.5:
                action = int(np.argmax(mask))
            c = float(task['stage_compute'][local_stage])
            d = float(task['stage_data'][local_stage])
            comp_assignments.append((action, c))
            flow_idx = None
            if prev != action:
                flow_idx = len(flows)
                flows.append(((prev, action), d))
            task_stage_meta.append((len(comp_assignments) - 1, flow_idx))
            est_node_loads[action] += 1.0
            prev = action
        per_task_latency.append(task_stage_meta)
    comp_lats = kkt_compute_latencies(scn.node_compute, comp_assignments)
    flow_lats = kkt_bandwidth_latencies(scn.bandwidth, flows)
    latencies = []
    for task_meta in per_task_latency:
        total = 0.0
        for comp_idx, flow_idx in task_meta:
            total += comp_lats[comp_idx]
            if flow_idx is not None:
                total += flow_lats[flow_idx]
        latencies.append(total)
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    total_reward = -float(sum(latencies))
    return {
        'total_reward': total_reward,
        'mean_window_latency': mean_lat * len(latencies),
        'mean_task_latency': mean_lat,
        'num_tasks': len(latencies),
    }
