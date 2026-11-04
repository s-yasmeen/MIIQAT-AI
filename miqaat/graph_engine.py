"""Small, dependency-free graph utilities for zone-level risk propagation."""
from collections import deque
from typing import Dict, Iterable, Mapping, Sequence


def validate_graph(zone_ids: Iterable[str], edges: Mapping[str, Sequence[str]]) -> None:
    nodes = set(zone_ids)
    if set(edges) - nodes:
        raise ValueError("graph contains an unknown source node")
    for source, targets in edges.items():
        if source in targets:
            raise ValueError(f"self-loop detected at {source}")
        if set(targets) - nodes:
            raise ValueError(f"graph contains an unknown target from {source}")


def topological_order(zone_ids: Iterable[str], edges: Mapping[str, Sequence[str]]) -> list[str]:
    """Return a topological order; raise ValueError when the route graph is cyclic."""
    nodes = list(zone_ids)
    validate_graph(nodes, edges)
    indegree = {node: 0 for node in nodes}
    for source in nodes:
        for target in edges.get(source, []):
            indegree[target] += 1
    queue = deque(node for node in nodes if indegree[node] == 0)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for target in edges.get(node, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(result) != len(nodes):
        raise ValueError("route graph contains a cycle")
    return result


def normalized_adjacency(zone_ids: Sequence[str], edges: Mapping[str, Sequence[str]]) -> list[list[float]]:
    """Build row-normalized adjacency with self-loops; rows aggregate source features."""
    validate_graph(zone_ids, edges)
    index = {zone: i for i, zone in enumerate(zone_ids)}
    n = len(zone_ids)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
    for source, targets in edges.items():
        for target in targets:
            # Target receives information from source.
            matrix[index[target]][index[source]] = 1.0
    for row in matrix:
        total = sum(row)
        if total:
            for j in range(n):
                row[j] /= total
    return matrix


def propagate_max_risk(
    base_risk: Mapping[str, float],
    edges: Mapping[str, Sequence[str]],
    coupling: float = 0.10,
    tolerance: float = 1e-9,
    max_iterations: int = 500,
) -> Dict[str, float]:
    """Fixed-point downstream propagation; supports directed graphs with cycles."""
    if not 0.0 <= coupling < 1.0:
        raise ValueError("coupling must be in [0, 1)")
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")
    validate_graph(base_risk.keys(), edges)
    risks = {node: min(1.0, max(0.0, float(value))) for node, value in base_risk.items()}
    for _ in range(max_iterations):
        updated = {
            node: min(
                1.0,
                max(0.0, float(base_risk[node]))
                + coupling * max((risks.get(target, 0.0) for target in edges.get(node, [])), default=0.0),
            )
            for node in risks
        }
        delta = max((abs(updated[node] - risks[node]) for node in risks), default=0.0)
        risks = updated
        if delta <= tolerance:
            return risks
    raise RuntimeError("risk propagation did not converge within max_iterations")
