"""Golden fixtures for packages/math/src/graphical/dSeparation.ts.

Two independent oracles, neither of which is the TypeScript algorithm:

- `dSeparated` cases come straight from `networkx.is_d_separator`, exhaustively over
  every ordered pair of nodes and every subset of the remaining nodes as the observed
  set, for four small DAGs (PRML 8.2.1's three canonical graphs plus a five-node graph
  combining a collider, a descendant of a collider, and a second collider on an
  alternate path).
- `classifyPath` cases come from an independent Python re-implementation of the
  per-node role rule (8.2, page 373-374: tail-to-tail or head-to-tail blocks iff
  observed; head-to-head blocks unless it or a descendant is observed), applied to
  every simple undirected path `networkx` enumerates between the same pairs. A
  consistency check below (not part of the emitted fixture) confirms that "every path
  blocked" agrees with `is_d_separator` on all four graphs before anything is written,
  so a bug in the independent classifier would fail this script, not just the eventual
  TypeScript test.
"""

import itertools
import json
from pathlib import Path

import networkx as nx

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

GRAPHS = {
    "tailToTail": {"nodes": ["A", "B", "C"], "edges": [["C", "A"], ["C", "B"]]},
    "headToTail": {"nodes": ["A", "B", "C"], "edges": [["A", "C"], ["C", "B"]]},
    "headToHead": {"nodes": ["A", "B", "C"], "edges": [["A", "C"], ["B", "C"]]},
    "explainingAway": {
        "nodes": ["A", "B", "C", "D", "E"],
        "edges": [["A", "C"], ["B", "C"], ["C", "D"], ["D", "E"], ["A", "E"]],
    },
}


def build_digraph(spec):
    g = nx.DiGraph()
    g.add_nodes_from(spec["nodes"])
    g.add_edges_from(spec["edges"])
    return g


def subsets(items):
    for r in range(len(items) + 1):
        yield from itertools.combinations(items, r)


def role_and_block(g, prev, node, nxt, ancestors_of_z, observed):
    prev_is_parent = g.has_edge(prev, node)
    next_is_parent = g.has_edge(nxt, node)
    node_to_prev = g.has_edge(node, prev)
    node_to_next = g.has_edge(node, nxt)

    if prev_is_parent and next_is_parent:
        role = "head-to-head"
        blocking = node not in observed and node not in ancestors_of_z
    elif node_to_prev and node_to_next:
        role = "tail-to-tail"
        blocking = node in observed
    else:
        role = "head-to-tail"
        blocking = node in observed
    return role, blocking


def classify_path(g, path, observed):
    z = set(observed)
    ancestors_of_z = set()
    for node in z:
        ancestors_of_z |= nx.ancestors(g, node)
    ancestors_of_z |= z

    steps = []
    blocked = False
    for i in range(1, len(path) - 1):
        role, blocking = role_and_block(
            g, path[i - 1], path[i], path[i + 1], ancestors_of_z, z
        )
        steps.append({"node": path[i], "role": role, "blocking": blocking})
        if blocking:
            blocked = True
    return {"nodes": list(path), "blocked": blocked, "steps": steps}


def all_simple_undirected_paths(g, x, y):
    return list(nx.all_simple_paths(g.to_undirected(), x, y))


cases_d_separated = []
cases_classify_path = []

for name, spec in GRAPHS.items():
    g = build_digraph(spec)
    nodes = spec["nodes"]
    for x, y in itertools.permutations(nodes, 2):
        others = [n for n in nodes if n not in (x, y)]
        for z in subsets(others):
            expected = bool(nx.is_d_separator(g, {x}, {y}, set(z)))
            cases_d_separated.append(
                {
                    "graph": name,
                    "nodes": nodes,
                    "edges": spec["edges"],
                    "x": x,
                    "y": y,
                    "observed": list(z),
                    "expected": expected,
                }
            )

            paths = all_simple_undirected_paths(g, x, y)
            classified = [classify_path(g, p, z) for p in paths]
            all_blocked = all(c["blocked"] for c in classified) if classified else True
            assert all_blocked == expected, (
                f"{name}: path-blocking aggregate disagrees with is_d_separator for "
                f"x={x} y={y} z={z}: paths={classified} expected={expected}"
            )
            for c in classified:
                cases_classify_path.append(
                    {
                        "graph": name,
                        "nodes": nodes,
                        "edges": spec["edges"],
                        "observed": list(z),
                        "path": c["nodes"],
                        "expectedBlocked": c["blocked"],
                        "expectedSteps": c["steps"],
                    }
                )

(FIXTURES / "dSeparation.json").write_text(
    json.dumps(
        {"dSeparated": cases_d_separated, "classifyPath": cases_classify_path}, indent=2
    )
)
print(
    f"wrote {len(cases_d_separated)} dSeparated cases and {len(cases_classify_path)} "
    f"classifyPath cases to {FIXTURES / 'dSeparation.json'}"
)
