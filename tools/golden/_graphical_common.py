"""Shared graph literals and a from-scratch joint-enumeration oracle for the factor-graph
golden scripts (gen_factorGraph.py, gen_sumProduct.py, gen_maxSum.py). Not a gen_*.py
script itself, so tools/golden/generate.mjs's discovery never invokes it directly.

Every graph is a bipartite variable/factor tree, stored as plain dicts so json.dumps
serialises it unchanged into the fixture (the TypeScript side reads the identical shape
back). Factor tables are flattened row-major over the factor's own `scope`, last scope
variable fastest, matching numpy's default C order and the stride convention the
TypeScript `FactorGraph` uses.
"""

import itertools
import math

import numpy as np


def strides(sizes):
    s = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        s[i] = s[i + 1] * sizes[i + 1]
    return s


def var_sizes(graph):
    return {v["id"]: v["states"] for v in graph["variables"]}


def factor_value(graph, factor, assignment):
    sizes = var_sizes(graph)
    scope_sizes = [sizes[v] for v in factor["scope"]]
    st = strides(scope_sizes)
    idx = sum(assignment[v] * s for v, s in zip(factor["scope"], st))
    return factor["table"][idx]


def joint_value(graph, assignment):
    value = 1.0
    for factor in graph["factors"]:
        value *= factor_value(graph, factor, assignment)
    return value


def _enumerate_assignments(graph):
    sizes = var_sizes(graph)
    var_ids = [v["id"] for v in graph["variables"]]
    ranges = [range(sizes[v]) for v in var_ids]
    for combo in itertools.product(*ranges):
        yield dict(zip(var_ids, combo))


def brute_force_marginals(graph):
    sizes = var_sizes(graph)
    totals = {v: np.zeros(sizes[v]) for v in sizes}
    z = 0.0
    for assignment in _enumerate_assignments(graph):
        jv = joint_value(graph, assignment)
        z += jv
        for v, val in assignment.items():
            totals[v][val] += jv
    return {v: (totals[v] / z).tolist() for v in sizes}, z


def brute_force_map(graph):
    best_assignment = None
    best_value = -math.inf
    second_best = -math.inf
    for assignment in _enumerate_assignments(graph):
        jv = joint_value(graph, assignment)
        if jv > best_value:
            second_best = best_value
            best_value = jv
            best_assignment = dict(assignment)
        elif jv > second_best:
            second_best = jv
    margin = best_value - second_best
    assert margin > 1e-6, f"MAP configuration is not clearly unique (margin {margin})"
    return best_assignment, best_value


# Three graphs of increasing structural interest:
#   chain4:    x1-fa-x2-fb-x3-fc-x4, a plain chain (PRML 8.4.1).
#   tree1:     x3 is a hub joined to three separate binary factors (PRML 8.4.2's
#              generalisation from chain to tree).
#   hyperTree: g has three variables in its scope at once, so sending a message out of
#              g requires summing/maxing over two other variables jointly, not one.

CHAIN4 = {
    "variables": [
        {"id": "x1", "states": 2},
        {"id": "x2", "states": 3},
        {"id": "x3", "states": 2},
        {"id": "x4", "states": 2},
    ],
    "factors": [
        {"id": "fa", "scope": ["x1", "x2"], "table": [1.0, 2.0, 0.5, 1.5, 3.0, 0.2]},
        {"id": "fb", "scope": ["x2", "x3"], "table": [0.7, 1.3, 2.0, 0.4, 1.1, 0.9]},
        {"id": "fc", "scope": ["x3", "x4"], "table": [1.0, 0.6, 0.3, 2.2]},
    ],
}

TREE1 = {
    "variables": [
        {"id": "x1", "states": 2},
        {"id": "x2", "states": 3},
        {"id": "x3", "states": 2},
        {"id": "x4", "states": 2},
    ],
    "factors": [
        {"id": "fa", "scope": ["x1", "x3"], "table": [1.0, 0.4, 0.9, 1.7]},
        {"id": "fb", "scope": ["x2", "x3"], "table": [0.5, 1.2, 1.0, 0.3, 2.1, 0.8]},
        {"id": "fc", "scope": ["x3", "x4"], "table": [1.4, 0.6, 0.2, 1.1]},
    ],
}

HYPER_TREE = {
    "variables": [
        {"id": "y1", "states": 2},
        {"id": "y2", "states": 2},
        {"id": "y3", "states": 2},
        {"id": "y4", "states": 2},
    ],
    "factors": [
        # scope [y1, y2, y3], flattened row-major with y3 fastest.
        {
            "id": "g",
            "scope": ["y1", "y2", "y3"],
            "table": [1.0, 1.8, 0.6, 0.3, 0.9, 1.4, 2.0, 0.5],
        },
        {"id": "h", "scope": ["y3", "y4"], "table": [1.1, 0.4, 0.7, 1.6]},
    ],
}

GRAPHS = {"chain4": CHAIN4, "tree1": TREE1, "hyperTree": HYPER_TREE}
