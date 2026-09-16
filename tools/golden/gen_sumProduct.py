"""Golden fixtures for packages/math/src/graphical/sumProduct.ts.

Expected marginals come from `brute_force_marginals` in _graphical_common.py, which
enumerates the full joint over every variable and sums it out directly; that is a
completely different computation path from the tree message-passing the TypeScript
`sumProduct` performs, so agreement between them is a real check of the algorithm, not
of the arithmetic library both happen to share.
"""

import json
from pathlib import Path

from _graphical_common import GRAPHS, brute_force_marginals

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

# The chain is run from two different roots to also pin down the property that a tree's
# exact marginals do not depend on which node the algorithm treats as the root.
roots_by_graph = {"chain4": ["x1", "x3"], "tree1": ["x3"], "hyperTree": ["y3"]}

for graph_name, roots in roots_by_graph.items():
    graph = GRAPHS[graph_name]
    marginals, z = brute_force_marginals(graph)
    for root in roots:
        cases.append(
            {
                "fn": "sumProductMarginals",
                "graph": graph,
                "root": root,
                "expectedMarginals": marginals,
                "expectedZ": z,
            }
        )

(FIXTURES / "sumProduct.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'sumProduct.json'}")
