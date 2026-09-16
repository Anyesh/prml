"""Golden fixtures for packages/math/src/graphical/maxSum.ts.

Expected MAP configurations come from `brute_force_map` in _graphical_common.py, an
exhaustive argmax over the full joint, independent of the message-passing and
backtracking the TypeScript `maxSum` performs. Each graph's tables were chosen so the
best and second-best joint value differ by more than 1e-6 (asserted in
_graphical_common.py), so the configuration compared against is unambiguous.
"""

import json
import math
from pathlib import Path

from _graphical_common import GRAPHS, brute_force_map

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

roots_by_graph = {"chain4": "x1", "tree1": "x3", "hyperTree": "y3"}

for graph_name, root in roots_by_graph.items():
    graph = GRAPHS[graph_name]
    assignment, value = brute_force_map(graph)
    cases.append(
        {
            "fn": "maxSumMap",
            "graph": graph,
            "root": root,
            "expectedAssignment": assignment,
            "expectedMaxLogValue": math.log(value),
        }
    )

(FIXTURES / "maxSum.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'maxSum.json'}")
