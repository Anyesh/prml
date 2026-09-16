import json
from pathlib import Path

from _graphical_common import GRAPHS, factor_value

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

probes = [
    ("chain4", "fa", {"x1": 0, "x2": 2}),
    ("chain4", "fb", {"x2": 1, "x3": 0}),
    ("chain4", "fc", {"x3": 1, "x4": 1}),
    ("tree1", "fa", {"x1": 1, "x3": 0}),
    ("hyperTree", "g", {"y1": 1, "y2": 0, "y3": 1}),
    ("hyperTree", "h", {"y3": 0, "y4": 1}),
]

for graph_name, factor_id, assignment in probes:
    graph = GRAPHS[graph_name]
    factor = next(f for f in graph["factors"] if f["id"] == factor_id)
    cases.append(
        {
            "fn": "evaluateFactor",
            "graph": graph,
            "factorId": factor_id,
            "assignment": assignment,
            "expected": factor_value(graph, factor, assignment),
        }
    )

(FIXTURES / "factorGraph.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'factorGraph.json'}")
