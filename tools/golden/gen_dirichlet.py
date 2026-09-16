import json
import os

from scipy.stats import dirichlet
from scipy.special import psi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "dirichlet.json")

cases = []

pdf_cases = [
    ([1.0, 1.0, 1.0], [1 / 3, 1 / 3, 1 / 3]),
    ([2.0, 3.0, 4.0], [0.2, 0.3, 0.5]),
    ([10.0, 10.0, 10.0], [1 / 3, 1 / 3, 1 / 3]),
    ([1000.0, 1.0, 1.0], [0.998, 0.001, 0.001]),
    ([0.5, 0.5], [0.999999, 0.000001]),
    ([2.0, 2.0], [0.5, 0.5]),
    ([0.1, 0.1, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1]),
]

for alpha, x in pdf_cases:
    cases.append(
        {
            "fn": "logPdf",
            "x": x,
            "params": {"alpha": alpha},
            "expected": float(dirichlet.logpdf(x, alpha)),
        }
    )
    cases.append(
        {
            "fn": "pdf",
            "x": x,
            "params": {"alpha": alpha},
            "expected": float(dirichlet.pdf(x, alpha)),
        }
    )

mean_alphas = [
    [1.0, 1.0, 1.0],
    [2.0, 3.0, 5.0],
    [0.001, 0.001, 0.001],
    [1e6, 1.0, 1.0],
    [10.0, 20.0],
]

for alpha in mean_alphas:
    m = dirichlet.mean(alpha)
    cases.append(
        {"fn": "mean", "params": {"alpha": alpha}, "expected": [float(v) for v in m]}
    )

expected_log_alphas = [
    [1.0, 1.0, 1.0],
    [2.0, 3.0, 5.0],
    [0.001, 1.0, 1000.0],
    [0.5, 0.5, 0.5],
    [1e6, 1e6],
]

for alpha in expected_log_alphas:
    total = sum(alpha)
    values = [float(psi(a) - psi(total)) for a in alpha]
    cases.append({"fn": "expectedLog", "params": {"alpha": alpha}, "expected": values})

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"wrote {len(cases)} cases to {OUT}")
