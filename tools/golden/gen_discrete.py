import json
import os

from scipy.stats import bernoulli, binom, multinomial
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "discrete.json")

cases = []

for x, mu in [(1, 0.3), (0, 0.3), (1, 0.999999), (0, 1e-6), (1, 0.5), (0, 0.5)]:
    cases.append(
        {
            "fn": "bernoulliLogPmf",
            "x": x,
            "params": {"mu": mu},
            "expected": float(bernoulli.logpmf(x, mu)),
        }
    )

for m, n, mu in [
    (5, 10, 0.5),
    (0, 10, 0.3),
    (10, 10, 0.3),
    (500, 1000, 0.5),
    (0, 0, 0.5),
    (3, 20, 0.001),
    (997, 1000, 0.999),
]:
    cases.append(
        {
            "fn": "binomialLogPmf",
            "m": m,
            "params": {"n": n, "mu": mu},
            "expected": float(binom.logpmf(m, n, mu)),
        }
    )
    cases.append(
        {
            "fn": "binomialPmf",
            "m": m,
            "params": {"n": n, "mu": mu},
            "expected": float(binom.pmf(m, n, mu)),
        }
    )

multinomial_cases = [
    ([2, 3, 5], [0.2, 0.3, 0.5], 10),
    ([0, 0, 10], [0.1, 0.1, 0.8], 10),
    ([250, 250, 500], [0.25, 0.25, 0.5], 1000),
    ([1, 0, 0, 0], [0.7, 0.1, 0.1, 0.1], 1),
]

for counts, probs, n in multinomial_cases:
    cases.append(
        {
            "fn": "multinomialLogPmf",
            "counts": counts,
            "params": {"n": n, "probs": probs},
            "expected": float(multinomial.logpmf(counts, n, probs)),
        }
    )

for n, k in [(10, 3), (1000, 500), (0, 0), (200, 100), (170, 85), (2000, 1)]:
    expected = float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))
    cases.append({"fn": "logBinomialCoefficient", "n": n, "k": k, "expected": expected})

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"wrote {len(cases)} cases to {OUT}")
