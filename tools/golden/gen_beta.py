import json
import os

from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "beta.json")

cases = []

pdf_params = [
    (0.5, 2.0, 3.0),
    (0.001, 0.5, 0.5),
    (0.999, 5.0, 2.0),
    (1e-6, 1.0, 1.0),
    (0.5, 1000.0, 1000.0),
    (0.0, 1.0, 3.0),
    (1.0, 3.0, 1.0),
    (0.3, 0.001, 0.001),
    (0.999999, 0.5, 0.5),
]

for x, a, b in pdf_params:
    cases.append(
        {
            "fn": "logPdf",
            "x": x,
            "params": {"a": a, "b": b},
            "expected": float(beta_dist.logpdf(x, a, b)),
        }
    )
    cases.append(
        {
            "fn": "pdf",
            "x": x,
            "params": {"a": a, "b": b},
            "expected": float(beta_dist.pdf(x, a, b)),
        }
    )

cdf_params = [
    (0.5, 0.5, 0.5),
    (0.5, 2.0, 3.0),
    (0.1, 5.0, 5.0),
    (0.9, 5.0, 5.0),
    (0.001, 0.5, 0.5),
    (0.999, 0.5, 0.5),
    (0.5, 1000.0, 1000.0),
]

for x, a, b in cdf_params:
    cases.append(
        {
            "fn": "cdf",
            "x": x,
            "params": {"a": a, "b": b},
            "expected": float(beta_dist.cdf(x, a, b)),
        }
    )

mean_var_params = [
    (2.0, 3.0),
    (0.5, 0.5),
    (1.0, 1.0),
    (1000.0, 1.0),
    (0.001, 0.001),
    (1e6, 1e6),
]

for a, b in mean_var_params:
    m, v = beta_dist.stats(a, b, moments="mv")
    cases.append({"fn": "mean", "params": {"a": a, "b": b}, "expected": float(m)})
    cases.append({"fn": "variance", "params": {"a": a, "b": b}, "expected": float(v)})

posterior_params = [
    ({"a": 1.0, "b": 1.0}, 5, 3),
    ({"a": 2.5, "b": 7.1}, 0, 10),
    ({"a": 0.5, "b": 0.5}, 100, 0),
]

for prior, successes, failures in posterior_params:
    cases.append(
        {
            "fn": "posterior",
            "prior": prior,
            "successes": successes,
            "failures": failures,
            "expected": {"a": prior["a"] + successes, "b": prior["b"] + failures},
        }
    )

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"wrote {len(cases)} cases to {OUT}")
