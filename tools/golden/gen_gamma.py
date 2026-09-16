import json
import os

from scipy.stats import gamma as gamma_dist

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "gamma.json")

cases = []

# (x, shape, rate) - scipy is shape-scale, so scale = 1 / rate.
pdf_params = [
    (1.0, 2.0, 1.0),
    (0.001, 0.5, 1.0),
    (1e6, 2.0, 1e-4),
    (1.0, 1.0, 1.0),
    (100.0, 50.0, 1.0),
    (1e-8, 2.0, 1.0),
    (0.001, 1.0, 1.0),
    (1e-8, 0.5, 1.0),
    (500.0, 1000.0, 5.0),
]

for x, shape, rate in pdf_params:
    scale = 1.0 / rate
    cases.append(
        {
            "fn": "logPdf",
            "x": x,
            "params": {"shape": shape, "rate": rate},
            "expected": float(gamma_dist.logpdf(x, shape, scale=scale)),
        }
    )
    cases.append(
        {
            "fn": "pdf",
            "x": x,
            "params": {"shape": shape, "rate": rate},
            "expected": float(gamma_dist.pdf(x, shape, scale=scale)),
        }
    )

cdf_params = [
    (1.0, 2.0, 1.0),
    (50.0, 10.0, 1.0),
    (0.001, 0.5, 1.0),
    (1e6, 5.0, 1e-5),
    (200.0, 100.0, 1.0),
]

for x, shape, rate in cdf_params:
    scale = 1.0 / rate
    cases.append(
        {
            "fn": "cdf",
            "x": x,
            "params": {"shape": shape, "rate": rate},
            "expected": float(gamma_dist.cdf(x, shape, scale=scale)),
        }
    )

mean_var_params = [
    (2.0, 1.0),
    (0.5, 2.0),
    (1000.0, 1.0),
    (0.001, 1000.0),
    (1e6, 1e6),
]

for shape, rate in mean_var_params:
    scale = 1.0 / rate
    m, v = gamma_dist.stats(shape, scale=scale, moments="mv")
    cases.append(
        {"fn": "mean", "params": {"shape": shape, "rate": rate}, "expected": float(m)}
    )
    cases.append(
        {
            "fn": "variance",
            "params": {"shape": shape, "rate": rate},
            "expected": float(v),
        }
    )

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"wrote {len(cases)} cases to {OUT}")
