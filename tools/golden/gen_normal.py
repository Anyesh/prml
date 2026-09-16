import json
import os

from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "normal.json")

cases = []

# (x, mu, sigma2)
pdf_params = [
    (0.0, 0.0, 1.0),
    (5.0, 0.0, 1.0),
    (-10.0, 0.0, 1.0),
    (1e6, 0.0, 1.0),
    (2.0, 3.0, 0.0001),
    (100.0, 0.0, 1e6),
    (0.0, -5.0, 2.5),
    (-40.0, -5.0, 2.5),
]

for x, mu, sigma2 in pdf_params:
    sigma = sigma2**0.5
    cases.append(
        {
            "fn": "logPdf",
            "x": x,
            "params": {"mu": mu, "sigma2": sigma2},
            "expected": float(norm.logpdf(x, loc=mu, scale=sigma)),
        }
    )
    cases.append(
        {
            "fn": "pdf",
            "x": x,
            "params": {"mu": mu, "sigma2": sigma2},
            "expected": float(norm.pdf(x, loc=mu, scale=sigma)),
        }
    )

cdf_params = [
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (-1.0, 0.0, 1.0),
    (5.0, 0.0, 1.0),
    (-5.0, 0.0, 1.0),
    (6.0, 0.0, 1.0),
    (-6.0, 0.0, 1.0),
    (8.0, 0.0, 1.0),
    (10.0, 3.0, 4.0),
    (-100.0, 3.0, 4.0),
]

for x, mu, sigma2 in cdf_params:
    sigma = sigma2**0.5
    cases.append(
        {
            "fn": "cdf",
            "x": x,
            "params": {"mu": mu, "sigma2": sigma2},
            "expected": float(norm.cdf(x, loc=mu, scale=sigma)),
        }
    )

quantile_params = [
    (0.5, 0.0, 1.0),
    (0.975, 0.0, 1.0),
    (0.025, 0.0, 1.0),
    (0.001, 0.0, 1.0),
    (0.999, 0.0, 1.0),
    (1e-6, 0.0, 1.0),
    (1 - 1e-6, 0.0, 1.0),
    (0.5, 3.0, 4.0),
    (0.1, -5.0, 0.01),
]

for q, mu, sigma2 in quantile_params:
    sigma = sigma2**0.5
    cases.append(
        {
            "fn": "quantile",
            "q": q,
            "params": {"mu": mu, "sigma2": sigma2},
            "expected": float(norm.ppf(q, loc=mu, scale=sigma)),
        }
    )

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"wrote {len(cases)} cases to {OUT}")
