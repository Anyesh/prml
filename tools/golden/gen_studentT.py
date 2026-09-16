import json
import os

import numpy as np
from scipy.stats import t as t_dist
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "studentT.json")

cases = []

# (x, mu, scale2, nu) - scipy's t.pdf(x, df, loc, scale) takes scale (not scale2).
pdf_params = [
    (0.0, 0.0, 1.0, 5.0),
    (10.0, 0.0, 1.0, 3.0),
    (0.0, 0.0, 1.0, 1e6),
    (0.0, 0.0, 1.0, 1.0),
    (100.0, 0.0, 1.0, 1.0),
    (5.0, 2.0, 4.0, 2.5),
    (-50.0, 0.0, 0.01, 0.5),
    (1e-6, 0.0, 1.0, 30.0),
]

for x, mu, scale2, nu in pdf_params:
    scale = scale2**0.5
    cases.append(
        {
            "fn": "logPdf",
            "x": x,
            "params": {"mu": mu, "scale2": scale2, "nu": nu},
            "expected": float(t_dist.logpdf(x, nu, loc=mu, scale=scale)),
        }
    )
    cases.append(
        {
            "fn": "pdf",
            "x": x,
            "params": {"mu": mu, "scale2": scale2, "nu": nu},
            "expected": float(t_dist.pdf(x, nu, loc=mu, scale=scale)),
        }
    )

cdf_params = [
    (0.0, 0.0, 1.0, 5.0),
    (2.0, 0.0, 1.0, 5.0),
    (-2.0, 0.0, 1.0, 5.0),
    (100.0, 0.0, 1.0, 1.0),
    (-100.0, 0.0, 1.0, 1.0),
    (10.0, 0.0, 1.0, 1e6),
    (5.0, 2.0, 4.0, 3.0),
]

for x, mu, scale2, nu in cdf_params:
    scale = scale2**0.5
    cases.append(
        {
            "fn": "cdf",
            "x": x,
            "params": {"mu": mu, "scale2": scale2, "nu": nu},
            "expected": float(t_dist.cdf(x, nu, loc=mu, scale=scale)),
        }
    )


def multivariate_t_logpdf(x, mean, scale, nu):
    x = np.array(x)
    mean = np.array(mean)
    scale = np.array(scale)
    d = len(x)
    diff = x - mean
    inv = np.linalg.inv(scale)
    delta = diff @ inv @ diff
    sign, logdet = np.linalg.slogdet(scale)
    return (
        gammaln((nu + d) / 2)
        - gammaln(nu / 2)
        - (d / 2) * np.log(nu * np.pi)
        - 0.5 * logdet
        - ((nu + d) / 2) * np.log(1 + delta / nu)
    )


mvt_cases = [
    ([0.5, -0.5], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 5.0),
    (
        [2.0, 1.0, -1.0],
        [0.0, 0.0, 0.0],
        [[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 1.5]],
        3.0,
    ),
    ([10.0, 10.0], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 1.0),
    ([0.1, 0.2], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 1e6),
]

for x, mean, scale, nu in mvt_cases:
    cases.append(
        {
            "fn": "multivariateLogPdf",
            "x": x,
            "params": {"mean": mean, "scale": scale, "nu": nu},
            "expected": float(multivariate_t_logpdf(x, mean, scale, nu)),
        }
    )

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"wrote {len(cases)} cases to {OUT}")
