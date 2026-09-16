import json
import os

import numpy as np
from scipy.stats import vonmises
from scipy.special import ive
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "vonMises.json")

cases = []

# (theta, mu, kappa)
pdf_params = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.0, 5.0),
    (3.0, 0.0, 5.0),
    (0.0, 1.5, 10.0),
    (np.pi, 0.0, 1e-6),
    (0.1, 0.0, 700.0),
    (-3.14, 3.14, 2.0),
    (2.0, -2.0, 50.0),
]

for theta, mu, kappa in pdf_params:
    cases.append(
        {
            "fn": "logPdf",
            "theta": float(theta),
            "params": {"mu": mu, "kappa": kappa},
            "expected": float(vonmises.logpdf(theta, kappa, loc=mu)),
        }
    )
    cases.append(
        {
            "fn": "pdf",
            "theta": float(theta),
            "params": {"mu": mu, "kappa": kappa},
            "expected": float(vonmises.pdf(theta, kappa, loc=mu)),
        }
    )


def resultant_ratio(kappa):
    return ive(1, kappa) / ive(0, kappa)


def fit_kappa(theta):
    theta = np.asarray(theta)
    c = np.mean(np.cos(theta))
    s = np.mean(np.sin(theta))
    r = float(np.hypot(c, s))
    mu = float(np.arctan2(s, c))
    if r < 1e-12:
        return mu, 0.0
    if r > 1 - 1e-12:
        return mu, 1e6
    kappa = brentq(lambda k: resultant_ratio(k) - r, 1e-8, 1e6)
    return mu, float(kappa)


rng = np.random.default_rng(1234)

fit_cases = [
    rng.vonmises(0.0, 20.0, size=200).tolist(),
    rng.vonmises(1.5, 2.0, size=300).tolist(),
    rng.uniform(-np.pi, np.pi, size=400).tolist(),
    [0.0, np.pi / 2, np.pi, -np.pi / 2],
    rng.vonmises(-1.0, 200.0, size=150).tolist(),
]

for theta in fit_cases:
    mu, kappa = fit_kappa(theta)
    cases.append({"fn": "fit", "theta": theta, "expected": {"mu": mu, "kappa": kappa}})

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"wrote {len(cases)} cases to {OUT}")
