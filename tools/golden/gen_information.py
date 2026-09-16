import json
import os

import numpy as np
from scipy import integrate
from scipy.stats import entropy, norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "information.json")

cases = []

# discreteEntropy: natural log (nats), matching PRML's switch to ln part-way through 1.6.
discrete_cases = [
    [0.5, 0.5],
    [1.0 / 8] * 8,
    # PRML's own worked example (page 50): the skewed 8-state code-length distribution.
    [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 64, 1 / 64, 1 / 64, 1 / 64],
    [0.999, 0.0005, 0.0005],
    [0.5, 0.5, 0.0],
    [1.0 / 30] * 30,
    [1.0],
]
for p in discrete_cases:
    p = np.array(p)
    cases.append(
        {"fn": "discreteEntropy", "p": p.tolist(), "expected": float(entropy(p))}
    )

# differentialEntropyGaussian: scipy.stats.norm's own closed-form entropy method, an
# implementation independent of the algebra the TypeScript function performs.
for sigma2 in [1.0, 1.0 / (2 * np.pi * np.e), 0.01, 100.0, 1e6, 1e-6]:
    sigma = sigma2**0.5
    cases.append(
        {
            "fn": "differentialEntropyGaussian",
            "sigma2": sigma2,
            "expected": float(norm(scale=sigma).entropy()),
        }
    )

# klDivergence (discrete): scipy.stats.entropy(p, q) computes sum(rel_entr(p, q)) directly.
kl_pairs = [
    ([0.5, 0.5], [0.5, 0.5]),
    ([0.5, 0.25, 0.25], [0.25, 0.25, 0.5]),
    ([0.9, 0.05, 0.05], [0.3, 0.3, 0.4]),
    ([0.3, 0.3, 0.4], [0.9, 0.05, 0.05]),
    ([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]),
    ([0.999, 0.001], [0.5, 0.5]),
]
for p, q in kl_pairs:
    p_arr, q_arr = np.array(p), np.array(q)
    cases.append(
        {"fn": "klDivergence", "p": p, "q": q, "expected": float(entropy(p_arr, q_arr))}
    )

# gaussianKlDivergence: numerical integration of p(x) ln(p(x)/q(x)) over the real line,
# independent of the closed-form algebra the TypeScript implementation evaluates.
gaussian_pairs = [
    ((0.0, 1.0), (0.0, 1.0)),
    ((0.0, 1.0), (1.0, 1.0)),
    ((0.0, 1.0), (0.0, 4.0)),
    ((2.0, 0.5), (-1.0, 3.0)),
    ((0.0, 0.01), (0.0, 100.0)),
    ((5.0, 2.0), (5.5, 2.2)),
]
for (mu_p, var_p), (mu_q, var_q) in gaussian_pairs:
    sigma_p, sigma_q = var_p**0.5, var_q**0.5

    def integrand(x, mu_p=mu_p, sigma_p=sigma_p, mu_q=mu_q, sigma_q=sigma_q):
        return norm.pdf(x, loc=mu_p, scale=sigma_p) * (
            norm.logpdf(x, loc=mu_p, scale=sigma_p)
            - norm.logpdf(x, loc=mu_q, scale=sigma_q)
        )

    value, _ = integrate.quad(
        integrand, -np.inf, np.inf, limit=200, epsabs=1e-13, epsrel=1e-13
    )
    cases.append(
        {
            "fn": "gaussianKlDivergence",
            "p": {"mu": mu_p, "sigma2": var_p},
            "q": {"mu": mu_q, "sigma2": var_q},
            "expected": float(value),
        }
    )

# mutualInformation / conditionalEntropy: joint tables, cross-checked through the entropy
# identity I[X,Y] = H[X] + H[Y] - H[X,Y] and H[Y|X] = H[X,Y] - H[X], both built from
# scipy.stats.entropy on the flattened joint and its marginals rather than a direct sum
# over the table (which is the derivation the TypeScript implementation performs).
joint_tables = [
    # Independent by construction: outer product of two marginals.
    (np.outer([0.5, 0.5], [0.3, 0.7])).tolist(),
    # PRML Exercise 1.39, Table 1.3.
    [[1 / 3, 1 / 3], [0.0, 1 / 3]],
    # Deterministic: Y is a function of X, so H[Y|X] = 0.
    [[0.25, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.25]],
    # A generic asymmetric 3x2 table.
    [[0.10, 0.15], [0.25, 0.05], [0.20, 0.25]],
]
for joint in joint_tables:
    j = np.array(joint)
    p_x = j.sum(axis=1)
    p_y = j.sum(axis=0)
    h_x = float(entropy(p_x))
    h_y = float(entropy(p_y))
    h_xy = float(entropy(j.flatten()))
    cases.append(
        {"fn": "mutualInformation", "joint": joint, "expected": h_x + h_y - h_xy}
    )
    cases.append({"fn": "conditionalEntropy", "joint": joint, "expected": h_xy - h_x})

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"wrote {len(cases)} cases to {OUT}")
