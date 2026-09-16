"""Golden fixtures for packages/math/src/variational/factorizedGaussian.ts.

The forward-KL fixed point and the reverse-KL marginals are both closed forms (PRML
10.12-10.17); this script derives each independently with plain numpy linear algebra
rather than re-deriving the TypeScript's own coordinate-ascent recursion, and unrolls the
recursion itself for the forward-KL trajectory case so the iteration count is not assumed
to match.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []


def forward_kl_step(mean, precision, q1, q2):
    mu1, mu2 = mean
    l11, l12 = precision[0]
    l21, l22 = precision[1]
    m1 = mu1 - (l12 / l11) * (q2[0] - mu2)
    q1_new = (m1, 1.0 / l11)
    m2 = mu2 - (l21 / l22) * (m1 - mu1)
    q2_new = (m2, 1.0 / l22)
    return q1_new, q2_new


# Case 1: forward-KL trajectory from a deliberately wrong initial guess, correlated 2x2 Gaussian.
mean = [0.3, -0.7]
precision = [[2.0, 0.9], [0.9, 1.5]]
initial = ((5.0, 1.0), (-4.0, 1.0))
trajectory = [initial]
q1, q2 = initial[0], initial[1]
for _ in range(6):
    q1, q2 = forward_kl_step(mean, precision, q1, q2)
    trajectory.append((q1, q2))

cases.append(
    {
        "fn": "factorizedGaussianForwardKlFit",
        "p": {"mean": mean, "precision": precision},
        "initial": {
            "q1": {"mu": initial[0][0], "sigma2": initial[0][1]},
            "q2": {"mu": initial[1][0], "sigma2": initial[1][1]},
        },
        "iterations": 6,
        "expected": [
            {
                "q1": {"mu": t[0][0], "sigma2": t[0][1]},
                "q2": {"mu": t[1][0], "sigma2": t[1][1]},
            }
            for t in trajectory
        ],
    }
)

# Case 2: reverse-KL marginals, cross-checked against an independent matrix inverse.
prec_np = np.array(precision)
cov = np.linalg.inv(prec_np)
cases.append(
    {
        "fn": "factorizedGaussianReverseKl",
        "p": {"mean": mean, "precision": precision},
        "expected": {
            "q1": {"mu": mean[0], "sigma2": float(cov[0, 0])},
            "q2": {"mu": mean[1], "sigma2": float(cov[1, 1])},
        },
    }
)

# Case 3: a second, more strongly correlated example, to catch a sign error the first case's
# near-symmetric precision matrix could hide.
mean_b = [1.0, 2.0]
precision_b = [[4.0, -3.5], [-3.5, 4.0]]
cov_b = np.linalg.inv(np.array(precision_b))
cases.append(
    {
        "fn": "factorizedGaussianReverseKl",
        "p": {"mean": mean_b, "precision": precision_b},
        "expected": {
            "q1": {"mu": mean_b[0], "sigma2": float(cov_b[0, 0])},
            "q2": {"mu": mean_b[1], "sigma2": float(cov_b[1, 1])},
        },
    }
)

# The forward-KL fixed point itself: starting exactly at the true mean, one step must be a
# no-op, confirming (mu1, mu2) with precisions (Lambda11, Lambda22) is a genuine fixed point.
fixed_initial = ((mean[0], 1 / precision[0][0]), (mean[1], 1 / precision[1][1]))
q1f, q2f = forward_kl_step(mean, precision, fixed_initial[0], fixed_initial[1])
cases.append(
    {
        "fn": "factorizedGaussianForwardKlFit",
        "p": {"mean": mean, "precision": precision},
        "initial": {
            "q1": {"mu": fixed_initial[0][0], "sigma2": fixed_initial[0][1]},
            "q2": {"mu": fixed_initial[1][0], "sigma2": fixed_initial[1][1]},
        },
        "iterations": 1,
        "expected": [
            {
                "q1": {"mu": fixed_initial[0][0], "sigma2": fixed_initial[0][1]},
                "q2": {"mu": fixed_initial[1][0], "sigma2": fixed_initial[1][1]},
            },
            {
                "q1": {"mu": q1f[0], "sigma2": q1f[1]},
                "q2": {"mu": q2f[0], "sigma2": q2f[1]},
            },
        ],
    }
)

(FIXTURES / "factorizedGaussian.json").write_text(
    json.dumps({"cases": cases}, indent=2)
)
print(f"wrote {len(cases)} cases to {FIXTURES / 'factorizedGaussian.json'}")
