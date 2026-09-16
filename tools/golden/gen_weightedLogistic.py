"""Golden fixtures for packages/math/src/ensemble/weightedLogistic.ts.

`weightedLogisticFit` generalises PRML's IRLS (4.99-4.100) with a per-point weight w_n
multiplying the cross-entropy term, and it is the shared M-step primitive for both the
mixture of logistic regression models (14.5.2, weight = responsibility, target = the
actual 0/1 label) and the gating network of the mixture of experts (14.5.3, weight = 1,
target = a soft responsibility in [0, 1] rather than a hard label). The Newton step is the
same in both regimes because the cross-entropy gradient/Hessian identities (sum w_n(y_n -
t_n) phi_n, sum w_n y_n(1-y_n) phi_n phi_n^T) never assumed t_n in {0, 1}.

Reference solve is an independent Newton-Raphson in numpy (not a call into the eventual
TypeScript), run to a tight tolerance so both implementations land at the same stationary
point of a strictly convex objective.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260918)


def weighted_newton_raphson(phi, t, w, tol=1e-13, max_iter=100):
    d = phi.shape[1]
    weights = np.zeros(d)
    for _ in range(max_iter):
        a = phi @ weights
        y = expit(a)
        r = w * y * (1 - y)
        r = np.maximum(r, 1e-10)
        hessian = phi.T @ (phi * r[:, None])
        gradient = phi.T @ (w * (y - t))
        step = np.linalg.solve(hessian, gradient)
        next_weights = weights - step
        if np.max(np.abs(next_weights - weights)) <= tol:
            weights = next_weights
            break
        weights = next_weights
    return weights


def design_matrix(x1, x2):
    return np.stack([np.ones_like(x1), x1, x2], axis=1)


N = 60
X1 = rng.normal(0.0, 1.0, N)
X2 = rng.normal(0.0, 1.0, N)
PHI = design_matrix(X1, X2)
TRUE_W = np.array([0.3, 1.4, -0.9])
PROB = expit(PHI @ TRUE_W)
LABELS = (rng.uniform(0.0, 1.0, N) < PROB).astype(float)

cases = []

# Hard binary labels, uniform weights: ordinary (unweighted) logistic regression.
cases.append(
    {
        "fn": "weightedLogisticFit",
        "design": PHI.tolist(),
        "targets": LABELS.tolist(),
        "weights": np.ones(N).tolist(),
        "expected": weighted_newton_raphson(PHI, LABELS, np.ones(N)).tolist(),
    }
)

# Hard binary labels, responsibility-shaped weights (mixture-of-logistic-models M-step).
resp_w = expit(0.8 * X1)
cases.append(
    {
        "fn": "weightedLogisticFit",
        "design": PHI.tolist(),
        "targets": LABELS.tolist(),
        "weights": resp_w.tolist(),
        "expected": weighted_newton_raphson(PHI, LABELS, resp_w).tolist(),
    }
)

# Soft targets in (0, 1), uniform weights (mixture-of-experts gating M-step).
soft_t = expit(0.6 * X1 - 0.4 * X2)
cases.append(
    {
        "fn": "weightedLogisticFit",
        "design": PHI.tolist(),
        "targets": soft_t.tolist(),
        "weights": np.ones(N).tolist(),
        "expected": weighted_newton_raphson(PHI, soft_t, np.ones(N)).tolist(),
    }
)

fixture = {"cases": cases}
(FIXTURES / "weightedLogistic.json").write_text(json.dumps(fixture, indent=2))
print(f"wrote {len(cases)} cases to weightedLogistic.json")
