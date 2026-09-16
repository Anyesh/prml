"""Golden fixtures for packages/math/src/ensemble/weightedLeastSquares.ts.

`weightedLeastSquaresFit` is the shared M-step primitive PRML 14.42 needs for both the
mixture of linear regression models (14.5.1) and the regression experts of the mixture of
experts (14.5.3): minimising sum_n w_n (t_n - phi_n^T w)^2 is ordinary least squares on a
design whose rows are pre-scaled by sqrt(w_n), so the fixture checks that identity directly
against an independent weighted normal-equations solve (not the row-scaling trick itself).

Two-line synthetic data (mirrors book figure 14.8's two-regime toy set) with responsibility-
shaped weights, plus edge cases: uniform weights (must reduce to plain OLS), and a highly
skewed weight vector where most of the mass sits on one line's points.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260917)


def design_matrix(x):
    return np.stack([np.ones_like(x), x], axis=1)


def weighted_normal_equations(phi, t, w):
    wmat = np.diag(w)
    gram = phi.T @ wmat @ phi
    rhs = phi.T @ wmat @ t
    return np.linalg.solve(gram, rhs)


N = 30
X = np.sort(rng.uniform(-1.0, 1.0, N))
T = 0.5 + 1.5 * X + rng.normal(0.0, 0.05, N)
PHI = design_matrix(X)

cases = []

# Uniform weights: must reduce to ordinary least squares.
uniform_w = np.ones(N)
cases.append(
    {
        "fn": "weightedLeastSquaresFit",
        "design": PHI.tolist(),
        "targets": T.tolist(),
        "weights": uniform_w.tolist(),
        "expected": weighted_normal_equations(PHI, T, uniform_w).tolist(),
    }
)

# Smooth responsibility-shaped weights, favouring the left half of the data.
smooth_w = 1.0 / (1.0 + np.exp(4.0 * X))
cases.append(
    {
        "fn": "weightedLeastSquaresFit",
        "design": PHI.tolist(),
        "targets": T.tolist(),
        "weights": smooth_w.tolist(),
        "expected": weighted_normal_equations(PHI, T, smooth_w).tolist(),
    }
)

# Highly skewed weights: almost all mass on the first third of the points.
skew_w = np.full(N, 1e-4)
skew_w[:10] = 1.0
cases.append(
    {
        "fn": "weightedLeastSquaresFit",
        "design": PHI.tolist(),
        "targets": T.tolist(),
        "weights": skew_w.tolist(),
        "expected": weighted_normal_equations(PHI, T, skew_w).tolist(),
    }
)

# Three-column design (quadratic term), so the fit is exercised beyond 2 parameters.
PHI3 = np.stack([np.ones_like(X), X, X**2], axis=1)
w3 = 0.3 + 0.7 * rng.uniform(0.0, 1.0, N)
cases.append(
    {
        "fn": "weightedLeastSquaresFit",
        "design": PHI3.tolist(),
        "targets": T.tolist(),
        "weights": w3.tolist(),
        "expected": weighted_normal_equations(PHI3, T, w3).tolist(),
    }
)

fixture = {"cases": cases}
(FIXTURES / "weightedLeastSquares.json").write_text(json.dumps(fixture, indent=2))
print(f"wrote {len(cases)} cases to weightedLeastSquares.json")
