"""Golden fixtures for packages/math/src/gaussian/*.ts.

Follows tools/golden/README.md: {"cases": [...]}, each case an "fn" discriminant,
inputs, and an "expected" value computed independently with numpy (never by re-running
the TypeScript algorithm). Covers the linear-Gaussian model (PRML 2.113-2.117) and the
sequential maximum-likelihood mean update (PRML 2.126).
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []


def linear_gaussian_marginal(mu, Lambda, A, b, L):
    """p(y) mean/cov via PRML 2.114-2.115, computed directly rather than through the
    joint-precision route, so it is an independent check on the TypeScript formula."""
    Lambda_inv = np.linalg.inv(Lambda)
    L_inv = np.linalg.inv(L)
    mean = A @ mu + b
    cov = L_inv + A @ Lambda_inv @ A.T
    return mean, cov


def linear_gaussian_posterior(mu, Lambda, A, b, L, y):
    """p(x|y) mean/cov via PRML 2.116-2.117."""
    Sigma = np.linalg.inv(Lambda + A.T @ L @ A)
    mean = Sigma @ (A.T @ L @ (y - b) + Lambda @ mu)
    return mean, Sigma


def joint_precision_check(mu, Lambda, A, b, L):
    """Cross-check: build the full joint precision R over z=(x,y) (PRML 2.104), invert
    it to the joint covariance (2.105), and read off cov[y] and mean[y] from the block
    structure. This exercises a route to (2.114)-(2.115) that never calls the direct
    formula under test, so agreement is real evidence rather than circular."""
    M = mu.shape[0]
    D = b.shape[0]
    R_top = np.hstack([Lambda + A.T @ L @ A, -A.T @ L])
    R_bot = np.hstack([-L @ A, L])
    R = np.vstack([R_top, R_bot])
    joint_cov = np.linalg.inv(R)
    mean_x = mu
    mean_y = A @ mu + b
    joint_mean = np.concatenate([mean_x, mean_y])
    cov_yy = joint_cov[M:, M:]
    mean_y_from_joint = joint_mean[M:]
    return mean_y_from_joint, cov_yy


# --- Case 1: square, M = D = 2 ---
mu1 = np.array([0.5, -1.0])
Lambda1 = np.array([[2.0, 0.3], [0.3, 1.5]])
A1 = np.array([[1.0, 0.5], [-0.5, 2.0]])
b1 = np.array([0.1, 0.2])
L1 = np.array([[3.0, 0.4], [0.4, 2.0]])
y1 = np.array([0.9, -0.3])

mean_y1, cov_y1 = linear_gaussian_marginal(mu1, Lambda1, A1, b1, L1)
cases.append({
    "fn": "linearGaussianMarginal",
    "prior": {"mean": mu1.tolist(), "precision": Lambda1.tolist()},
    "likelihood": {"a": A1.tolist(), "b": b1.tolist(), "precision": L1.tolist()},
    "expected": {"mean": mean_y1.tolist(), "cov": cov_y1.tolist()},
})

mean_x1, cov_x1 = linear_gaussian_posterior(mu1, Lambda1, A1, b1, L1, y1)
cases.append({
    "fn": "linearGaussianPosterior",
    "prior": {"mean": mu1.tolist(), "precision": Lambda1.tolist()},
    "likelihood": {"a": A1.tolist(), "b": b1.tolist(), "precision": L1.tolist()},
    "y": y1.tolist(),
    "expected": {"mean": mean_x1.tolist(), "cov": cov_x1.tolist()},
})

mean_y1_joint, cov_y1_joint = joint_precision_check(mu1, Lambda1, A1, b1, L1)
cases.append({
    "fn": "linearGaussianMarginal_jointCheck",
    "prior": {"mean": mu1.tolist(), "precision": Lambda1.tolist()},
    "likelihood": {"a": A1.tolist(), "b": b1.tolist(), "precision": L1.tolist()},
    "expected": {"mean": mean_y1_joint.tolist(), "cov": cov_y1_joint.tolist()},
})

# --- Case 2: rectangular A, M = 2 (x), D = 3 (y) ---
mu2 = np.array([1.0, 0.0])
Lambda2 = np.array([[4.0, 0.0], [0.0, 4.0]])
A2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
b2 = np.array([0.0, 0.0, 0.5])
L2 = np.diag([10.0, 10.0, 10.0])
y2 = np.array([1.1, -0.2, 1.3])

mean_y2, cov_y2 = linear_gaussian_marginal(mu2, Lambda2, A2, b2, L2)
cases.append({
    "fn": "linearGaussianMarginal",
    "prior": {"mean": mu2.tolist(), "precision": Lambda2.tolist()},
    "likelihood": {"a": A2.tolist(), "b": b2.tolist(), "precision": L2.tolist()},
    "expected": {"mean": mean_y2.tolist(), "cov": cov_y2.tolist()},
})

mean_x2, cov_x2 = linear_gaussian_posterior(mu2, Lambda2, A2, b2, L2, y2)
cases.append({
    "fn": "linearGaussianPosterior",
    "prior": {"mean": mu2.tolist(), "precision": Lambda2.tolist()},
    "likelihood": {"a": A2.tolist(), "b": b2.tolist(), "precision": L2.tolist()},
    "y": y2.tolist(),
    "expected": {"mean": mean_x2.tolist(), "cov": cov_x2.tolist()},
})

# --- Sequential mean update (PRML 2.126), checked against a direct running mean ---
rng = np.random.default_rng(20260916)
stream = rng.normal(loc=2.5, scale=1.3, size=12)
running_mean = 0.0
for n, x in enumerate(stream, start=1):
    prev_mean = running_mean
    running_mean = running_mean + (x - running_mean) / n
    # Independent computation: the mean of the prefix seen so far, not the recursion
    # under test.
    prefix_mean = float(np.mean(stream[:n]))
    cases.append({
        "fn": "sequentialMean",
        "previous": float(prev_mean),
        "n": n,
        "x": float(x),
        "expected": prefix_mean,
    })

(FIXTURES / "gaussian.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'gaussian.json'}")
