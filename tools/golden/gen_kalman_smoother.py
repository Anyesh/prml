"""Golden fixtures for the RTS smoother half of packages/math/src/sequential/kalman.ts.

Plain numpy: runs the same filter recursion as gen_kalman_filter.py (same model and
observations, so the filter fixture's values are reproduced here as a sanity check) and
then the backward RTS recursion (PRML 13.100-13.104), computed directly from the
definitions.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

dt = 1.0
A = np.array(
    [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
Gamma = np.diag([0.01, 0.01, 0.05, 0.05])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Sigma = np.diag([0.3, 0.3])
mu0 = np.array([0.0, 0.0, 1.0, 0.5])
V0 = np.diag([1.0, 1.0, 1.0, 1.0])

rng = np.random.default_rng(20260917)
N = 6
true_state = mu0.copy()
observations = []
for n in range(N):
    true_state = A @ true_state + rng.multivariate_normal(np.zeros(4), Gamma)
    observations.append(C @ true_state + rng.multivariate_normal(np.zeros(2), Sigma))
observations = np.array(observations)


def kalman_filter(observations):
    means, covs, predicted_means, predicted_covs = [], [], [], []
    mean, cov = mu0, V0
    for n in range(N):
        if n == 0:
            pred_mean, pred_cov = mu0, V0
        else:
            pred_mean = A @ mean
            pred_cov = A @ cov @ A.T + Gamma
        innovation_cov = C @ pred_cov @ C.T + Sigma
        gain = pred_cov @ C.T @ np.linalg.inv(innovation_cov)
        mean = pred_mean + gain @ (observations[n] - C @ pred_mean)
        cov = (np.eye(4) - gain @ C) @ pred_cov
        means.append(mean)
        covs.append(cov)
        predicted_means.append(pred_mean)
        predicted_covs.append(pred_cov)
    return means, covs, predicted_means, predicted_covs


means, covs, predicted_means, predicted_covs = kalman_filter(observations)


def kalman_smoother(means, covs, predicted_means, predicted_covs):
    smoothed_means = [None] * N
    smoothed_covs = [None] * N
    Js = [None] * (N - 1)
    pairwise = [None] * (N - 1)

    smoothed_means[N - 1] = means[N - 1]
    smoothed_covs[N - 1] = covs[N - 1]

    for n in range(N - 2, -1, -1):
        J = covs[n] @ A.T @ np.linalg.inv(predicted_covs[n + 1])
        Js[n] = J
        smoothed_means[n] = means[n] + J @ (
            smoothed_means[n + 1] - predicted_means[n + 1]
        )
        smoothed_covs[n] = (
            covs[n] + J @ (smoothed_covs[n + 1] - predicted_covs[n + 1]) @ J.T
        )
        pairwise[n] = J @ smoothed_covs[n + 1]

    return smoothed_means, smoothed_covs, Js, pairwise


smoothed_means, smoothed_covs, Js, pairwise = kalman_smoother(
    means, covs, predicted_means, predicted_covs
)

params = {
    "A": A.tolist(),
    "Gamma": Gamma.tolist(),
    "C": C.tolist(),
    "Sigma": Sigma.tolist(),
    "mu0": mu0.tolist(),
    "V0": V0.tolist(),
}

cases = [
    {
        "fn": "kalmanSmoother",
        "observations": observations.tolist(),
        "params": params,
        "expected": {
            "mean": [m.tolist() for m in smoothed_means],
            "cov": [v.tolist() for v in smoothed_covs],
            "J": [j.tolist() for j in Js],
            "pairwiseCov": [p.tolist() for p in pairwise],
        },
    }
]

(FIXTURES / "kalman_smoother.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'kalman_smoother.json'}")
