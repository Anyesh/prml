"""Golden fixtures for the EM (M-step) half of packages/math/src/sequential/kalman.ts.

Plain numpy: runs the same filter+smoother as gen_kalman_smoother.py, builds the
expectations PRML 13.105-13.107 need, and applies the closed-form M-step update
13.110-13.116 directly from those definitions.
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
    pairwise = [None] * (N - 1)

    smoothed_means[N - 1] = means[N - 1]
    smoothed_covs[N - 1] = covs[N - 1]

    for n in range(N - 2, -1, -1):
        J = covs[n] @ A.T @ np.linalg.inv(predicted_covs[n + 1])
        smoothed_means[n] = means[n] + J @ (
            smoothed_means[n + 1] - predicted_means[n + 1]
        )
        smoothed_covs[n] = (
            covs[n] + J @ (smoothed_covs[n + 1] - predicted_covs[n + 1]) @ J.T
        )
        pairwise[n] = J @ smoothed_covs[n + 1]

    return smoothed_means, smoothed_covs, pairwise


smoothed_means, smoothed_covs, pairwise = kalman_smoother(
    means, covs, predicted_means, predicted_covs
)

Ez = [smoothed_means[n] for n in range(N)]
Ezz = [
    smoothed_covs[n] + np.outer(smoothed_means[n], smoothed_means[n]) for n in range(N)
]
# Ezzlag[t] = E[z_{t+1} z_t^T], t = 0..N-2.
Ezzlag = [
    pairwise[t] + np.outer(smoothed_means[t + 1], smoothed_means[t])
    for t in range(N - 1)
]

# M step, PRML 13.110-13.116.
mu0_new = Ez[0]
V0_new = Ezz[0] - np.outer(Ez[0], Ez[0])

sum_Ezzlag = sum(Ezzlag)
sum_Ezz_prev = sum(Ezz[t] for t in range(N - 1))
A_new = sum_Ezzlag @ np.linalg.inv(sum_Ezz_prev)

Gamma_terms = []
for t in range(N - 1):
    n = t + 1
    term = Ezz[n] - A_new @ Ezzlag[t].T - Ezzlag[t] @ A_new.T + A_new @ Ezz[t] @ A_new.T
    Gamma_terms.append(term)
Gamma_new = sum(Gamma_terms) / (N - 1)

sum_x_Ez = sum(np.outer(observations[n], Ez[n]) for n in range(N))
sum_Ezz_all = sum(Ezz)
C_new = sum_x_Ez @ np.linalg.inv(sum_Ezz_all)

Sigma_terms = []
for n in range(N):
    x = observations[n]
    term = (
        np.outer(x, x)
        - C_new @ np.outer(Ez[n], x)
        - np.outer(x, Ez[n]) @ C_new.T
        + C_new @ Ezz[n] @ C_new.T
    )
    Sigma_terms.append(term)
Sigma_new = sum(Sigma_terms) / N

cases = [
    {
        "fn": "kalmanEmExpectations",
        "smoothedMean": [m.tolist() for m in smoothed_means],
        "smoothedCov": [v.tolist() for v in smoothed_covs],
        "pairwiseCov": [p.tolist() for p in pairwise],
        "expected": {
            "Ez": [v.tolist() for v in Ez],
            "Ezz": [m.tolist() for m in Ezz],
            "Ezzlag": [m.tolist() for m in Ezzlag],
        },
    },
    {
        "fn": "kalmanMStep",
        "observations": observations.tolist(),
        "Ez": [v.tolist() for v in Ez],
        "Ezz": [m.tolist() for m in Ezz],
        "Ezzlag": [m.tolist() for m in Ezzlag],
        "expected": {
            "A": A_new.tolist(),
            "Gamma": Gamma_new.tolist(),
            "C": C_new.tolist(),
            "Sigma": Sigma_new.tolist(),
            "mu0": mu0_new.tolist(),
            "V0": V0_new.tolist(),
        },
    },
]

(FIXTURES / "kalman_em.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'kalman_em.json'}")
