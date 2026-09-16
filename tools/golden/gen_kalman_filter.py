"""Golden fixtures for the Kalman filter half of packages/math/src/sequential/kalman.ts.

Plain numpy implementing the filter recursion directly from PRML 13.85-13.97: predict
with A/Gamma, correct with the Kalman gain built from C/Sigma. scipy.stats.multivariate_normal
supplies the innovation density c_n (13.91/13.96), independent of the TypeScript algorithm.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

# A constant-velocity model tracking a 2-D position: state = (x, y, vx, vy).
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
    means, covs, predicted_means, predicted_covs, gains, cs, log_cs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    mean, cov = mu0, V0
    for n in range(N):
        if n == 0:
            pred_mean, pred_cov = mu0, V0
        else:
            pred_mean = A @ mean
            pred_cov = A @ cov @ A.T + Gamma
        innovation_cov = C @ pred_cov @ C.T + Sigma
        gain = pred_cov @ C.T @ np.linalg.inv(innovation_cov)
        predicted_obs = C @ pred_mean
        mean = pred_mean + gain @ (observations[n] - predicted_obs)
        cov = (np.eye(4) - gain @ C) @ pred_cov
        c = float(
            multivariate_normal.pdf(observations[n], predicted_obs, innovation_cov)
        )
        log_c = float(
            multivariate_normal.logpdf(observations[n], predicted_obs, innovation_cov)
        )

        means.append(mean.tolist())
        covs.append(cov.tolist())
        predicted_means.append(pred_mean.tolist())
        predicted_covs.append(pred_cov.tolist())
        gains.append(gain.tolist())
        cs.append(c)
        log_cs.append(log_c)
    return means, covs, predicted_means, predicted_covs, gains, cs, log_cs


means, covs, predicted_means, predicted_covs, gains, cs, log_cs = kalman_filter(
    observations
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
        "fn": "kalmanFilter",
        "observations": observations.tolist(),
        "params": params,
        "expected": {
            "mean": means,
            "cov": covs,
            "predictedMean": predicted_means,
            "predictedCov": predicted_covs,
            "gain": gains,
            "c": cs,
            "logC": log_cs,
        },
    },
    {
        "fn": "kalmanLogLikelihood",
        "logC": log_cs,
        "expected": float(sum(log_cs)),
    },
]

(FIXTURES / "kalman_filter.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'kalman_filter.json'}")
