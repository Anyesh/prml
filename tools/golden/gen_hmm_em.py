"""Golden fixtures for packages/math/src/sequential/em.ts.

Plain numpy for the HMM EM steps (PRML 13.17-13.21): the E-step's gamma/xi come from the
same scaled forward-backward recursion as gen_hmm_forward_backward.py (verified there
against xi_unscaled), and the M-step is the closed-form update computed directly from
gamma and xi, not by re-deriving the TypeScript implementation.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

K = 2
pi = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.2, 0.8]])
means = np.array([[0.0, 0.0], [3.0, 3.0]])
covs = np.array([[[1.0, 0.2], [0.2, 1.0]], [[1.2, -0.3], [-0.3, 0.8]]])

rng = np.random.default_rng(20260917)
N = 8
data = np.concatenate(
    [
        rng.multivariate_normal(means[0], covs[0], size=4),
        rng.multivariate_normal(means[1], covs[1], size=4),
    ]
)


def emission_matrix(x):
    return np.array([multivariate_normal.pdf(x, means[k], covs[k]) for k in range(K)]).T


B = emission_matrix(data)


def forward_scaled(B):
    alpha_hat = np.zeros((N, K))
    c = np.zeros(N)
    raw0 = pi * B[0]
    c[0] = raw0.sum()
    alpha_hat[0] = raw0 / c[0]
    for n in range(1, N):
        raw = B[n] * (alpha_hat[n - 1] @ A)
        c[n] = raw.sum()
        alpha_hat[n] = raw / c[n]
    return alpha_hat, c


def backward_scaled(B, c):
    beta_hat = np.zeros((N, K))
    beta_hat[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        raw = (A * (B[n + 1] * beta_hat[n + 1])[None, :]).sum(axis=1)
        beta_hat[n] = raw / c[n + 1]
    return beta_hat


alpha_hat, c = forward_scaled(B)
beta_hat = backward_scaled(B, c)
gamma = alpha_hat * beta_hat

xi = []
for n in range(1, N):
    mat = (
        np.outer(alpha_hat[n - 1], np.ones(K))
        * A
        * B[n][None, :]
        * beta_hat[n][None, :]
        / c[n]
    )
    xi.append(mat)
xi = np.array(xi)

log_likelihood = float(np.log(c).sum())

# M step (PRML 13.18-13.21).
pi_new = gamma[0] / gamma[0].sum()
xi_sum = xi.sum(axis=0)
A_new = xi_sum / xi_sum.sum(axis=1, keepdims=True)

Nk = gamma.sum(axis=0)
means_new = (gamma.T @ data) / Nk[:, None]
covs_new = []
for k in range(K):
    d = data - means_new[k]
    weighted = (gamma[:, k : k + 1] * d).T @ d
    covs_new.append(weighted / Nk[k])
covs_new = np.array(covs_new)

cases = [
    {
        "fn": "hmmEStep",
        "data": data.tolist(),
        "params": {
            "pi": pi.tolist(),
            "A": A.tolist(),
            "components": [
                {"mean": means[k].tolist(), "cov": covs[k].tolist()} for k in range(K)
            ],
        },
        "expected": {
            "gamma": gamma.tolist(),
            "xi": xi.tolist(),
            "logLikelihood": log_likelihood,
        },
    },
    {
        "fn": "hmmMStep",
        "data": data.tolist(),
        "gamma": gamma.tolist(),
        "xi": xi.tolist(),
        "expected": {
            "pi": pi_new.tolist(),
            "A": A_new.tolist(),
            "components": [
                {"mean": means_new[k].tolist(), "cov": covs_new[k].tolist()}
                for k in range(K)
            ],
        },
    },
]

(FIXTURES / "hmm_em.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'hmm_em.json'}")
