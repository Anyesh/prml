"""Golden fixtures for packages/math/src/sequential/forwardBackward.ts and hmm.ts.

Plain numpy for the alpha/beta/gamma/xi recursions (PRML 13.36, 13.38, 13.33, 13.43,
13.59-13.65): computed directly from the definitions, not by re-deriving the
TypeScript algorithm. scipy.stats.multivariate_normal supplies the Gaussian emission
matrix B[n, k] = p(x_n | z_n = k).
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
N = 6
data = np.concatenate(
    [
        rng.multivariate_normal(means[0], covs[0], size=3),
        rng.multivariate_normal(means[1], covs[1], size=3),
    ]
)


def emission_matrix(x):
    return np.array([multivariate_normal.pdf(x, means[k], covs[k]) for k in range(K)]).T


B = emission_matrix(data)


def forward_unscaled(B):
    alpha = np.zeros((N, K))
    alpha[0] = pi * B[0]
    for n in range(1, N):
        alpha[n] = B[n] * (alpha[n - 1] @ A)
    return alpha


def backward_unscaled(B):
    beta = np.zeros((N, K))
    beta[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = (A * (B[n + 1] * beta[n + 1])[None, :]).sum(axis=1)
    return beta


alpha = forward_unscaled(B)
beta = backward_unscaled(B)
pX = float(alpha[N - 1].sum())
gamma = alpha * beta / pX


def xi_unscaled():
    out = []
    for n in range(1, N):
        mat = (
            np.outer(alpha[n - 1], np.ones(K))
            * A
            * B[n][None, :]
            * beta[n][None, :]
            / pX
        )
        out.append(mat.tolist())
    return out


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


alpha_hat, c = forward_scaled(B)


def backward_scaled(B, c):
    beta_hat = np.zeros((N, K))
    beta_hat[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        raw = (A * (B[n + 1] * beta_hat[n + 1])[None, :]).sum(axis=1)
        beta_hat[n] = raw / c[n + 1]
    return beta_hat


beta_hat = backward_scaled(B, c)
gamma_scaled = alpha_hat * beta_hat


def xi_scaled():
    # PRML's printed (13.65) reads "= c_n alpha_hat(n-1) ...", but substituting the
    # hatted definitions (13.58, 13.60) into the unscaled xi (13.43) gives a division by
    # c_n, not a multiplication: verified numerically against xi_unscaled below, and the
    # printed "c_n" is the OCR losing a fraction bar over what should read "1 / c_n".
    out = []
    for n in range(1, N):
        mat = (
            np.outer(alpha_hat[n - 1], np.ones(K))
            * A
            * B[n][None, :]
            * beta_hat[n][None, :]
            / c[n]
        )
        out.append(mat.tolist())
    return out


log_likelihood_scaled = float(np.log(c).sum())

# Self-check: the scaled and unscaled xi must agree exactly on this short (no-underflow)
# chain. This is the check that caught the sign/placement error above; keep it.
assert np.allclose(np.array(xi_scaled()), np.array(xi_unscaled()), atol=1e-9)

cases = [
    {
        "fn": "hmmForwardUnscaled",
        "pi": pi.tolist(),
        "A": A.tolist(),
        "B": B.tolist(),
        "expected": alpha.tolist(),
    },
    {
        "fn": "hmmBackwardUnscaled",
        "A": A.tolist(),
        "B": B.tolist(),
        "expected": beta.tolist(),
    },
    {
        "fn": "hmmGammaUnscaled",
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "expected": gamma.tolist(),
    },
    {
        "fn": "hmmLikelihoodUnscaled",
        "alpha": alpha.tolist(),
        "expected": pX,
    },
    {
        "fn": "hmmXiUnscaled",
        "alpha": alpha.tolist(),
        "A": A.tolist(),
        "B": B.tolist(),
        "beta": beta.tolist(),
        "pX": pX,
        "expected": xi_unscaled(),
    },
    {
        "fn": "hmmForwardScaled",
        "pi": pi.tolist(),
        "A": A.tolist(),
        "B": B.tolist(),
        "expected": {"alphaHat": alpha_hat.tolist(), "c": c.tolist()},
    },
    {
        "fn": "hmmBackwardScaled",
        "A": A.tolist(),
        "B": B.tolist(),
        "c": c.tolist(),
        "expected": beta_hat.tolist(),
    },
    {
        "fn": "hmmGammaScaled",
        "alphaHat": alpha_hat.tolist(),
        "betaHat": beta_hat.tolist(),
        "expected": gamma_scaled.tolist(),
    },
    {
        "fn": "hmmXiScaled",
        "alphaHat": alpha_hat.tolist(),
        "A": A.tolist(),
        "B": B.tolist(),
        "betaHat": beta_hat.tolist(),
        "c": c.tolist(),
        "expected": xi_scaled(),
    },
    {
        "fn": "hmmLogLikelihoodScaled",
        "c": c.tolist(),
        "expected": log_likelihood_scaled,
    },
    {
        "fn": "hmmGaussianEmissionMatrix",
        "data": data.tolist(),
        "components": [
            {"mean": means[k].tolist(), "cov": covs[k].tolist()} for k in range(K)
        ],
        "expected": B.tolist(),
    },
]

(FIXTURES / "hmm_forward_backward.json").write_text(
    json.dumps({"cases": cases}, indent=2)
)
print(f"wrote {len(cases)} cases to {FIXTURES / 'hmm_forward_backward.json'}")
