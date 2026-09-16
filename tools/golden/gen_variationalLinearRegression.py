"""Golden fixtures for packages/math/src/variational/linearRegression.ts.

Independently derives the variational treatment of Bayesian linear regression (PRML 10.3):
a Gamma(a0, b0) prior over the weight-precision alpha, inferred jointly with q(w) rather
than fixed at a type-II point estimate. Every closed form is coded fresh here in
numpy/scipy: `psi` and `gammaln` come from scipy.special, never from a call into the
TypeScript module under test.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln, psi

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260917)
M = 3  # dimension of w: constant, linear, quadratic basis
N = 12
xs = np.linspace(-1, 1, N)
Phi = np.stack([np.ones(N), xs, xs**2], axis=1)
true_w = np.array([0.5, -1.2, 0.8])
beta = 9.0
targets = Phi @ true_w + rng.normal(scale=1 / np.sqrt(beta), size=N)

a0, b0 = 1e-3, 1e-3


def q_w(e_alpha, Phi, targets, beta):
    S_inv = e_alpha * np.eye(M) + beta * (Phi.T @ Phi)
    S = np.linalg.inv(S_inv)
    m = beta * S @ (Phi.T @ targets)
    return m, S


def q_alpha_params(a0, b0, m, S):
    aN = a0 + M / 2
    e_wtw = m @ m + np.trace(S)
    bN = b0 + 0.5 * e_wtw
    return aN, bN


def fit(Phi, targets, beta, a0, b0, iters):
    e_alpha = a0 / b0
    history = []
    for _ in range(iters):
        m, S = q_w(e_alpha, Phi, targets, beta)
        aN, bN = q_alpha_params(a0, b0, m, S)
        e_alpha = aN / bN
        history.append({"m": m.copy(), "S": S.copy(), "aN": aN, "bN": bN})
    return history


def gaussian_entropy(S):
    d = S.shape[0]
    sign, logdet = np.linalg.slogdet(S)
    return 0.5 * logdet + 0.5 * d * (1 + np.log(2 * np.pi))


def gamma_entropy(a, b):
    return a - np.log(b) + gammaln(a) + (1 - a) * psi(a)


def lower_bound(Phi, targets, beta, a0, b0, m, S, aN, bN):
    n = len(targets)
    d = m.shape[0]
    e_alpha = aN / bN
    e_ln_alpha = psi(aN) - np.log(bN)
    e_wtw = m @ m + np.trace(S)

    resid = targets - Phi @ m
    e_data_fit = resid @ resid + np.trace(Phi @ S @ Phi.T)
    term_likelihood = 0.5 * n * np.log(beta / (2 * np.pi)) - 0.5 * beta * e_data_fit

    term_prior_w = (
        0.5 * d * e_ln_alpha - 0.5 * d * np.log(2 * np.pi) - 0.5 * e_alpha * e_wtw
    )

    term_prior_alpha = (
        a0 * np.log(b0) - gammaln(a0) + (a0 - 1) * e_ln_alpha - b0 * e_alpha
    )

    term_entropy_w = gaussian_entropy(S)
    term_entropy_alpha = gamma_entropy(aN, bN)

    return float(
        term_likelihood
        + term_prior_w
        + term_prior_alpha
        + term_entropy_w
        + term_entropy_alpha
    )


history = fit(Phi, targets, beta, a0, b0, iters=6)

cases = []

cases.append(
    {
        "fn": "vlrQAlphaParams",
        "a0": a0,
        "b0": b0,
        "mean": history[0]["m"].tolist(),
        "cov": history[0]["S"].tolist(),
        "expected": {"a": history[0]["aN"], "b": history[0]["bN"]},
    }
)

# One full closed-form update step, starting from the prior mean E[alpha] = a0/b0.
e_alpha0 = a0 / b0
m1, S1 = q_w(e_alpha0, Phi, targets, beta)
aN1, bN1 = q_alpha_params(a0, b0, m1, S1)
cases.append(
    {
        "fn": "vlrUpdate",
        "design": Phi.tolist(),
        "targets": targets.tolist(),
        "beta": beta,
        "prior": {"a0": a0, "b0": b0},
        "current": {"a": a0, "b": b0},
        "expected": {"mean": m1.tolist(), "cov": S1.tolist(), "a": aN1, "b": bN1},
    }
)

lb1 = lower_bound(Phi, targets, beta, a0, b0, m1, S1, aN1, bN1)
cases.append(
    {
        "fn": "vlrLowerBound",
        "design": Phi.tolist(),
        "targets": targets.tolist(),
        "beta": beta,
        "prior": {"a0": a0, "b0": b0},
        "posterior": {"mean": m1.tolist(), "cov": S1.tolist(), "a": aN1, "b": bN1},
        "expected": lb1,
    }
)

fit_trace = [
    {"mean": h["m"].tolist(), "cov": h["S"].tolist(), "a": h["aN"], "b": h["bN"]}
    for h in history
]
lb_trace = [
    lower_bound(Phi, targets, beta, a0, b0, h["m"], h["S"], h["aN"], h["bN"])
    for h in history
]
cases.append(
    {
        "fn": "vlrFit",
        "design": Phi.tolist(),
        "targets": targets.tolist(),
        "beta": beta,
        "prior": {"a0": a0, "b0": b0},
        "iters": 6,
        "expectedTrace": fit_trace,
        "expectedLowerBoundTrace": lb_trace,
    }
)

(FIXTURES / "variationalLinearRegression.json").write_text(
    json.dumps({"cases": cases}, indent=2)
)
print(f"wrote {len(cases)} cases to {FIXTURES / 'variationalLinearRegression.json'}")
