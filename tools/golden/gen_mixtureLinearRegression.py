"""Golden fixtures for packages/math/src/ensemble/mixtureLinearRegression.ts.

PRML 14.5.1: a mixture of K linear-regression models sharing one noise precision beta,
p(t|phi) = sum_k pi_k N(t | w_k^T phi, beta^-1) (14.34). This is the conditional analogue
of the unconditional Gaussian mixture in mixtures/gmm.ts + em.ts.

Two-line synthetic set (mirrors book figure 14.8's toy example): x uniform on [-1, 1], each
point independently assigned to one of two lines with its own noise.

Beta's M-step convention: PRML 14.44 is evaluated against the FRESHLY updated w_k from the
same M-step, not against the w_k that produced the responsibilities, because the M step
maximises Q(theta, theta_old) over the whole new theta at once, so both `mStep` here and the
TypeScript compute weights first and beta second, from those new weights.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260929)

N = 40
X = rng.uniform(-1.0, 1.0, N)
LINE = (rng.uniform(0.0, 1.0, N) < 0.5).astype(int)
NOISE_SD = 0.15
TRUE_W = np.array([[0.5, 0.6], [-0.4, -0.7]])  # row k: [intercept, slope]
DESIGN = np.stack([np.ones(N), X], axis=1)
MEANS = np.where(LINE == 0, DESIGN @ TRUE_W[0], DESIGN @ TRUE_W[1])
T = MEANS + NOISE_SD * rng.normal(size=N)


def params_dict(weights, mixing, beta):
    return {"weights": weights.tolist(), "mixing": mixing.tolist(), "beta": float(beta)}


def responsibilities(design, t, weights, mixing, beta):
    k = len(mixing)
    n = len(t)
    log_terms = np.zeros((n, k))
    for c in range(k):
        mean = design @ weights[c]
        log_terms[:, c] = np.log(mixing[c]) + norm.logpdf(
            t, loc=mean, scale=1.0 / np.sqrt(beta)
        )
    m = log_terms.max(axis=1, keepdims=True)
    unnorm = np.exp(log_terms - m)
    return unnorm / unnorm.sum(axis=1, keepdims=True)


def log_likelihood(design, t, weights, mixing, beta):
    k = len(mixing)
    n = len(t)
    log_terms = np.zeros((n, k))
    for c in range(k):
        mean = design @ weights[c]
        log_terms[:, c] = np.log(mixing[c]) + norm.logpdf(
            t, loc=mean, scale=1.0 / np.sqrt(beta)
        )
    m = log_terms.max(axis=1, keepdims=True)
    return float(np.sum(m[:, 0] + np.log(np.sum(np.exp(log_terms - m), axis=1))))


def m_step(design, t, resp):
    n, k = resp.shape
    mixing = resp.mean(axis=0)
    weights = np.zeros((k, design.shape[1]))
    for c in range(k):
        w = resp[:, c]
        wmat = np.diag(w)
        gram = design.T @ wmat @ design
        rhs = design.T @ wmat @ t
        weights[c] = np.linalg.solve(gram, rhs)
    sum_sq = 0.0
    for c in range(k):
        residual = t - design @ weights[c]
        sum_sq += np.sum(resp[:, c] * residual**2)
    beta = n / sum_sq
    return weights, mixing, beta


INITIAL_WEIGHTS = np.array([[0.0, 1.0], [0.0, -1.0]])
INITIAL_MIXING = np.array([0.5, 0.5])
INITIAL_BETA = 10.0

cases = []

resp0 = responsibilities(DESIGN, T, INITIAL_WEIGHTS, INITIAL_MIXING, INITIAL_BETA)
cases.append(
    {
        "fn": "responsibilities",
        "design": DESIGN.tolist(),
        "targets": T.tolist(),
        "params": params_dict(INITIAL_WEIGHTS, INITIAL_MIXING, INITIAL_BETA),
        "expected": resp0.tolist(),
    }
)

ll0 = log_likelihood(DESIGN, T, INITIAL_WEIGHTS, INITIAL_MIXING, INITIAL_BETA)
cases.append(
    {
        "fn": "logLikelihood",
        "design": DESIGN.tolist(),
        "targets": T.tolist(),
        "params": params_dict(INITIAL_WEIGHTS, INITIAL_MIXING, INITIAL_BETA),
        "expected": ll0,
    }
)

w1, mix1, beta1 = m_step(DESIGN, T, resp0)
cases.append(
    {
        "fn": "mStep",
        "design": DESIGN.tolist(),
        "targets": T.tolist(),
        "responsibilities": resp0.tolist(),
        "expected": params_dict(w1, mix1, beta1),
    }
)

STEPS = 4
trace = []
weights, mixing, beta = INITIAL_WEIGHTS, INITIAL_MIXING, INITIAL_BETA
for _ in range(STEPS):
    resp = responsibilities(DESIGN, T, weights, mixing, beta)
    ll = log_likelihood(DESIGN, T, weights, mixing, beta)
    weights, mixing, beta = m_step(DESIGN, T, resp)
    trace.append(
        {
            "responsibilities": resp.tolist(),
            "logLikelihoodBeforeMStep": ll,
            "paramsAfterMStep": params_dict(weights, mixing, beta),
        }
    )

cases.append(
    {
        "fn": "fitEM",
        "design": DESIGN.tolist(),
        "targets": T.tolist(),
        "initialParams": params_dict(INITIAL_WEIGHTS, INITIAL_MIXING, INITIAL_BETA),
        "steps": STEPS,
        "expectedTrace": trace,
    }
)

fixture = {"cases": cases}
(FIXTURES / "mixtureLinearRegression.json").write_text(json.dumps(fixture, indent=2))
print(f"wrote {len(cases)} cases to mixtureLinearRegression.json")
