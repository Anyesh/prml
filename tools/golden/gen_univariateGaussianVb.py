"""Golden fixtures for packages/math/src/variational/univariateGaussianVb.ts.

Independently unrolls the PRML 10.1.3 coordinate-ascent recursion for the conjugate
univariate Gaussian in plain numpy: q(mu) from the current E[tau], then q(tau) from the
freshly updated q(mu), matching the order the book (and the TypeScript) iterates in.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def step(data, prior, q_mu, q_tau):
    n = len(data)
    xbar = float(np.mean(data))
    mu0, lambda0, a0, b0 = prior
    expected_tau = q_tau[0] / q_tau[1]

    mu_n = (lambda0 * mu0 + n * xbar) / (lambda0 + n)
    lambda_n = (lambda0 + n) * expected_tau

    a_n = a0 + n / 2
    var_mu = 1.0 / lambda_n
    sum_sq = sum((x - mu_n) ** 2 + var_mu for x in data)
    prior_term = lambda0 * ((mu_n - mu0) ** 2 + var_mu)
    b_n = b0 + 0.5 * (sum_sq + prior_term)

    return (mu_n, lambda_n), (a_n, b_n)


cases = []

rng = np.random.default_rng(20260917)
data = (rng.normal(loc=2.5, scale=1.3, size=12)).tolist()
prior = (
    0.0,
    1e-3,
    1e-3,
    1e-3,
)  # near-noninformative, PRML's own choice for this example
initial_q_mu = (0.0, 1.0)
initial_q_tau = (1.0, 1.0)

trajectory = [
    {
        "qMu": {"mu": initial_q_mu[0], "lambda": initial_q_mu[1]},
        "qTau": {"a": initial_q_tau[0], "b": initial_q_tau[1]},
    }
]
q_mu, q_tau = initial_q_mu, initial_q_tau
for _ in range(8):
    q_mu, q_tau = step(data, prior, q_mu, q_tau)
    trajectory.append(
        {
            "qMu": {"mu": q_mu[0], "lambda": q_mu[1]},
            "qTau": {"a": q_tau[0], "b": q_tau[1]},
        }
    )

cases.append(
    {
        "fn": "univariateGaussianVbFit",
        "data": data,
        "prior": {"mu0": prior[0], "lambda0": prior[1], "a0": prior[2], "b0": prior[3]},
        "initial": trajectory[0],
        "iterations": 8,
        "expected": trajectory,
    }
)

# A second run under the exactly noninformative prior mu0=lambda0=a0=b0=0 (PRML's own
# choice for 10.31-10.33): mu_N is then xbar on the very first sweep regardless of E[tau],
# and iterating drives E[tau] to a genuine fixed point solvable in closed form. Substituting
# the fixed-point condition E[tau] = a_N / b_N into the b_N recursion and solving gives
# E[tau] = (N - 1) / sum((x - xbar)^2), the *unbiased* sample variance's reciprocal (10.33),
# not the biased one: this is checked directly below rather than assumed.
noninformative = (0.0, 0.0, 0.0, 0.0)
data2 = (rng.normal(loc=-1.0, scale=0.7, size=40)).tolist()
q_mu2, q_tau2 = initial_q_mu, initial_q_tau
for _ in range(200):
    q_mu2, q_tau2 = step(data2, noninformative, q_mu2, q_tau2)

sample_mean = float(np.mean(data2))
xbar = sample_mean
sum_sq_dev = float(np.sum((np.array(data2) - xbar) ** 2))
n2 = len(data2)
expected_tau_closed_form = (n2 - 1) / sum_sq_dev

converged_expected_tau = q_tau2[0] / q_tau2[1]
assert abs(converged_expected_tau - expected_tau_closed_form) < 1e-9, (
    converged_expected_tau,
    expected_tau_closed_form,
)
assert abs(q_mu2[0] - sample_mean) < 1e-9

cases.append(
    {
        "fn": "univariateGaussianVbConvergence",
        "data": data2,
        "prior": {
            "mu0": noninformative[0],
            "lambda0": noninformative[1],
            "a0": noninformative[2],
            "b0": noninformative[3],
        },
        "initial": {
            "qMu": {"mu": initial_q_mu[0], "lambda": initial_q_mu[1]},
            "qTau": {"a": initial_q_tau[0], "b": initial_q_tau[1]},
        },
        "iterations": 200,
        "expectedMu": sample_mean,
        "expectedExpectedTau": expected_tau_closed_form,
    }
)

(FIXTURES / "univariateGaussianVb.json").write_text(
    json.dumps({"cases": cases}, indent=2)
)
print(f"wrote {len(cases)} cases to {FIXTURES / 'univariateGaussianVb.json'}")
