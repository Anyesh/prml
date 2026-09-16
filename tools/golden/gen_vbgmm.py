"""Golden fixtures for packages/math/src/variational/vbgmm.ts.

Every quantity here (responsibilities, the Dirichlet/Gaussian-Wishart M-step, the lower
bound, the Student-t predictive) is computed from scratch in numpy/scipy against the PRML
10.2 equations directly, never by re-deriving the TypeScript's own control flow.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import digamma, gammaln, multigammaln
from scipy.stats import multivariate_t

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

D = 2
K = 2


def wishart_log_normaliser(scale, dof):
    d = scale.shape[0]
    sign, logdet = np.linalg.slogdet(scale)
    return -(dof / 2) * logdet - (dof * d / 2) * np.log(2) - multigammaln(dof / 2, d)


def dirichlet_log_normaliser(alpha):
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))


def expected_log_pi(alpha):
    return digamma(alpha) - digamma(np.sum(alpha))


def expected_log_det(scale, dof):
    d = scale.shape[0]
    sign, logdet = np.linalg.slogdet(scale)
    return (
        sum(digamma((dof + 1 - i) / 2) for i in range(1, d + 1))
        + d * np.log(2)
        + logdet
    )


def component_dict(beta, mean, scale, dof):
    return {
        "beta": float(beta),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "dof": float(dof),
    }


def posterior_dict(alpha, components):
    return {"alpha": alpha.tolist(), "components": components}


# A hand-built posterior (not fit from data) so responsibilities and the lower bound can be
# checked against a fixed, known state rather than only end-to-end after an M-step.
alpha_post = np.array([3.0, 5.0])
components = [
    {
        "beta": 2.0,
        "mean": np.array([0.0, 0.0]),
        "scale": np.array([[1.0, 0.1], [0.1, 1.2]]),
        "dof": 4.0,
    },
    {
        "beta": 3.0,
        "mean": np.array([3.0, 2.0]),
        "scale": np.array([[0.8, -0.05], [-0.05, 0.9]]),
        "dof": 5.0,
    },
]

cases = []

# vbGmmExpectedLogPi / vbGmmExpectedLogDet
cases.append(
    {
        "fn": "vbGmmExpectedLogPi",
        "posterior": posterior_dict(
            alpha_post, [component_dict(**c) for c in components]
        ),
        "expected": expected_log_pi(alpha_post).tolist(),
    }
)
cases.append(
    {
        "fn": "vbGmmExpectedLogDet",
        "posterior": posterior_dict(
            alpha_post, [component_dict(**c) for c in components]
        ),
        "expected": [expected_log_det(c["scale"], c["dof"]) for c in components],
    }
)

# vbGmmResponsibilities at a handful of points.
points = [[0.0, 0.0], [3.0, 2.0], [1.5, 1.0], [-1.0, 4.0]]


def responsibilities_at(x, alpha, comps):
    log_rho = []
    lnpi = expected_log_pi(alpha)
    for k, c in enumerate(comps):
        ln_lambda = expected_log_det(c["scale"], c["dof"])
        diff = np.array(x) - c["mean"]
        quad = D / c["beta"] + c["dof"] * (diff @ c["scale"] @ diff)
        log_rho.append(
            lnpi[k] + 0.5 * ln_lambda - (D / 2) * np.log(2 * np.pi) - 0.5 * quad
        )
    log_rho = np.array(log_rho)
    m = log_rho.max()
    w = np.exp(log_rho - m)
    return (w / w.sum()).tolist()


for x in points:
    cases.append(
        {
            "fn": "vbGmmResponsibilities",
            "data": [x],
            "posterior": posterior_dict(
                alpha_post, [component_dict(**c) for c in components]
            ),
            "expected": [responsibilities_at(x, alpha_post, components)],
        }
    )

# A small synthetic dataset drives the M-step and lower-bound cases.
rng = np.random.default_rng(20260917)
data = np.concatenate(
    [
        rng.multivariate_normal([0.0, 0.0], [[1.0, 0.2], [0.2, 1.0]], size=6),
        rng.multivariate_normal([4.0, 3.0], [[0.7, -0.1], [-0.1, 0.6]], size=6),
    ]
)
r = np.array([responsibilities_at(x, alpha_post, components) for x in data])

alpha0, beta0, dof0 = 1e-3, 1.0, 3.0
mean0 = np.zeros(D)
scale0 = np.eye(D)

nk = r.sum(axis=0)
xbar = (r.T @ data) / nk[:, None]
sk = []
for k in range(K):
    diff = data - xbar[k]
    weighted = (r[:, k : k + 1] * diff).T @ diff / nk[k]
    sk.append(weighted)
sk = np.array(sk)

alpha_new = alpha0 + nk
beta_new = beta0 + nk
mean_new = (beta0 * mean0 + nk[:, None] * xbar) / beta_new[:, None]
scale_new = []
dof_new = dof0 + nk
scale0_inv = np.linalg.inv(scale0)
for k in range(K):
    diff = xbar[k] - mean0
    shrink = (beta0 * nk[k]) / (beta0 + nk[k])
    scale_inv_k = scale0_inv + nk[k] * sk[k] + shrink * np.outer(diff, diff)
    inv = np.linalg.inv(scale_inv_k)
    scale_new.append(
        (inv + inv.T) / 2
    )  # exact symmetry: matrix inversion leaves ~1e-16 asymmetry a strict Cholesky check rejects
scale_new = np.array(scale_new)

cases.append(
    {
        "fn": "vbGmmMStep",
        "data": data.tolist(),
        "r": r.tolist(),
        "prior": {
            "alpha0": alpha0,
            "beta0": beta0,
            "mean0": mean0.tolist(),
            "scale0": scale0.tolist(),
            "dof0": dof0,
        },
        "expected": posterior_dict(
            alpha_new,
            [
                component_dict(beta_new[k], mean_new[k], scale_new[k], dof_new[k])
                for k in range(K)
            ],
        ),
    }
)

# vbGmmLowerBound: assembled term by term, matching 10.70-10.77 exactly, against the
# posterior produced by the M-step above (so the E-step's r and the M-step's posterior are
# mutually consistent, as they would be inside a real fit).
posterior_new = {
    "alpha": alpha_new,
    "components": [
        {
            "beta": beta_new[k],
            "mean": mean_new[k],
            "scale": scale_new[k],
            "dof": dof_new[k],
        }
        for k in range(K)
    ],
}


def lower_bound(data, r, posterior, prior):
    nk = r.sum(axis=0)
    xbar = (r.T @ data) / np.where(nk > 0, nk, 1)[:, None]
    sk = []
    for k in range(K):
        diff = data - xbar[k]
        w = (r[:, k : k + 1] * diff).T @ diff
        sk.append(w / nk[k] if nk[k] > 0 else np.zeros((D, D)))
    sk = np.array(sk)

    ln_pi = expected_log_pi(posterior["alpha"])
    ln_lambda = np.array(
        [expected_log_det(c["scale"], c["dof"]) for c in posterior["components"]]
    )

    e_ln_p_x = 0.0
    for k, c in enumerate(posterior["components"]):
        if nk[k] == 0:
            continue
        diff = xbar[k] - c["mean"]
        mahalanobis = diff @ c["scale"] @ diff
        e_ln_p_x += nk[k] * (
            ln_lambda[k]
            - D / c["beta"]
            - c["dof"] * np.trace(sk[k] @ c["scale"])
            - c["dof"] * mahalanobis
            - D * np.log(2 * np.pi)
        )
    e_ln_p_x *= 0.5

    e_ln_p_z = float(np.sum(r * ln_pi[None, :]))

    alpha0_vec = np.full(K, prior["alpha0"])
    e_ln_p_pi = dirichlet_log_normaliser(alpha0_vec) + (prior["alpha0"] - 1) * np.sum(
        ln_pi
    )

    scale0_inv = np.linalg.inv(prior["scale0"])
    e_ln_p_mu_lambda = 0.0
    for k, c in enumerate(posterior["components"]):
        diff = np.array(c["mean"]) - np.array(prior["mean0"])
        mahalanobis = diff @ c["scale"] @ diff
        e_ln_p_mu_lambda += 0.5 * (
            D * np.log(prior["beta0"] / (2 * np.pi))
            + ln_lambda[k]
            - D * prior["beta0"] / c["beta"]
            - prior["beta0"] * c["dof"] * mahalanobis
        )
        e_ln_p_mu_lambda += ((prior["dof0"] - D - 1) / 2) * ln_lambda[k]
        e_ln_p_mu_lambda -= 0.5 * c["dof"] * np.trace(scale0_inv @ c["scale"])
    e_ln_p_mu_lambda += K * wishart_log_normaliser(
        np.array(prior["scale0"]), prior["dof0"]
    )

    e_ln_q_z = float(np.sum(np.where(r > 0, r * np.log(np.where(r > 0, r, 1)), 0.0)))

    e_ln_q_pi = float(
        np.sum((posterior["alpha"] - 1) * ln_pi)
    ) + dirichlet_log_normaliser(posterior["alpha"])

    e_ln_q_mu_lambda = 0.0
    for k, c in enumerate(posterior["components"]):
        entropy_wishart = (
            -wishart_log_normaliser(c["scale"], c["dof"])
            - ((c["dof"] - D - 1) / 2) * ln_lambda[k]
            + (c["dof"] * D) / 2
        )
        e_ln_q_mu_lambda += (
            0.5 * ln_lambda[k]
            + (D / 2) * np.log(c["beta"] / (2 * np.pi))
            - D / 2
            - entropy_wishart
        )

    return (
        e_ln_p_x
        + e_ln_p_z
        + e_ln_p_pi
        + e_ln_p_mu_lambda
        - e_ln_q_z
        - e_ln_q_pi
        - e_ln_q_mu_lambda
    )


prior_dict = {
    "alpha0": alpha0,
    "beta0": beta0,
    "mean0": mean0.tolist(),
    "scale0": scale0.tolist(),
    "dof0": dof0,
}
lb = lower_bound(data, r, posterior_new, prior_dict)

cases.append(
    {
        "fn": "vbGmmLowerBound",
        "data": data.tolist(),
        "r": r.tolist(),
        "posterior": posterior_dict(
            alpha_new,
            [
                component_dict(beta_new[k], mean_new[k], scale_new[k], dof_new[k])
                for k in range(K)
            ],
        ),
        "prior": prior_dict,
        "expected": float(lb),
    }
)

# vbGmmPredictiveLogPdf, cross-checked against scipy.stats.multivariate_t independently.
alpha_hat = float(np.sum(alpha_new))
predictive_points = [[0.0, 0.0], [4.0, 3.0], [2.0, 1.5]]
predictive_expected = []
for x in predictive_points:
    total = 0.0
    for k in range(K):
        nu = dof_new[k] + 1 - D
        precision_scale = (dof_new[k] + 1 - D) * beta_new[k] / (1 + beta_new[k])
        lk = scale_new[k] * precision_scale
        scale_matrix = np.linalg.inv(lk)
        pdf = multivariate_t(loc=mean_new[k], shape=scale_matrix, df=nu).pdf(x)
        total += (alpha_new[k] / alpha_hat) * pdf
    predictive_expected.append(float(np.log(total)))

cases.append(
    {
        "fn": "vbGmmPredictiveLogPdf",
        "points": predictive_points,
        "posterior": posterior_dict(
            alpha_new,
            [
                component_dict(beta_new[k], mean_new[k], scale_new[k], dof_new[k])
                for k in range(K)
            ],
        ),
        "expected": predictive_expected,
    }
)

(FIXTURES / "vbgmm.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'vbgmm.json'}")
