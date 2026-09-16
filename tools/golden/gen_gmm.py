"""Golden fixtures for packages/math/src/mixtures/gmm.ts.

scipy.stats.multivariate_normal is the reference density (as in tools/golden/gen_mvn.py);
mixture quantities on top of it (log density, responsibilities, log-likelihood, the M-step
update) are plain numpy, computed the same way a from-scratch EM implementation would,
never by re-deriving the TypeScript algorithm.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

weights = np.array([0.6, 0.4])
means = np.array([[0.0, 0.0], [3.0, 3.0]])
covs = np.array([[[1.0, 0.3], [0.3, 1.0]], [[1.5, -0.2], [-0.2, 0.5]]])


def component_dicts():
    return [
        {
            "weight": float(weights[k]),
            "mean": means[k].tolist(),
            "cov": covs[k].tolist(),
        }
        for k in range(2)
    ]


def log_component_pdfs(x):
    return np.array(
        [multivariate_normal.logpdf(x, means[k], covs[k]) for k in range(2)]
    )


def mixture_log_pdf(x):
    lc = log_component_pdfs(x)
    return float(logsumexp(np.log(weights) + lc))


def responsibilities(x):
    lc = log_component_pdfs(x)
    log_num = np.log(weights) + lc
    return np.exp(log_num - logsumexp(log_num)).tolist()


cases = []

points = [[0.0, 0.0], [3.0, 3.0], [1.5, 1.5], [-2.0, 4.0]]
for x in points:
    xnp = np.array(x)
    cases.append(
        {
            "fn": "gmmLogPdf",
            "x": x,
            "params": {"components": component_dicts()},
            "expected": mixture_log_pdf(xnp),
        }
    )
    cases.append(
        {
            "fn": "gmmResponsibilities",
            "x": x,
            "params": {"components": component_dicts()},
            "expected": responsibilities(xnp),
        }
    )

# A dataset for the batch E-step, log-likelihood, gmmInit, and gmmMStep cases.
rng = np.random.default_rng(20260916)
n_per = 5
data = np.concatenate(
    [
        rng.multivariate_normal(means[0], covs[0], size=n_per),
        rng.multivariate_normal(means[1], covs[1], size=n_per),
    ]
)
data_list = data.tolist()

resp_batch = np.array([responsibilities(x) for x in data])
cases.append(
    {
        "fn": "gmmEStep",
        "data": data_list,
        "params": {"components": component_dicts()},
        "expected": resp_batch.tolist(),
    }
)

log_likelihood = float(sum(mixture_log_pdf(x) for x in data))
cases.append(
    {
        "fn": "gmmLogLikelihood",
        "data": data_list,
        "params": {"components": component_dicts()},
        "expected": log_likelihood,
    }
)

# gmmInit: build an initial mixture from a hard clustering (as K-means would hand EM).
assignments = np.array([0] * n_per + [1] * n_per)
init_means = np.array([data[assignments == k].mean(axis=0) for k in range(2)])


def empirical_cov(pts):
    d = pts - pts.mean(axis=0)
    return (d.T @ d) / len(pts)


init_covs = np.array([empirical_cov(data[assignments == k]) for k in range(2)])
init_weights = np.array([np.sum(assignments == k) / len(data) for k in range(2)])

cases.append(
    {
        "fn": "gmmInit",
        "data": data_list,
        "means": init_means.tolist(),
        "assignments": assignments.tolist(),
        "k": 2,
        "expected": {
            "components": [
                {
                    "weight": float(init_weights[k]),
                    "mean": init_means[k].tolist(),
                    "cov": init_covs[k].tolist(),
                }
                for k in range(2)
            ]
        },
    }
)

# gmmMStep: given fixed responsibilities (not necessarily the exact posterior), the
# closed-form weighted mean/covariance/mixing-coefficient update (PRML 9.17-9.22).
Nk = resp_batch.sum(axis=0)
m_means = (resp_batch.T @ data) / Nk[:, None]
m_covs = []
for k in range(2):
    d = data - m_means[k]
    weighted = (resp_batch[:, k : k + 1] * d).T @ d
    m_covs.append(weighted / Nk[k])
m_weights = Nk / len(data)

cases.append(
    {
        "fn": "gmmMStep",
        "data": data_list,
        "responsibilities": resp_batch.tolist(),
        "expected": {
            "components": [
                {
                    "weight": float(m_weights[k]),
                    "mean": m_means[k].tolist(),
                    "cov": m_covs[k].tolist(),
                }
                for k in range(2)
            ]
        },
    }
)

(FIXTURES / "gmm.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'gmm.json'}")
