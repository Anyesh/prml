"""Golden fixtures for packages/math/src/mixtures/em.ts.

Covers the general-EM decomposition ln p(X|theta) = L(q,theta) + KL(q||p) (PRML 9.70-9.71)
specialised to a Gaussian mixture, and a full hand-rolled EM iterate trace (E-step then
M-step, repeated) from a fixed initial mixture with no RNG involved, so
packages/math/src/mixtures/em.ts's gmmEmStep/gmmFitEM can be checked iterate-by-iterate.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def component_dicts(weights, means, covs):
    k = len(weights)
    return [
        {
            "weight": float(weights[j]),
            "mean": means[j].tolist(),
            "cov": covs[j].tolist(),
        }
        for j in range(k)
    ]


def log_component_pdfs(x, means, covs):
    return np.array(
        [multivariate_normal.logpdf(x, means[j], covs[j]) for j in range(len(means))]
    )


def mixture_log_pdf(x, weights, means, covs):
    lc = log_component_pdfs(x, means, covs)
    return float(logsumexp(np.log(weights) + lc))


def log_likelihood(data, weights, means, covs):
    return float(sum(mixture_log_pdf(x, weights, means, covs) for x in data))


def e_step(data, weights, means, covs):
    k = len(weights)
    resp = np.zeros((len(data), k))
    for n, x in enumerate(data):
        lc = log_component_pdfs(x, means, covs)
        log_num = np.log(weights) + lc
        resp[n] = np.exp(log_num - logsumexp(log_num))
    return resp


def m_step(data, resp):
    Nk = resp.sum(axis=0)
    means = (resp.T @ data) / Nk[:, None]
    covs = []
    for k in range(resp.shape[1]):
        d = data - means[k]
        weighted = (resp[:, k : k + 1] * d).T @ d
        covs.append(weighted / Nk[k])
    weights = Nk / len(data)
    return weights, means, np.array(covs)


def expected_complete_data_ll(data, resp, weights, means, covs):
    total = 0.0
    for n, x in enumerate(data):
        lc = log_component_pdfs(x, means, covs)
        for k in range(len(weights)):
            if resp[n, k] > 0:
                total += resp[n, k] * (np.log(weights[k]) + lc[k])
    return float(total)


def responsibility_entropy(resp):
    r = resp[resp > 0]
    return float(-(r * np.log(r)).sum())


rng = np.random.default_rng(7)
true_means = np.array([[0.0, 0.0], [4.0, 4.0]])
true_covs = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
data = np.concatenate(
    [
        rng.multivariate_normal(true_means[0], true_covs[0], size=6),
        rng.multivariate_normal(true_means[1], true_covs[1], size=6),
    ]
)
data_list = data.tolist()

cases = []

init_weights = np.array([0.5, 0.5])
init_means = np.array([[1.0, 0.0], [3.0, 4.0]])
init_covs = np.array([[[2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 2.0]]])

resp_exact = e_step(data, init_weights, init_means, init_covs)
ll = log_likelihood(data, init_weights, init_means, init_covs)
ecll = expected_complete_data_ll(data, resp_exact, init_weights, init_means, init_covs)
ent = responsibility_entropy(resp_exact)
lower_bound_exact = ecll + ent

cases.append(
    {
        "fn": "expectedCompleteDataLogLikelihood",
        "data": data_list,
        "responsibilities": resp_exact.tolist(),
        "params": {"components": component_dicts(init_weights, init_means, init_covs)},
        "expected": ecll,
    }
)
cases.append(
    {
        "fn": "responsibilityEntropy",
        "responsibilities": resp_exact.tolist(),
        "expected": ent,
    }
)
cases.append(
    {
        "fn": "emLowerBound_atExactPosterior",
        "data": data_list,
        "responsibilities": resp_exact.tolist(),
        "params": {"components": component_dicts(init_weights, init_means, init_covs)},
        "expected": lower_bound_exact,
    }
)
cases.append(
    {
        "fn": "emKlGap_atExactPosterior",
        "data": data_list,
        "responsibilities": resp_exact.tolist(),
        "params": {"components": component_dicts(init_weights, init_means, init_covs)},
        "expectedLogLikelihood": ll,
        "expectedGap": float(ll - lower_bound_exact),
    }
)

resp_perturbed = 0.7 * resp_exact + 0.3 * np.full_like(resp_exact, 0.5)
resp_perturbed = resp_perturbed / resp_perturbed.sum(axis=1, keepdims=True)
ecll_p = expected_complete_data_ll(
    data, resp_perturbed, init_weights, init_means, init_covs
)
ent_p = responsibility_entropy(resp_perturbed)
lower_bound_p = ecll_p + ent_p
cases.append(
    {
        "fn": "emKlGap_awayFromPosterior",
        "data": data_list,
        "responsibilities": resp_perturbed.tolist(),
        "params": {"components": component_dicts(init_weights, init_means, init_covs)},
        "expectedLogLikelihood": ll,
        "expectedGap": float(ll - lower_bound_p),
    }
)

weights, means, covs = init_weights.copy(), init_means.copy(), init_covs.copy()
trace = []
for _ in range(4):
    resp = e_step(data, weights, means, covs)
    ll_before = log_likelihood(data, weights, means, covs)
    weights, means, covs = m_step(data, resp)
    trace.append(
        {
            "responsibilities": resp.tolist(),
            "logLikelihoodBeforeMStep": ll_before,
            "paramsAfterMStep": {"components": component_dicts(weights, means, covs)},
        }
    )

cases.append(
    {
        "fn": "gmmEmTrace",
        "data": data_list,
        "initialParams": {
            "components": component_dicts(init_weights, init_means, init_covs)
        },
        "steps": 4,
        "expectedTrace": trace,
    }
)

(FIXTURES / "em.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'em.json'}")
