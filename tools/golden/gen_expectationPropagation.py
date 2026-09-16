"""Golden fixtures for packages/math/src/variational/expectationPropagation.ts.

Every quantity (the cavity distribution, the clutter problem's moment-matched tilted
posterior, and the refined site) is computed directly from the PRML 10.7.1 equations in
plain numpy, never by re-deriving the TypeScript's own control flow. The multi-point trace
unrolls the same recursion independently, one point at a time.

The trace is capped at a single full sweep over the data. A second sweep, with this
script's own random draw, drives one clutter point's site variance negative (EP's Gaussian
sites are not guaranteed positive; PRML notes this itself in the exercises), which then
makes that site's variance sum with its cavity's negative too, and scipy's
`multivariate_normal` refuses a non-positive-definite covariance outright. That is a real
property of iterated EP, not a bug in this fixture, so the multi-point case is kept to the
one sweep that stays numerically clean and is not evidence about later sweeps.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

D = 2


def isotropic_logpdf(x, mean, variance):
    return multivariate_normal.logpdf(x, mean=mean, cov=variance * np.eye(len(mean)))


def cavity(q_mean, q_var, site_mean, site_var):
    cavity_precision = 1 / q_var - 1 / site_var
    v = 1 / cavity_precision
    m = q_mean + v * (1 / site_var) * (q_mean - site_mean)
    return m, v


def moment_match(x, cav_mean, cav_var, w, a):
    signal_density = np.exp(isotropic_logpdf(x, cav_mean, cav_var + 1))
    clutter_density = np.exp(isotropic_logpdf(x, np.zeros(D), a))
    z = (1 - w) * signal_density + w * clutter_density
    rho = 1 - (w * clutter_density) / z

    diff = x - cav_mean
    shrink = cav_var / (cav_var + 1)
    new_mean = cav_mean + rho * shrink * diff

    sq_dist = float(diff @ diff)
    new_var = (
        cav_var
        - rho * cav_var**2 / (cav_var + 1)
        + rho * (1 - rho) * cav_var**2 * sq_dist / (D * (cav_var + 1) ** 2)
    )
    return z, new_mean, new_var, rho


def refine_site(cav_mean, cav_var, new_mean, new_var, z):
    site_precision = 1 / new_var - 1 / cav_var
    site_var = 1 / site_precision
    site_mean = cav_mean + (site_var + cav_var) * (1 / cav_var) * (new_mean - cav_mean)
    log_normal_at_site = isotropic_logpdf(site_mean, cav_mean, site_var + cav_var)
    log_scale = np.log(z) - (D / 2) * np.log(2 * np.pi * site_var) - log_normal_at_site
    return log_scale, site_mean, site_var


cases = []

# isotropicCavity: an arbitrary q and a finite-variance site.
q_mean, q_var = np.array([0.5, -0.3]), 2.0
site_mean, site_var = np.array([1.2, 0.1]), 5.0
cav_mean, cav_var = cavity(q_mean, q_var, site_mean, site_var)
cases.append(
    {
        "fn": "isotropicCavity",
        "q": {"mean": q_mean.tolist(), "variance": q_var},
        "site": {"logScale": 0.0, "mean": site_mean.tolist(), "variance": site_var},
        "expected": {"mean": cav_mean.tolist(), "variance": float(cav_var)},
    }
)

# clutterMomentMatch, at a point plausibly signal and one plausibly clutter.
model = {"clutterWeight": 0.3, "clutterVariance": 10.0}
for x in ([1.0, -0.5], [8.0, 7.0]):
    xnp = np.array(x)
    z, new_mean, new_var, rho = moment_match(xnp, cav_mean, cav_var, model["clutterWeight"], model["clutterVariance"])
    cases.append(
        {
            "fn": "clutterMomentMatch",
            "x": x,
            "cavity": {"mean": cav_mean.tolist(), "variance": float(cav_var)},
            "model": model,
            "expected": {
                "normaliser": float(z),
                "posterior": {"mean": new_mean.tolist(), "variance": float(new_var)},
                "signalProbability": float(rho),
            },
        }
    )

# refineIsotropicSite, chained from the first moment-match case above.
z0, new_mean0, new_var0, _ = moment_match(np.array([1.0, -0.5]), cav_mean, cav_var, model["clutterWeight"], model["clutterVariance"])
log_scale, refined_mean, refined_var = refine_site(cav_mean, cav_var, new_mean0, new_var0, z0)
cases.append(
    {
        "fn": "refineIsotropicSite",
        "cavity": {"mean": cav_mean.tolist(), "variance": float(cav_var)},
        "moments": {
            "normaliser": float(z0),
            "posterior": {"mean": new_mean0.tolist(), "variance": float(new_var0)},
            "signalProbability": 0.0,
        },
        "expected": {"logScale": float(log_scale), "mean": refined_mean.tolist(), "variance": float(refined_var)},
    }
)

# A full one-sweep clutter EP trace: Minka's own clutter-problem parameters
# (a = 10, prior variance b = 100, w = 0.5), six 2-D points (four signal, two clutter).
w, a, b = 0.5, 10.0, 100.0
rng = np.random.default_rng(20260917)
true_theta = np.array([3.0, -1.0])
signal_pts = rng.multivariate_normal(true_theta, np.eye(D), size=4)
clutter_pts = rng.multivariate_normal(np.zeros(D), a * np.eye(D), size=2)
data = np.concatenate([signal_pts, clutter_pts])

prior_mean, prior_var = np.zeros(D), b
n = len(data)
UNINFORMATIVE_VAR = 1e12
site_means = [np.zeros(D) for _ in range(n)]
site_vars = [UNINFORMATIVE_VAR for _ in range(n)]
site_scales = [0.0 for _ in range(n)]
post_mean, post_var = prior_mean.copy(), prior_var

sweeps = 1
expected_trace = []
for _sweep in range(sweeps):
    for i in range(n):
        cm, cv = cavity(post_mean, post_var, site_means[i], site_vars[i])
        z, nm, nv, _rho = moment_match(data[i], cm, cv, w, a)
        ls, sm, sv = refine_site(cm, cv, nm, nv, z)
        site_means[i], site_vars[i], site_scales[i] = sm, sv, ls
        post_mean, post_var = nm, nv
    expected_trace.append(
        {
            "posterior": {"mean": post_mean.tolist(), "variance": float(post_var)},
            "sites": [
                {"logScale": float(site_scales[k]), "mean": site_means[k].tolist(), "variance": float(site_vars[k])}
                for k in range(n)
            ],
        }
    )

cases.append(
    {
        "fn": "clutterEpFit",
        "data": data.tolist(),
        "prior": {"mean": prior_mean.tolist(), "variance": float(prior_var)},
        "model": {"clutterWeight": w, "clutterVariance": a},
        "sweeps": sweeps,
        "expectedTrace": expected_trace,
    }
)

(FIXTURES / "expectationPropagation.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'expectationPropagation.json'}")
