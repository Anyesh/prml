"""Golden fixtures for packages/math/src/distributions/mvn.ts.

Follows tools/golden/README.md: {"cases": [...]}, each case an "fn" discriminant,
inputs, and an "expected" value. mvnSample has no fixture, matching the README's
policy for every sampler: there is no single correct sample sequence to pin down, only
a distribution to match, which the TypeScript test checks statistically against the
input parameters directly rather than against a fixture.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, multivariate_normal

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

mean3 = [1.0, -1.0, 2.0]
cov3 = [
    [3.0, 1.0, 0.5],
    [1.0, 2.0, 0.3],
    [0.5, 0.3, 1.0],
]
mean3_np = np.array(mean3)
cov3_np = np.array(cov3)

points = {
    "off_mean": [1.5, -0.5, 2.2],
    "at_mean": list(mean3),
    "origin": [0.0, 0.0, 0.0],
}

cases = []
for x in points.values():
    x_np = np.array(x)
    cases.append(
        {
            "fn": "mvnLogPdf",
            "x": x,
            "params": {"mean": mean3, "cov": cov3},
            "expected": float(multivariate_normal.logpdf(x_np, mean3_np, cov3_np)),
        }
    )
    cases.append(
        {
            "fn": "mvnPdf",
            "x": x,
            "params": {"mean": mean3, "cov": cov3},
            "expected": float(multivariate_normal.pdf(x_np, mean3_np, cov3_np)),
        }
    )

# keep is intentionally not ascending, to pin down that mvnMarginal follows `keep`'s order.
keep = [2, 0]
marginal_mean = mean3_np[keep]
marginal_cov = cov3_np[np.ix_(keep, keep)]
cases.append(
    {
        "fn": "mvnMarginal",
        "params": {"mean": mean3, "cov": cov3},
        "keep": keep,
        "expected": {"mean": marginal_mean.tolist(), "cov": marginal_cov.tolist()},
    }
)


def condition(observed_idx, xb):
    remaining_idx = [i for i in range(3) if i not in observed_idx]
    xb_np = np.array(xb)
    mu_a, mu_b = mean3_np[remaining_idx], mean3_np[observed_idx]
    saa = cov3_np[np.ix_(remaining_idx, remaining_idx)]
    sab = cov3_np[np.ix_(remaining_idx, observed_idx)]
    sbb = cov3_np[np.ix_(observed_idx, observed_idx)]
    mean_cond = mu_a + sab @ np.linalg.solve(sbb, xb_np - mu_b)
    cov_cond = saa - sab @ np.linalg.solve(sbb, sab.T)
    return mean_cond.tolist(), cov_cond.tolist()


mean_cond1, cov_cond1 = condition([1], [0.2])
cases.append(
    {
        "fn": "mvnConditional",
        "params": {"mean": mean3, "cov": cov3},
        "observed": {"1": 0.2},
        "expected": {"mean": mean_cond1, "cov": cov_cond1},
    }
)

mean_cond2, cov_cond2 = condition([0, 2], [0.1, -0.2])
cases.append(
    {
        "fn": "mvnConditional",
        "params": {"mean": mean3, "cov": cov3},
        "observed": {"0": 0.1, "2": -0.2},
        "expected": {"mean": mean_cond2, "cov": cov_cond2},
    }
)

# mvnCovarianceEllipse has no single scipy call to compare against; the chi-squared
# quantile that converts probability mass to a Mahalanobis radius is the one number
# worth pinning to an independent source (scipy's chi2.ppf), which the TypeScript test
# uses both for a direct isotropic-case check and for a property-based (rotation- and
# sign-independent) check on a non-diagonal covariance.
cases.append(
    {
        "fn": "mvnCovarianceEllipse_chi2Quantile",
        "mass": 0.95,
        "df": 2,
        "expected": float(chi2.ppf(0.95, df=2)),
    }
)
cases.append(
    {
        "fn": "mvnCovarianceEllipse_chi2Quantile",
        "mass": 0.9,
        "df": 2,
        "expected": float(chi2.ppf(0.9, df=2)),
    }
)

(FIXTURES / "mvn.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'mvn.json'}")
