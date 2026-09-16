"""Golden fixtures for packages/math/src/distributions/wishart.ts.

Follows tools/golden/README.md: {"cases": [...]}, each case an "fn" discriminant,
inputs, and an "expected" value. wishartSample has no fixture, matching the README's
policy for every sampler: the TypeScript test checks the Bartlett decomposition
statistically against the analytic mean `nu * scale` directly, not against a fixture.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import digamma
from scipy.stats import wishart

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

W = [
    [2.0, 0.3, 0.1],
    [0.3, 1.5, 0.2],
    [0.1, 0.2, 1.0],
]
X = [
    [3.0, 0.5, 0.2],
    [0.5, 2.0, 0.4],
    [0.2, 0.4, 1.5],
]
W_np = np.array(W)
X_np = np.array(X)
nu = (
    6.5  # non-integer, exercised deliberately since PRML's Wishart allows real nu > d-1
)
d = W_np.shape[0]

dist = wishart(df=nu, scale=W_np)

expected_log_det = (
    sum(digamma((nu + 1 - i) / 2) for i in range(1, d + 1))
    + d * np.log(2)
    + np.linalg.slogdet(W_np)[1]
)

cases = [
    {
        "fn": "wishartLogPdf",
        "x": X,
        "params": {"scale": W, "nu": nu},
        "expected": float(dist.logpdf(X_np)),
    },
    {
        "fn": "wishartPdf",
        "x": X,
        "params": {"scale": W, "nu": nu},
        "expected": float(dist.pdf(X_np)),
    },
    {
        "fn": "wishartMean",
        "params": {"scale": W, "nu": nu},
        "expected": dist.mean().tolist(),
    },
    {
        "fn": "wishartExpectedLogDet",
        "params": {"scale": W, "nu": nu},
        "expected": float(expected_log_det),
    },
]

(FIXTURES / "wishart.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'wishart.json'}")
