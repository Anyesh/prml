"""Golden fixtures for packages/math/src/sampling/importance.ts (PRML 11.1.4).

Fully deterministic given a fixed sample set, so every case here is exact: no RNG is
involved anywhere in importance weighting or the effective-sample-size diagnostic.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

samples = [-4.0, -2.0, -0.5, 0.5, 1.5, 2.0, 3.0, 6.0]
target = norm(loc=2.0, scale=1.0)
proposal = norm(loc=0.0, scale=3.0)

log_weights = target.logpdf(samples) - proposal.logpdf(samples)
weights = np.exp(log_weights - log_weights.max())
weights = weights / weights.sum()

cases.append(
    {
        "fn": "importanceLogWeights",
        "samples": samples,
        "targetMean": 2.0,
        "targetScale": 1.0,
        "proposalMean": 0.0,
        "proposalScale": 3.0,
        "expected": log_weights.tolist(),
    }
)
cases.append(
    {
        "fn": "normalizeImportanceWeights",
        "logWeights": log_weights.tolist(),
        "expected": weights.tolist(),
    }
)

for values, label in [(samples, "identity"), ([s * s for s in samples], "square")]:
    estimate = float(np.sum(weights * np.array(values)))
    cases.append(
        {
            "fn": "importanceEstimate",
            "label": label,
            "values": values,
            "weights": weights.tolist(),
            "expected": estimate,
        }
    )

ess = float(1.0 / np.sum(weights**2))
cases.append(
    {"fn": "effectiveSampleSize", "weights": weights.tolist(), "expected": ess}
)

# A degenerate case: one sample carries almost all the weight (11.1.4's failure mode).
degenerate_weights = [0.001, 0.001, 0.001, 0.994, 0.001, 0.001, 0.0005, 0.0005]
degenerate_weights = (np.array(degenerate_weights) / sum(degenerate_weights)).tolist()
ess_degenerate = float(1.0 / np.sum(np.array(degenerate_weights) ** 2))
cases.append(
    {
        "fn": "effectiveSampleSize",
        "weights": degenerate_weights,
        "expected": ess_degenerate,
    }
)

(FIXTURES / "sampling_importance.json").write_text(
    json.dumps({"cases": cases}, indent=2)
)
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_importance.json'}")
