"""Golden fixtures for packages/math/src/ensemble/errorFunctions.ts.

Book section 14.3.1-14.3.2 and figures 14.3-14.4 compare error functions of the
classification margin z = t*y(x) (exponential, cross-entropy, hinge, misclassification)
and of the raw regression residual (squared, absolute). scipy's expit/logaddexp give an
independently-implemented numerically stable cross-entropy reference, not a re-derivation
of the softplus identity the TypeScript itself uses.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

Z_VALUES = [-50.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 0.5, 1.0, 2.0, 10.0, 50.0]
RESIDUAL_VALUES = [-10.0, -1.5, -1e-9, 0.0, 1e-9, 0.3, 1.0, 7.25]

cases = []

for z in Z_VALUES:
    cases.append(
        {"fn": "exponentialMarginError", "z": z, "expected": float(np.exp(-z))}
    )

    # ln(1 + exp(-z)) via logsumexp([0, -z]), scipy's own numerically stable path,
    # independent of the softplus identity the TypeScript implementation uses.
    cross_entropy = float(logsumexp([0.0, -z]))
    cases.append({"fn": "crossEntropyMarginError", "z": z, "expected": cross_entropy})

    cases.append(
        {"fn": "hingeMarginError", "z": z, "expected": float(max(0.0, 1.0 - z))}
    )

    cases.append(
        {
            "fn": "misclassificationMarginError",
            "z": z,
            "expected": 1.0 if z < 0 else 0.0,
        }
    )

for residual in RESIDUAL_VALUES:
    cases.append(
        {
            "fn": "squaredResidualError",
            "residual": residual,
            "expected": float(residual**2),
        }
    )
    cases.append(
        {
            "fn": "absoluteResidualError",
            "residual": residual,
            "expected": float(abs(residual)),
        }
    )

payload = {"cases": cases}
out = FIXTURES / "errorFunctions.json"
out.write_text(json.dumps(payload))
print(f"wrote {out} ({len(cases)} cases)")
