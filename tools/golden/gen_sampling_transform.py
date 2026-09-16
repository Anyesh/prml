"""Golden fixtures for packages/math/src/sampling/transform.ts (PRML 11.1.1).

Covers the deterministic pieces of the transformation method: the exponential
inverse-CDF (11.6-11.7), the Cauchy inverse-CDF used as the rejection-sampling
envelope in 11.1.2 (11.8, 11.16, exercise 11.7), and a full bit-exact trace of the
Box-Muller/Marsaglia-polar procedure (11.10-11.12) against the shared PCG32 reference
port, since that procedure has no scipy call to compare against but is otherwise
fully deterministic once the RNG stream is fixed.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import cauchy, expon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prml_rng_reference import Rng  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

for lam in [0.5, 1.0, 3.0]:
    for u in [0.001, 0.25, 0.5, 0.75, 0.999]:
        y = expon.ppf(u, scale=1 / lam)
        cases.append(
            {"fn": "exponentialQuantile", "u": u, "lambda": lam, "expected": float(y)}
        )
        cases.append(
            {
                "fn": "exponentialLogPdf",
                "x": float(y),
                "lambda": lam,
                "expected": float(expon.logpdf(y, scale=1 / lam)),
            }
        )

for b, c in [(1.0, 0.0), (2.5, -1.0), (0.7, 3.0)]:
    for u in [0.001, 0.1, 0.5, 0.9, 0.999]:
        y = cauchy.ppf(u, loc=c, scale=b)
        cases.append(
            {"fn": "cauchyQuantile", "u": u, "b": b, "c": c, "expected": float(y)}
        )
    for x in [c - 5 * b, c - b, c, c + b, c + 5 * b]:
        cases.append(
            {
                "fn": "cauchyPdf",
                "x": float(x),
                "b": b,
                "c": c,
                "expected": float(cauchy.pdf(x, loc=c, scale=b)),
            }
        )


def box_muller_trace(rng: Rng):
    attempts = []
    while True:
        z1 = 2 * rng.next() - 1
        z2 = 2 * rng.next() - 1
        r2 = z1 * z1 + z2 * z2
        accepted = 0 < r2 <= 1
        attempts.append({"z1": z1, "z2": z2, "r2": r2, "accepted": accepted})
        if accepted:
            break
    factor = np.sqrt(-2 * np.log(r2) / r2)
    y1 = z1 * factor
    y2 = z2 * factor
    return {"attempts": attempts, "y1": float(y1), "y2": float(y2)}


for seed in [1, 12345, 999]:
    rng = Rng(seed, 1)
    trace = box_muller_trace(rng)
    cases.append({"fn": "boxMullerTrace", "seed": seed, "stream": 1, "expected": trace})

(FIXTURES / "sampling_transform.json").write_text(
    json.dumps({"cases": cases}, indent=2)
)
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_transform.json'}")
