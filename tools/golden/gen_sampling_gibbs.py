"""Golden fixtures for packages/math/src/sampling/gibbs.ts (PRML 11.3).

Target is the correlated bivariate Gaussian PRML's own Figure 11.11 illustrates.
The conditional mean/variance formulas are the 2x2 specialisation of PRML 2.81-2.82,
computed here independently of the shared `mvnConditional` the TypeScript reuses.
Produces a bit-exact reference trajectory of the full coordinate-by-coordinate sweep
sequence against the shared PCG32 (+ standard-normal) port, plus an exact check of the
over-relaxation update (11.50) for fixed inputs.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prml_rng_reference import Rng  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

MEAN = [0.0, 0.0]
COV = [[1.0, 0.8], [0.8, 1.0]]


def conditional_1_given_2(z2):
    mean = MEAN[0] + (COV[0][1] / COV[1][1]) * (z2 - MEAN[1])
    var = COV[0][0] - (COV[0][1] ** 2) / COV[1][1]
    return mean, var


def conditional_2_given_1(z1):
    mean = MEAN[1] + (COV[1][0] / COV[0][0]) * (z1 - MEAN[0])
    var = COV[1][1] - (COV[1][0] ** 2) / COV[0][0]
    return mean, var


def gibbs_chain(rng: Rng, initial, n_sweeps):
    current = list(initial)
    states = [list(current)]
    updates = []
    for _ in range(n_sweeps):
        m0, v0 = conditional_1_given_2(current[1])
        current[0] = m0 + (v0**0.5) * rng.standard_normal()
        updates.append({"index": 0, "z": list(current)})

        m1, v1 = conditional_2_given_1(current[0])
        current[1] = m1 + (v1**0.5) * rng.standard_normal()
        updates.append({"index": 1, "z": list(current)})

        states.append(list(current))
    return states, updates


for seed, initial, n_sweeps in [(2026, [3.0, -3.0], 15), (5, [0.0, 0.0], 10)]:
    rng = Rng(seed, 6)
    states, updates = gibbs_chain(rng, initial, n_sweeps)
    cases.append(
        {
            "fn": "gibbsSampleMvn",
            "seed": seed,
            "initial": initial,
            "nSweeps": n_sweeps,
            "mean": MEAN,
            "cov": COV,
            "expected": {"states": states, "updates": updates},
        }
    )

for zi, mean, variance, alpha, nu in [
    (2.0, 0.0, 1.0, 0.0, 0.5),
    (2.0, 0.0, 1.0, -0.7, 0.5),
    (1.0, 0.5, 4.0, 0.3, -1.2),
]:
    expected = (
        mean + alpha * (zi - mean) + (variance**0.5) * ((1 - alpha * alpha) ** 0.5) * nu
    )
    cases.append(
        {
            "fn": "overRelaxationStep",
            "zi": zi,
            "mean": mean,
            "variance": variance,
            "alpha": alpha,
            "nu": nu,
            "expected": float(expected),
        }
    )

(FIXTURES / "sampling_gibbs.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_gibbs.json'}")
