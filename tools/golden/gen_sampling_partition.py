"""Golden fixtures for packages/math/src/sampling/partitionFunction.ts (PRML 11.6).

Every function here is a deterministic arithmetic reduction over a fixed sample set or
a fixed list of ratios, so every case is exact with no RNG involved. `E` and `G` are
two different 1-D Gaussian energies (unnormalised, `E(z) = z^2 / (2 sigma^2)`), whose
true partition-function ratio `Z_E/Z_G = sigma_E / sigma_G` is independently known in
closed form and used as a sanity check alongside the finite-sample estimate.
"""

import json
import math
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

SIGMA_E = 1.5
SIGMA_G = 2.0


def energy_e(z):
    return z * z / (2 * SIGMA_E * SIGMA_E)


def energy_g(z):
    return z * z / (2 * SIGMA_G * SIGMA_G)


samples = [-2.0, -1.0, -0.3, 0.1, 0.6, 1.4, 2.2, 3.0]
estimate = sum(math.exp(-energy_e(z) + energy_g(z)) for z in samples) / len(samples)
cases.append(
    {
        "fn": "partitionFunctionRatioEstimate",
        "samples": samples,
        "sigmaE": SIGMA_E,
        "sigmaG": SIGMA_G,
        "expected": estimate,
    }
)
cases.append(
    {
        "fn": "partitionFunctionRatioEstimate_trueRatio",
        "sigmaE": SIGMA_E,
        "sigmaG": SIGMA_G,
        "expected": SIGMA_E / SIGMA_G,
    }
)

for ratios in [[1.2, 0.8, 1.05], [1.0, 1.0, 1.0, 1.0], [0.5, 2.0]]:
    product = 1.0
    for r in ratios:
        product *= r
    cases.append(
        {"fn": "chainedPartitionFunctionRatio", "ratios": ratios, "expected": product}
    )

for alpha, z in [(0.0, 1.0), (1.0, 1.0), (0.5, 2.0), (0.25, -1.5)]:
    expected = (1 - alpha) * energy_e(z) + alpha * energy_g(z)
    cases.append(
        {
            "fn": "interpolatedEnergy",
            "alpha": alpha,
            "z": z,
            "sigmaE": SIGMA_E,
            "sigmaG": SIGMA_G,
            "expected": expected,
        }
    )

(FIXTURES / "sampling_partition.json").write_text(
    json.dumps({"cases": cases}, indent=2)
)
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_partition.json'}")
