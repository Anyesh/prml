"""Golden fixtures for packages/math/src/sampling/sir.ts (PRML 11.1.5).

Proposal is Exponential(1) (inverse-CDF sampled, matching transform.ts), target is
Gam(z|a, 1). Produces a bit-exact reference trajectory of the full two-stage
sampling-importance-resampling procedure against the shared PCG32 port: the proposal
draws, the normalized weights (independently re-derived here, not re-used from
sampling_importance.json), and the resampled indices via the same `categorical` scan
rng.ts already implements.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import expon, gamma as scipy_gamma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prml_rng_reference import Rng  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []


def sir_trace(rng: Rng, a, l):
    proposal_samples = [expon.ppf(rng.next(), scale=1.0) for _ in range(l)]
    log_weights = np.array(
        [
            scipy_gamma.logpdf(z, a) - expon.logpdf(z, scale=1.0)
            for z in proposal_samples
        ]
    )
    weights = np.exp(log_weights - log_weights.max())
    weights = weights / weights.sum()
    resampled = [proposal_samples[rng.categorical(weights.tolist())] for _ in range(l)]
    return proposal_samples, weights.tolist(), resampled


for seed, a, l in [(2026, 3.0, 6), (17, 6.5, 10)]:
    rng = Rng(seed, 2)
    proposal_samples, weights, resampled = sir_trace(rng, a, l)
    cases.append(
        {
            "fn": "samplingImportanceResampling",
            "seed": seed,
            "a": a,
            "l": l,
            "expected": {
                "proposalSamples": proposal_samples,
                "weights": weights,
                "resampled": resampled,
            },
        }
    )

(FIXTURES / "sampling_sir.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_sir.json'}")
