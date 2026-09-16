"""Golden fixtures for packages/math/src/sequential/particleFilter.ts.

Plain numpy for the two deterministic pieces of the particle filter (PRML 13.117-13.119):
normalising raw emission likelihoods into weights, and systematic resampling from a single
fixed uniform draw. The RNG-driven wrapper around resampling (which draws that single
uniform value itself) is not fixture-tested, the same way `kmeansInit` is not: it only
consumes an already-verified `Rng`, so what needs checking is the deterministic core,
covered structurally instead in the vitest file.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def normalize_weights(raw):
    raw = np.array(raw)
    return (raw / raw.sum()).tolist()


def systematic_resample_indices(u0, weights, L):
    weights = np.array(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = (
        1.0  # guard float drift, matching the TS implementation's own guard
    )
    positions = (np.arange(L) + u0) / L
    return np.searchsorted(cumulative, positions).tolist()


cases = []

for raw in ([0.2, 0.5, 0.3], [1e-4, 1e-4, 100.0], [3.0, 1.0]):
    cases.append(
        {"fn": "particleFilterWeights", "raw": raw, "expected": normalize_weights(raw)}
    )

for u0, weights, L in (
    (0.05, [0.1, 0.2, 0.3, 0.4], 4),
    (0.5, [0.25, 0.25, 0.25, 0.25], 8),
    (0.0, [0.7, 0.1, 0.1, 0.1], 10),
    (0.9, [0.05, 0.9, 0.05], 6),
):
    cases.append(
        {
            "fn": "systematicResampleIndices",
            "u0": u0,
            "weights": weights,
            "L": L,
            "expected": systematic_resample_indices(u0, weights, L),
        }
    )

(FIXTURES / "particle_filter.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'particle_filter.json'}")
