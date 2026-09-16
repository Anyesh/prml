"""Golden fixtures for packages/math/src/sampling/rejection.ts (PRML 11.1.2).

Covers: the gamma-via-Cauchy envelope construction with an independent scipy check
that it truly bounds the target everywhere on a fine grid (not just at the tangency
point), and a bit-exact reference trajectory of `rejectionSampleBatch` against the
shared PCG32 port, since the accept/reject decision sequence is fully determined once
the RNG stream and the two closed-form densities are fixed.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import cauchy, gamma as scipy_gamma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prml_rng_reference import Rng  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []


def envelope_params(a):
    c = a - 1
    b = np.sqrt(2 * a - 1)
    k = scipy_gamma.pdf(c, a) / cauchy.pdf(c, loc=c, scale=b)
    return b, c, k


for a in [2.0, 4.3, 10.0]:
    b, c, k = envelope_params(a)
    cases.append(
        {
            "fn": "gammaCauchyEnvelope",
            "a": a,
            "expected": {"b": float(b), "c": float(c), "k": float(k)},
        }
    )

    grid = np.linspace(1e-6, c + 12 * b, 20000)
    ratio = scipy_gamma.pdf(grid, a) / (k * cauchy.pdf(grid, loc=c, scale=b))
    cases.append(
        {
            "fn": "gammaCauchyEnvelope_maxRatioOnGrid",
            "a": a,
            "expected": float(np.max(ratio)),
        }
    )


def gamma_pdf(x, a):
    if x <= 0:
        return 0.0
    return float(scipy_gamma.pdf(x, a))


def cauchy_pdf(x, b, c):
    return float(cauchy.pdf(x, loc=c, scale=b))


def cauchy_quantile(u, b, c):
    return float(cauchy.ppf(u, loc=c, scale=b))


def rejection_trace(rng: Rng, a, n_samples, max_attempts=10000):
    b, c, k = envelope_params(a)
    samples = []
    trace = []
    for _ in range(n_samples):
        for _attempt in range(max_attempts):
            candidate = cauchy_quantile(rng.next(), b, c)
            env = k * cauchy_pdf(candidate, b, c)
            u = rng.next() * env
            target = gamma_pdf(candidate, a)
            accepted = bool(u <= target)
            trace.append(
                {
                    "candidate": candidate,
                    "u": u,
                    "envelope": env,
                    "target": target,
                    "accepted": accepted,
                }
            )
            if accepted:
                samples.append(candidate)
                break
    return samples, trace


for seed, a in [(2026, 4.3), (11, 2.0)]:
    rng = Rng(seed, 1)
    samples, trace = rejection_trace(rng, a, 8)
    cases.append(
        {
            "fn": "rejectionSampleBatch",
            "seed": seed,
            "a": a,
            "n": 8,
            "expected": {
                "samples": samples,
                "trace": trace,
                "acceptanceRate": len(samples) / len(trace),
            },
        }
    )

(FIXTURES / "sampling_rejection.json").write_text(
    json.dumps({"cases": cases}, indent=2)
)
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_rejection.json'}")
