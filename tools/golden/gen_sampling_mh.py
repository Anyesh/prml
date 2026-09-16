"""Golden fixtures for packages/math/src/sampling/metropolisHastings.ts (PRML 11.2.2).

Two exact pieces: the acceptance-ratio arithmetic (11.44) for fixed state pairs (no
RNG), and a bit-exact reference trajectory of a full Metropolis-Hastings chain on a
correlated 2-D Gaussian target with an isotropic Gaussian random-walk proposal,
against the shared PCG32 (+ standard-normal) port. The target's log-density comes
from scipy.stats.multivariate_normal, independent of the TypeScript's own `mvnLogPdf`;
the proposal is symmetric so its log-ratio is 0 in both implementations regardless of
the exact isotropic formula used, which is why the reference does not need to
replicate `gaussianRandomWalkProposal`'s log-pdf line for line. Also covers
`autocorrelation` against numpy on a short fixed chain.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prml_rng_reference import Rng  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

for cand_lt, cur_lt, logq_a, logq_b in [
    (-2.0, -3.0, 0.0, 0.0),
    (-5.0, -1.0, 0.0, 0.0),
    (-1.0, -1.0, 0.3, -0.2),
]:
    ratio = cand_lt - cur_lt + logq_a - logq_b
    cases.append(
        {
            "fn": "mhAcceptanceLogRatio",
            "candidateLogTarget": cand_lt,
            "currentLogTarget": cur_lt,
            "logQCurrentGivenCandidate": logq_a,
            "logQCandidateGivenCurrent": logq_b,
            "expected": float(ratio),
        }
    )
    cases.append(
        {
            "fn": "mhAcceptanceProbability",
            "logRatio": ratio,
            "expected": float(min(1.0, np.exp(ratio))),
        }
    )

chain = [0.5, 1.2, 0.8, -0.3, -0.9, 0.1, 0.6, 1.0, -0.4, -1.1, 0.2, 0.7]
c = np.array(chain) - np.mean(chain)
c0 = np.mean(c**2)
max_lag = 5
n = len(chain)
rho = [
    1.0 if k == 0 else float(np.sum(c[: n - k] * c[k:]) / n / c0)
    for k in range(max_lag + 1)
]
cases.append(
    {"fn": "autocorrelation", "chain": chain, "maxLag": max_lag, "expected": rho}
)

MEAN = [0.0, 0.0]
COV = [[1.0, 0.8], [0.8, 1.0]]
target = multivariate_normal(mean=MEAN, cov=COV)


def mh_chain(rng: Rng, initial, n_steps, step_size):
    states = [list(initial)]
    accepted = []
    current = list(initial)
    for _ in range(n_steps):
        z1 = rng.standard_normal()
        z2 = rng.standard_normal()
        candidate = [current[0] + step_size * z1, current[1] + step_size * z2]
        log_ratio = target.logpdf(candidate) - target.logpdf(current)
        acceptance_probability = min(1.0, float(np.exp(log_ratio)))
        u = rng.next()
        accept = bool(u < acceptance_probability)
        current = candidate if accept else current
        states.append(list(current))
        accepted.append(accept)
    return states, accepted


for seed, step_size, n_steps in [(2026, 0.4, 30), (77, 1.5, 20)]:
    rng = Rng(seed, 5)
    states, accepted = mh_chain(rng, [0.0, 0.0], n_steps, step_size)
    cases.append(
        {
            "fn": "metropolisHastingsChain",
            "seed": seed,
            "stepSize": step_size,
            "nSteps": n_steps,
            "expected": {
                "states": states,
                "accepted": accepted,
                "acceptanceRate": sum(accepted) / len(accepted),
            },
        }
    )

(FIXTURES / "sampling_mh.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_mh.json'}")
