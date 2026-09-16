"""Golden fixtures for packages/math/src/sampling/hmc.ts (PRML 11.5).

The target is a correlated 2-D Gaussian with energy `E(z) = 0.5 (z-mean)^T Cov^-1
(z-mean)` (dropping the log-normalising constant, which every sampler in this chapter
is allowed to do). Its gradient is `Cov^-1 (z - mean)`, computed here via
`numpy.linalg.solve`, independent of the TypeScript's own linear-algebra module.
Covers one exact leapfrog step from a fixed (z, r) (no RNG at all), and a bit-exact
reference trajectory of the full momentum-resample/leapfrog/accept-reject chain
against the shared PCG32 (+ standard-normal) port.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prml_rng_reference import Rng  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

MEAN = np.array([0.5, -0.2])
COV = np.array([[1.5, 0.9], [0.9, 1.0]])


def energy(z):
    d = np.array(z) - MEAN
    return float(0.5 * d @ np.linalg.solve(COV, d))


def grad_energy(z):
    d = np.array(z) - MEAN
    return np.linalg.solve(COV, d).tolist()


def leapfrog_step(z, r, epsilon):
    z = np.array(z, dtype=float)
    r = np.array(r, dtype=float)
    half_r = r - (epsilon / 2) * np.array(grad_energy(z.tolist()))
    next_z = z + epsilon * half_r
    next_r = half_r - (epsilon / 2) * np.array(grad_energy(next_z.tolist()))
    return next_z.tolist(), next_r.tolist()


for z0, r0, epsilon in [
    ([1.0, -1.0], [0.5, 0.3], 0.1),
    ([0.0, 0.0], [1.2, -0.7], 0.25),
]:
    z1, r1 = leapfrog_step(z0, r0, epsilon)
    cases.append(
        {
            "fn": "leapfrogStep",
            "z": z0,
            "r": r0,
            "epsilon": epsilon,
            "expected": {"z": z1, "r": r1},
        }
    )

for z, r in [([1.0, -1.0], [0.5, 0.3]), ([0.0, 0.0], [0.0, 0.0])]:
    h = energy(z) + 0.5 * float(np.sum(np.array(r) ** 2))
    cases.append({"fn": "hamiltonian", "z": z, "r": r, "expected": h})


def hmc_chain(rng: Rng, initial, n_steps, epsilon, l):
    current = list(initial)
    states = [list(current)]
    accepted = []
    for _ in range(n_steps):
        r0 = [rng.standard_normal() for _ in current]
        direction = 1.0 if rng.next() < 0.5 else -1.0
        z, r = current, r0
        for _ in range(l):
            z, r = leapfrog_step(z, r, direction * epsilon)
        h0 = energy(current) + 0.5 * sum(x * x for x in r0)
        h1 = energy(z) + 0.5 * sum(x * x for x in r)
        acceptance_probability = min(1.0, float(np.exp(h0 - h1)))
        accept = bool(rng.next() < acceptance_probability)
        current = z if accept else current
        states.append(list(current))
        accepted.append(accept)
    return states, accepted


for seed, epsilon, l, n_steps in [(2026, 0.15, 10, 15), (13, 0.3, 6, 12)]:
    rng = Rng(seed, 14)
    states, accepted = hmc_chain(rng, [0.0, 0.0], n_steps, epsilon, l)
    cases.append(
        {
            "fn": "hmcChain",
            "seed": seed,
            "epsilon": epsilon,
            "l": l,
            "nSteps": n_steps,
            "expected": {
                "states": states,
                "accepted": accepted,
                "acceptanceRate": sum(accepted) / len(accepted),
            },
        }
    )

(FIXTURES / "sampling_hmc.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_hmc.json'}")
