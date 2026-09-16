"""Golden fixtures for packages/math/src/sampling/sliceSampling.ts (PRML 11.4).

`sliceSteppingOut` is fully deterministic given (z0, logU, w) and needs no RNG at all,
so it gets a plain exact fixture. The full step-and-shrink chain does consume the RNG
(the vertical level draw and every shrinkage proposal), so it gets a bit-exact
reference trajectory against the shared PCG32 port instead, on a standard normal
target restricted to nothing (its log-density is finite and smooth everywhere in the
range these fixtures probe).
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prml_rng_reference import Rng  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

LOG_2PI = math.log(2 * math.pi)


def log_target(z):
    return -0.5 * z * z - 0.5 * LOG_2PI


cases = []

for z0, log_u, w in [
    (0.0, math.log(0.3) + log_target(0.0), 0.5),
    (2.0, math.log(0.01) + log_target(2.0), 0.2),
    (-1.5, math.log(0.6) + log_target(-1.5), 1.0),
]:
    lo = z0 - w / 2
    hi = z0 + w / 2
    while log_target(lo) > log_u:
        lo -= w
    while log_target(hi) > log_u:
        hi += w
    cases.append(
        {
            "fn": "sliceSteppingOut",
            "z0": z0,
            "logU": log_u,
            "w": w,
            "expected": {"lo": lo, "hi": hi},
        }
    )


def slice_step(rng: Rng, z, w):
    log_u = log_target(z) + math.log(rng.next())
    lo = z - w / 2
    hi = z + w / 2
    while log_target(lo) > log_u:
        lo -= w
    while log_target(hi) > log_u:
        hi += w
    proposals = []
    while True:
        candidate = lo + rng.next() * (hi - lo)
        accepted = bool(log_target(candidate) > log_u)
        proposals.append({"candidate": candidate, "accepted": accepted})
        if accepted:
            return candidate, log_u, {"lo": lo, "hi": hi}, proposals
        if candidate < z:
            lo = candidate
        else:
            hi = candidate


def slice_chain(rng: Rng, initial, n_steps, w):
    current = initial
    states = [current]
    steps = []
    for _ in range(n_steps):
        current, log_u, interval, proposals = slice_step(rng, current, w)
        states.append(current)
        steps.append(
            {"z": current, "logU": log_u, "interval": interval, "proposals": proposals}
        )
    return states, steps


for seed, initial, n_steps, w in [(2026, 0.0, 12, 1.0), (7, 3.0, 8, 0.5)]:
    rng = Rng(seed, 9)
    states, steps = slice_chain(rng, initial, n_steps, w)
    cases.append(
        {
            "fn": "sliceSampleChain",
            "seed": seed,
            "initial": initial,
            "nSteps": n_steps,
            "w": w,
            "expected": {"states": states, "steps": steps},
        }
    )

(FIXTURES / "sampling_slice.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_slice.json'}")
