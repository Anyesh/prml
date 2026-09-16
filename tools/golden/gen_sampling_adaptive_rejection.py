"""Golden fixtures for packages/math/src/sampling/adaptiveRejection.ts (PRML 11.1.3).

Target is a standard normal restricted to a finite domain (log-concave: h(z) = -z^2/2
minus a constant, h'(z) = -z). Covers the tangent-line intersection breakpoints
(computed independently here from the same closed form, not by calling into the
TypeScript), the piecewise-exponential area/quantile formulas via numerical
integration (scipy.integrate.quad as an independent check of the closed form used by
the TypeScript inverse-CDF), and a bit-exact reference trajectory of the full
draw-test-refine loop against the shared PCG32 port.
"""

import json
import math
import sys
from pathlib import Path

from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prml_rng_reference import Rng  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

LOG_2PI = math.log(2 * math.pi)


def h(z):
    return -0.5 * z * z - 0.5 * LOG_2PI


def h_prime(z):
    return -z


DOMAIN = (-6.0, 6.0)
GRID = [-2.0, -0.5, 0.0, 1.0, 3.0]

cases = []


def tangent(z0, h0, hp0, z):
    return h0 + hp0 * (z - z0)


def intersect(za, ha, hpa, zb, hb, hpb):
    return (hb - ha - zb * hpb + za * hpa) / (hpa - hpb)


points = [(z, h(z), h_prime(z)) for z in GRID]
breakpoints = [DOMAIN[0]]
for i in range(len(points) - 1):
    za, ha, hpa = points[i]
    zb, hb, hpb = points[i + 1]
    breakpoints.append(intersect(za, ha, hpa, zb, hb, hpb))
breakpoints.append(DOMAIN[1])

pieces = []
for i, (z, hv, hp) in enumerate(points):
    pieces.append(
        {"lo": breakpoints[i], "hi": breakpoints[i + 1], "z": z, "h": hv, "hPrime": hp}
    )

cases.append(
    {
        "fn": "arsBuildEnvelope",
        "points": [{"z": z, "h": hv, "hPrime": hp} for z, hv, hp in points],
        "domain": list(DOMAIN),
        "expected": {"pieces": pieces},
    }
)

# Independent numerical-integration check of each piece's area and its median (t=0.5)
# quantile, against the closed-form inverse-CDF the TypeScript implementation uses.
shift = max(p["h"] for p in pieces)


def area_up_to(p, z):
    def integrand(zz):
        return math.exp(tangent(p["z"], p["h"], p["hPrime"], zz) - shift)

    val, _ = quad(integrand, p["lo"], z)
    return val


for i, p in enumerate(pieces):
    area = area_up_to(p, p["hi"])
    cases.append(
        {"fn": "arsPieceArea", "pieceIndex": i, "shift": shift, "expected": float(area)}
    )

    target_area = 0.5 * area
    median = brentq(
        lambda z, p=p, target_area=target_area: area_up_to(p, z) - target_area,
        p["lo"],
        p["hi"],
    )
    cases.append(
        {
            "fn": "arsPieceQuantile",
            "pieceIndex": i,
            "shift": shift,
            "t": 0.5,
            "expected": float(median),
        }
    )


def build_envelope(pts, domain):
    pts = sorted(pts, key=lambda p: p[0])
    bps = [domain[0]]
    for i in range(len(pts) - 1):
        za, ha, hpa = pts[i]
        zb, hb, hpb = pts[i + 1]
        bps.append(intersect(za, ha, hpa, zb, hb, hpb))
    bps.append(domain[1])
    return [
        {"lo": bps[i], "hi": bps[i + 1], "z": z, "h": hv, "hPrime": hp}
        for i, (z, hv, hp) in enumerate(pts)
    ]


def piece_area(p, shift):
    lo_h = tangent(p["z"], p["h"], p["hPrime"], p["lo"]) - shift
    hi_h = tangent(p["z"], p["h"], p["hPrime"], p["hi"]) - shift
    if abs(p["hPrime"]) < 1e-10:
        return math.exp(lo_h) * (p["hi"] - p["lo"])
    return (math.exp(hi_h) - math.exp(lo_h)) / p["hPrime"]


def piece_quantile(p, shift, t):
    if abs(p["hPrime"]) < 1e-10:
        return p["lo"] + t * (p["hi"] - p["lo"])
    lo_h = tangent(p["z"], p["h"], p["hPrime"], p["lo"]) - shift
    hi_h = tangent(p["z"], p["h"], p["hPrime"], p["hi"]) - shift
    target = lo_h + math.log(1 - t + t * math.exp(hi_h - lo_h))
    return p["z"] + (target + shift - p["h"]) / p["hPrime"]


def envelope_sample(rng: Rng, env_pieces):
    local_shift = max(p["h"] for p in env_pieces)
    areas = [piece_area(p, local_shift) for p in env_pieces]
    idx = rng.categorical(areas)
    t = rng.next()
    z = piece_quantile(env_pieces[idx], local_shift, t)
    return z, tangent(
        env_pieces[idx]["z"], env_pieces[idx]["h"], env_pieces[idx]["hPrime"], z
    )


def ars_sample_trace(rng: Rng, initial_grid, domain, max_attempts=200):
    pts = [(z, h(z), h_prime(z)) for z in initial_grid]
    env = build_envelope(pts, domain)
    trace = []
    for _ in range(max_attempts):
        z, env_h = envelope_sample(rng, env)
        log_u = math.log(rng.next())
        true_h = h(z)
        accepted = bool(log_u <= true_h - env_h)
        trace.append(
            {
                "z": z,
                "accepted": accepted,
                "envelopeLogHeight": env_h,
                "trueLog": true_h,
            }
        )
        if accepted:
            return z, env, trace
        pts = [(p["z"], p["h"], p["hPrime"]) for p in env] + [(z, true_h, h_prime(z))]
        env = build_envelope(pts, domain)
    raise RuntimeError("exceeded max_attempts")


for seed in [2026, 4177]:
    rng = Rng(seed, 1)
    sample, final_env, trace = ars_sample_trace(rng, GRID, DOMAIN)
    cases.append(
        {
            "fn": "arsSample",
            "seed": seed,
            "grid": GRID,
            "domain": list(DOMAIN),
            "expected": {
                "sample": sample,
                "trace": trace,
                "finalPieceCount": len(final_env),
            },
        }
    )

(FIXTURES / "sampling_adaptive_rejection.json").write_text(
    json.dumps({"cases": cases}, indent=2)
)
print(f"wrote {len(cases)} cases to {FIXTURES / 'sampling_adaptive_rejection.json'}")
