"""Golden fixtures for packages/math/src/latent/manifold.ts.

PRML 12.4.3's principal curves (Hastie & Stuetzle 1989): a curve f(lambda) satisfying
the self-consistency condition E[x | g_f(x) = lambda] = f(lambda) (12.92), fitted by
alternating a projection step (nearest point on the current curve, by arclength) with a
smoothing step (each curve vertex moved to a Gaussian-kernel-weighted average of the data,
weighted by closeness in arclength to that vertex). This is the same iterate-trace pattern
as gen_em.py and gen_ppca.py: the loop is reimplemented directly in numpy, not re-derived
by an independent method, since there is no scipy primitive for principal curves.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def swiss_roll_point(t, height):
    return np.array([t * np.cos(t), height, t * np.sin(t)])


ts = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
heights = np.array([-1.0, 0.5, -0.5, 1.0, 0.0])
swiss_points = np.array([swiss_roll_point(t, h) for t, h in zip(ts, heights)])


def arclengths(polyline):
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_to_polyline(point, polyline):
    arcs = arclengths(polyline)
    best_dist = np.inf
    best_arc = 0.0
    best_point = polyline[0]
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        ab = b - a
        denom = ab @ ab
        t = 0.0 if denom == 0 else np.clip((point - a) @ ab / denom, 0.0, 1.0)
        proj = a + t * ab
        dist = np.linalg.norm(point - proj)
        if dist < best_dist:
            best_dist = dist
            best_point = proj
            best_arc = arcs[i] + t * (arcs[i + 1] - arcs[i])
    return best_arc, best_point, best_dist


rng = np.random.default_rng(20260916)
n = 24
t_data = rng.uniform(2.0, 6.0, n)
h_data = rng.uniform(-1.0, 1.0, n)
data = np.array([swiss_roll_point(t, h) for t, h in zip(t_data, h_data)])
data += rng.normal(scale=0.1, size=data.shape)

num_vertices = 6
init_curve = np.array(
    [swiss_roll_point(t, 0.0) for t in np.linspace(2.0, 6.0, num_vertices)]
)

bandwidth = 1.5


def smooth_step(data, curve):
    arcs_curve = arclengths(curve)
    proj_arcs = np.array([project_to_polyline(x, curve)[0] for x in data])
    new_curve = np.zeros_like(curve)
    for i, a_i in enumerate(arcs_curve):
        weights = np.exp(-0.5 * ((proj_arcs - a_i) / bandwidth) ** 2)
        weights_sum = weights.sum()
        if weights_sum < 1e-12:
            new_curve[i] = curve[i]
        else:
            new_curve[i] = (weights[:, None] * data).sum(axis=0) / weights_sum
    return new_curve


curve = init_curve.copy()
trace = []
for _ in range(3):
    curve = smooth_step(data, curve)
    trace.append(curve.tolist())

cases = [
    {
        "fn": "swissRollPoint",
        "t": float(t),
        "height": float(h),
        "expected": swiss_roll_point(t, h).tolist(),
    }
    for t, h in zip(ts, heights)
]
cases += [
    {
        "fn": "projectToPolyline",
        "point": swiss_points[2].tolist(),
        "polyline": init_curve.tolist(),
        "expected": {
            "arclength": float(project_to_polyline(swiss_points[2], init_curve)[0]),
            "distance": float(project_to_polyline(swiss_points[2], init_curve)[2]),
        },
    },
    {
        "fn": "principalCurveFit",
        "data": data.tolist(),
        "initialCurve": init_curve.tolist(),
        "bandwidth": bandwidth,
        "steps": 3,
        "expectedTrace": trace,
    },
]

(FIXTURES / "manifold.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'manifold.json'}")
