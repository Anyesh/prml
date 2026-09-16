"""Golden fixtures for packages/math/src/mixtures/kmeans.ts.

Every case is a plain numpy re-implementation of the textbook K-means steps (PRML 9.1
assign, mean update, 9.1 distortion measure J), driven from fixed, hand-picked data and
initial means rather than through any random initialisation, so nothing here depends on
the seeded PCG32 RNG the TypeScript side uses. `kmeansInit` itself has no fixture, per
tools/golden/README.md's policy for RNG-driven functions: its test checks structural
properties (k distinct rows of the input data, deterministic given a seed) instead.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

# Two visible blobs around (0, 0) and (5, 5), plus one point equidistant from both
# starting means to pin down the tie-break rule (lowest index wins).
data = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0],
        [2.5, 2.5],  # equidistant from (0,0)-ish and (5,5)-ish starting means
    ]
)


def assign(data, means):
    d2 = ((data[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def update_means(data, assignments, k, previous):
    means = previous.copy()
    for j in range(k):
        pts = data[assignments == j]
        if len(pts) > 0:
            means[j] = pts.mean(axis=0)
    return means


def distortion(data, means, assignments):
    return float(((data - means[assignments]) ** 2).sum())


cases = []

means0 = np.array([[0.0, 0.0], [5.0, 5.0]])
assignments0 = assign(data, means0)
cases.append(
    {
        "fn": "kmeansAssign",
        "data": data.tolist(),
        "means": means0.tolist(),
        "expected": assignments0.tolist(),
    }
)
cases.append(
    {
        "fn": "kmeansDistortion",
        "data": data.tolist(),
        "means": means0.tolist(),
        "assignments": assignments0.tolist(),
        "expected": distortion(data, means0, assignments0),
    }
)

means1 = update_means(data, assignments0, 2, means0)
cases.append(
    {
        "fn": "kmeansUpdateMeans",
        "data": data.tolist(),
        "assignments": assignments0.tolist(),
        "k": 2,
        "previousMeans": means0.tolist(),
        "expected": means1.tolist(),
    }
)

# An empty-cluster case: three means, only two ever receive a point, so the third must
# be left exactly at its previous value rather than becoming NaN.
means_empty = np.array([[0.0, 0.0], [5.0, 5.0], [100.0, 100.0]])
assignments_empty = assign(data, means_empty)
means_empty_updated = update_means(data, assignments_empty, 3, means_empty)
cases.append(
    {
        "fn": "kmeansUpdateMeans_emptyCluster",
        "data": data.tolist(),
        "assignments": assignments_empty.tolist(),
        "k": 3,
        "previousMeans": means_empty.tolist(),
        "expected": means_empty_updated.tolist(),
    }
)

# Full iterate-by-iterate trace from fixed initial means, for kmeansFit's internal loop
# (assign -> update -> distortion, repeated), independent of any RNG.
means = means0.copy()
trace_means = [means.tolist()]
trace_assignments = []
trace_distortion = []
for _ in range(4):
    a = assign(data, means)
    trace_assignments.append(a.tolist())
    trace_distortion.append(distortion(data, means, a))
    means = update_means(data, a, 2, means)
    trace_means.append(means.tolist())

cases.append(
    {
        "fn": "kmeansTrace",
        "data": data.tolist(),
        "initialMeans": means0.tolist(),
        "steps": 4,
        "expectedMeansHistory": trace_means,
        "expectedAssignmentsHistory": trace_assignments,
        "expectedDistortionHistory": trace_distortion,
    }
)

(FIXTURES / "kmeans.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'kmeans.json'}")
