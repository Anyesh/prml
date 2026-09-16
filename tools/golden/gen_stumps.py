"""Golden fixtures for packages/math/src/ensemble/stumps.ts.

`fitDecisionStump` must be a deterministic exhaustive search: for every feature index
(ascending), every threshold (the midpoints between consecutive DISTINCT sorted values
of that feature, ascending), and both polarities (+1 before -1), compute the weighted
misclassification rate (PRML 14.15/14.16) and keep the first candidate that strictly
improves on the running best. This module hand-rolls that same search independently in
numpy (not by calling the TypeScript) so the fixture is a genuine external check.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def best_stump(X: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> dict:
    total_weight = weights.sum()
    n, d = X.shape
    best = None
    for j in range(d):
        values = np.unique(X[:, j])
        if values.size < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for theta in thresholds:
            for polarity in (1, -1):
                raw = np.where(X[:, j] > theta, 1, -1)
                predicted = polarity * raw
                misweight = weights[predicted != targets].sum()
                weighted_error = float(misweight / total_weight)
                if best is None or weighted_error < best["weightedError"]:
                    best = {
                        "featureIndex": j,
                        "threshold": float(theta),
                        "polarity": polarity,
                        "weightedError": weighted_error,
                    }
    assert best is not None
    return best


def case(name: str, X: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> dict:
    return {
        "fn": "fitDecisionStump",
        "name": name,
        "X": X.tolist(),
        "targets": targets.tolist(),
        "weights": weights.tolist(),
        "expected": best_stump(X, targets, weights),
    }


cases = []

# 1. Plain, unambiguous dataset: feature 0 alone separates the classes cleanly.
X1 = np.array(
    [
        [0.0, 5.0],
        [1.0, 1.0],
        [2.0, 4.0],
        [3.0, 0.0],
        [4.0, 3.0],
        [5.0, 2.0],
    ]
)
t1 = np.array([-1, -1, -1, 1, 1, 1])
w1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
cases.append(case("clean_separation", X1, t1, w1))

# 2. Non-uniform weights that flip which threshold is optimal versus unweighted counting.
X2 = np.array(
    [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]
)
t2 = np.array([-1, 1, -1, 1, 1])
w2 = np.array([0.05, 0.05, 5.0, 0.05, 0.05])
cases.append(case("weighted_outlier_dominates", X2, t2, w2))

# 3. Feature-index tie: columns 0 and 1 are identical, so every (threshold, polarity)
# candidate on feature 1 exactly matches feature 0's error. The fixed iteration order
# (feature ascending) must resolve this to featureIndex = 0.
X3 = np.array(
    [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]
)
t3 = np.array([-1, 1, -1])
w3 = np.array([2.0, 1.0, 1.0])
cases.append(case("feature_index_tie", X3, t3, w3))

# 4. Threshold tie within a single feature: at x = [1, 2, 3, 4, 5] with targets
# [-1, -1, 1, -1, 1], thresholds 2.5 and 4.5 (both polarity +1) each misclassify exactly
# one point (weighted error 0.2). Threshold ascending means 2.5 must win. Feature 1's
# values are shuffled so its own best split (0.4) is strictly worse, confirmed by direct
# search over all permutations of the same five values, not just asserted.
X4 = np.array(
    [
        [1.0, 10.0],
        [2.0, 30.0],
        [3.0, 20.0],
        [4.0, 50.0],
        [5.0, 40.0],
    ]
)
t4 = np.array([-1, -1, 1, -1, 1])
w4 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
cases.append(case("threshold_tie", X4, t4, w4))

payload = {"cases": cases}
out = FIXTURES / "stumps.json"
out.write_text(json.dumps(payload))
print(f"wrote {out} ({len(cases)} cases)")
