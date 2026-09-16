"""Golden fixtures for packages/math/src/ensemble/adaboost.ts.

AdaBoost (PRML 14.15-14.19) with a decision stump as its weak learner. This reference
is written from scratch in numpy: it reimplements the same feature-ascending /
threshold-ascending / polarity-(+1-then--1) / first-strict-improvement stump search as
`gen_stumps.py` and `stumps.ts` (not a call into either), then runs the boosting loop
independently of the TypeScript.

Weight update is 14.18 exactly: correctly classified points keep their weight (exp(0) =
1); nothing is renormalised, since epsilon's ratio-of-sums already makes each round
scale-invariant. epsilon = 0 or 1 would send ln((1-eps)/eps) to +-infinity, so alpha is
computed from epsilon clamped into [1e-10, 1 - 1e-10]; the recorded `epsilon` itself is
the raw (unclamped) weighted error.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

EPSILON_FLOOR = 1e-10


def fit_stump(X: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> dict:
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


def stump_predict(stump: dict, x: np.ndarray) -> int:
    raw = 1 if x[stump["featureIndex"]] > stump["threshold"] else -1
    return stump["polarity"] * raw


def adaboost_fit(X: np.ndarray, targets: np.ndarray, rounds: int) -> list[dict]:
    n = X.shape[0]
    weights = np.full(n, 1.0 / n)
    trace = []
    for _ in range(rounds):
        round_weights = weights.copy()
        stump = fit_stump(X, targets, round_weights)
        epsilon = stump["weightedError"]
        clamped = min(max(epsilon, EPSILON_FLOOR), 1.0 - EPSILON_FLOOR)
        alpha = float(np.log((1.0 - clamped) / clamped))
        trace.append(
            {
                "weights": round_weights.tolist(),
                "stump": {
                    "featureIndex": stump["featureIndex"],
                    "threshold": stump["threshold"],
                    "polarity": stump["polarity"],
                },
                "epsilon": epsilon,
                "alpha": alpha,
            }
        )
        predictions = np.array([stump_predict(stump, X[i]) for i in range(n)])
        misclassified = predictions != targets
        weights = np.where(misclassified, weights * np.exp(alpha), weights)
    return trace


def function_value(trace: list[dict], x: np.ndarray, up_to_round: int) -> float:
    total = 0.0
    for round_info in trace[:up_to_round]:
        total += round_info["alpha"] * stump_predict(round_info["stump"], x)
    return 0.5 * total


def predict(trace: list[dict], x: np.ndarray, up_to_round: int) -> int:
    return 1 if function_value(trace, x, up_to_round) >= 0 else -1


# Fixed-seed toy dataset: two noisy blobs around (-1, -1) -> target -1 and (1, 1) -> target
# +1, with a handful of overlap points so no single stump gets epsilon = 0 and every round
# does real work re-weighting.
rng = np.random.default_rng(1729)
N_PER_CLASS = 12
neg = rng.normal(loc=(-1.0, -1.0), scale=0.9, size=(N_PER_CLASS, 2))
pos = rng.normal(loc=(1.0, 1.0), scale=0.9, size=(N_PER_CLASS, 2))
X = np.vstack([neg, pos])
targets = np.array([-1] * N_PER_CLASS + [1] * N_PER_CLASS)

M = 6
trace = adaboost_fit(X, targets, M)

cases = [
    {
        "fn": "adaBoostRounds",
        "X": X.tolist(),
        "targets": targets.tolist(),
        "rounds": M,
        "expected": trace,
    }
]

test_points = [
    (0.0, 0.0),
    (-1.5, -1.5),
    (1.5, 1.5),
    (2.0, -2.0),
    (-0.3, 0.4),
    (0.8, -0.6),
]

for x in test_points:
    xa = np.array(x)
    for up_to in (None, 1, 3, M):
        limit = M if up_to is None else up_to
        cases.append(
            {
                "fn": "adaBoostFunctionValue",
                "x": list(x),
                "upToRound": up_to,
                "expected": function_value(trace, xa, limit),
            }
        )
        cases.append(
            {
                "fn": "adaBoostPredict",
                "x": list(x),
                "upToRound": up_to,
                "expected": predict(trace, xa, limit),
            }
        )

payload = {
    "dataset": {"X": X.tolist(), "targets": targets.tolist()},
    "rounds": M,
    "cases": cases,
}

out = FIXTURES / "adaboost.json"
out.write_text(json.dumps(payload))
print(f"wrote {out} ({len(cases)} cases)")
