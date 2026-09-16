"""Golden fixtures for packages/math/src/ensemble/weightedSoftmax.ts.

`weightedSoftmaxFit` generalises `weightedLogisticFit` from two classes to K: a per-point
weighted multiclass cross-entropy fit, PRML 4.104-4.109 with the one-hot target replaced
by an arbitrary distribution over classes (any row of `targets` summing to 1, entries in
[0, 1]). This is the primitive the mixture of experts' gating network (14.5.3) needs for
K > 2 experts, generalising 4.106's soft-target cross-entropy fit the same way
`weightedLogisticFit` already does at K = 2.

Identifiability: the K-class softmax is invariant to adding any constant vector to every
row of the weight matrix (softmax(a) = softmax(a + c) for scalar-per-feature c), so the
naive K*D-parameter Hessian is exactly singular, not merely ill-conditioned, at every
point, not only at convergence. The fix used here and in the TypeScript is the standard
one: fix class 0's weights to the zero vector (a "reference class") and solve only for
the remaining (K-1)*D parameters. This removes the redundancy structurally rather than
papering over it with a ridge, and gives a unique optimum both implementations can be
checked against bit-for-bit (to within Newton-Raphson's own convergence tolerance).

A small ridge (1e-10) is still added to the reduced (K-1)*D Hessian at every iteration,
for the same reason `weightedLogistic.ts`'s CURVATURE_FLOOR exists: a gate that has
learned a confident, sharp partition drives some responsibilities to exactly 0 or 1,
which can make even the reduced Hessian singular right where the fit is most confident.
Both implementations apply the identical ridge at every iteration, so they are compared
against the same regularised objective rather than two different ones that happen to
agree near a shared unregularised optimum.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import softmax as scipy_softmax

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

RIDGE = 1e-10
rng = np.random.default_rng(20260930)


def newton_raphson_softmax(design, targets, point_weights, tol=1e-13, max_iter=200):
    n, d = design.shape
    k = targets.shape[1]
    free = k - 1  # class 0 fixed at zero
    w = np.zeros((free, d))

    def full_weights(w_free):
        return np.vstack([np.zeros((1, d)), w_free])

    def probabilities(w_free):
        activations = design @ full_weights(w_free).T  # N x K
        return scipy_softmax(activations, axis=1)

    for _ in range(max_iter):
        y = probabilities(w)  # N x K
        gradient = np.zeros(free * d)
        hessian = np.zeros((free * d, free * d))
        for kk in range(1, k):
            fk = kk - 1
            gk = point_weights * (y[:, kk] - targets[:, kk])
            gradient[fk * d : (fk + 1) * d] = design.T @ gk
        for kk in range(1, k):
            fk = kk - 1
            for ll in range(1, k):
                fl = ll - 1
                indicator = 1.0 if kk == ll else 0.0
                curvature = point_weights * y[:, kk] * (indicator - y[:, ll])
                block = (design * curvature[:, None]).T @ design
                hessian[fk * d : (fk + 1) * d, fl * d : (fl + 1) * d] = block
        hessian = hessian + RIDGE * np.eye(free * d)
        step = np.linalg.solve(hessian, gradient)
        w_next = w - step.reshape(free, d)
        delta = np.sqrt(np.sum((w_next - w) ** 2))
        w = w_next
        if delta <= tol:
            break
    return full_weights(w)


def make_dataset(n, d, k, seed_stream):
    r = np.random.default_rng(seed_stream)
    x1 = r.uniform(-2.0, 2.0, n)
    x2 = r.uniform(-2.0, 2.0, n)
    design = np.stack([np.ones(n), x1, x2], axis=1)
    true_w = r.normal(0.0, 1.2, size=(k - 1, d))
    true_full = np.vstack([np.zeros((1, d)), true_w])
    probs = scipy_softmax(design @ true_full.T, axis=1)
    return design, probs


N, D, K = 80, 3, 4
DESIGN, TRUE_PROBS = make_dataset(N, D, K, 1)

cases = []

# Hard one-hot labels sampled from the true probabilities, uniform point weights:
# recovers plain multiclass logistic regression.
labels = np.array([rng.choice(K, p=TRUE_PROBS[i]) for i in range(N)])
one_hot = np.eye(K)[labels]
uniform_w = np.ones(N)
expected_hard = newton_raphson_softmax(DESIGN, one_hot, uniform_w)
cases.append(
    {
        "fn": "weightedSoftmaxFit",
        "design": DESIGN.tolist(),
        "targets": one_hot.tolist(),
        "pointWeights": uniform_w.tolist(),
        "expected": expected_hard.tolist(),
    }
)

# Soft targets (the true probabilities themselves, standing in for EM responsibilities),
# uniform point weights: the mixture-of-experts gate's own M-step shape.
expected_soft = newton_raphson_softmax(DESIGN, TRUE_PROBS, uniform_w)
cases.append(
    {
        "fn": "weightedSoftmaxFit",
        "design": DESIGN.tolist(),
        "targets": TRUE_PROBS.tolist(),
        "pointWeights": uniform_w.tolist(),
        "expected": expected_soft.tolist(),
    }
)

# Non-uniform point weights with hard labels, on a second independent dataset.
DESIGN2, TRUE_PROBS2 = make_dataset(N, D, K, 2)
labels2 = np.array([rng.choice(K, p=TRUE_PROBS2[i]) for i in range(N)])
one_hot2 = np.eye(K)[labels2]
skew_w = np.concatenate([np.full(N // 2, 1.0), np.full(N - N // 2, 0.15)])
expected_weighted = newton_raphson_softmax(DESIGN2, one_hot2, skew_w)
cases.append(
    {
        "fn": "weightedSoftmaxFit",
        "design": DESIGN2.tolist(),
        "targets": one_hot2.tolist(),
        "pointWeights": skew_w.tolist(),
        "expected": expected_weighted.tolist(),
    }
)

# K = 2 must still work: this is the exact function mixtureExperts.ts now calls for a
# two-expert gate, and it must agree with the dedicated binary weightedLogisticFit's own
# fixture in spirit (same problem, different parameterisation: weightedLogisticFit fits a
# single vector for p(class 1), weightedSoftmaxFit fits a zero reference class 0 plus one
# free vector for class 1 - so sigmoid(v.x) here must equal softmax([0, w1.x])[1], i.e.
# w1 = v exactly, since sigmoid(z) = softmax([0, z])[1] identically).
DESIGN3, _ = make_dataset(50, D, 2, 3)
true_v = np.array([0.4, -0.8, 1.1])
p1 = 1.0 / (1.0 + np.exp(-(DESIGN3 @ true_v)))
labels3 = (rng.uniform(0.0, 1.0, 50) < p1).astype(int)
one_hot3 = np.eye(2)[labels3]
expected_binary = newton_raphson_softmax(DESIGN3, one_hot3, np.ones(50))
cases.append(
    {
        "fn": "weightedSoftmaxFit",
        "design": DESIGN3.tolist(),
        "targets": one_hot3.tolist(),
        "pointWeights": np.ones(50).tolist(),
        "expected": expected_binary.tolist(),
    }
)

fixture = {"cases": cases}
(FIXTURES / "weightedSoftmax.json").write_text(json.dumps(fixture, indent=2))
print(f"wrote {len(cases)} cases to weightedSoftmax.json")
