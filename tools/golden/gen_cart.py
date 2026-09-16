"""Golden fixtures for packages/math/src/ensemble/cart.ts (PRML 14.4, tree-based models).

Independent numpy/pure-python reference for CART regression and classification trees,
the 14.32-14.34 impurity measures, and cost-complexity pruning for both task types. None
of this calls into the TypeScript; it re-derives the same greedy exhaustive splitter from
scratch so the fixture is a genuine external check, not a restatement of the implementation.

Node statistics are task-shaped, not borrowed across tasks: a regression node's stat is
its region's `sumT`/`sumTSq` (the SSE sufficient statistic, PRML 14.29-14.31), and a
classification node's stat is its region's per-class `counts` (PRML 14.32-14.33's `p_tau_k`
is `counts[k] / nSamples`). A regression leaf and a classification leaf are not the same
shape of thing, and forcing them to share one representation - the previous revision
summed raw class-index integers into a `sumT`/`sumTSq` pair with no statistical meaning,
just so classification nodes had *something* to put in those two fields - is what this
fixture and cart.ts now both avoid.

Tie-break rule (must match cart.ts:bestSplit bit-for-bit): for a node's candidate search,
iterate feature index ascending; for each feature, iterate thresholds ascending, where a
threshold is the midpoint between two consecutive DISTINCT sorted values of that feature
(mirrors ensemble/stumps.ts's fitDecisionStump); keep the running best split only on a
STRICT improvement (never `<=`), so the first candidate encountered in this fixed order
wins any exact tie - lower feature index first, then lower threshold. Two independent
implementations of this exact rule are forced to agree on ties by construction.

Sign correction to the extracted 14.32: our PDF-to-text extraction pipeline drops
superscripts and, here, a leading minus sign (the same kind of artefact hit chapter 13's
extracted 13.65, where `c_n^{-1}` lost its exponent on the same line that mangled a
subscript - the book's printing is fine in both places, only our extract is lossy). The
extracted equation text reads `Q(T) = sum_k p_k ln p_k` with no leading minus sign, but
the very next sentence says this quantity "vanishes for p=0 and p=1 and has a maximum at
p=0.5" - `sum p ln p` is <=0 and is MINIMISED (most negative) at p=0.5 (two classes at 0.5
gives ln(0.5)=-0.693, below the boundary value 0 at p in {0,1}), so the extracted sign
contradicts the book's own stated property in the very next sentence. That sentence is
what pins the sign: the standard Shannon-entropy form with the leading minus is the one
that actually vanishes at {0,1} and peaks at 0.5, so that is what this reference (and
cart.ts) implements: `crossEntropyImpurity(p) = -sum_k p_k ln(p_k)`.

Classification pruning uses the misclassification COUNT as its leaf cost (book: "for
subsequent pruning of the tree, the misclassification rate is generally used"), scaled by
nSamples to stay commensurate with a per-leaf integer penalty the same way regression's
SSE is a total, not a mean: `n * misclassificationImpurity(proportions)`.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def region_sse(sum_t: float, sum_t_sq: float, n: int) -> float:
    return sum_t_sq - sum_t**2 / n


# Search and tie-break order here must match cart.ts:bestSplit bit-for-bit; see the
# module docstring for the exact rule.
def best_split(X, idx, min_leaf, score_fn):
    n_features = len(X[0])
    best = None
    best_score = float("inf")
    for feature in range(n_features):
        values = sorted(set(X[i][feature] for i in idx))
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            left = [i2 for i2 in idx if X[i2][feature] <= threshold]
            right = [i2 for i2 in idx if X[i2][feature] > threshold]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            score = score_fn(left, right)
            if score < best_score:
                best_score = score
                best = (feature, threshold)
    return best


def build_regression_node(X, t, idx, depth, min_leaf, max_depth):
    n = len(idx)
    sum_t = float(sum(t[i] for i in idx))
    sum_t_sq = float(sum(t[i] ** 2 for i in idx))
    stat = {"sumT": sum_t, "sumTSq": sum_t_sq}

    def leaf():
        return {"kind": "leaf", "value": sum_t / n, "nSamples": n, "stat": stat}

    if n < 2 * min_leaf or depth >= max_depth:
        return leaf()

    def score(left, right):
        sl = sum(t[i] for i in left)
        sql = sum(t[i] ** 2 for i in left)
        sr = sum(t[i] for i in right)
        sqr = sum(t[i] ** 2 for i in right)
        return region_sse(sl, sql, len(left)) + region_sse(sr, sqr, len(right))

    split = best_split(X, idx, min_leaf, score)
    if split is None:
        return leaf()
    feature, threshold = split
    left_idx = [i for i in idx if X[i][feature] <= threshold]
    right_idx = [i for i in idx if X[i][feature] > threshold]
    left = build_regression_node(X, t, left_idx, depth + 1, min_leaf, max_depth)
    right = build_regression_node(X, t, right_idx, depth + 1, min_leaf, max_depth)
    return {
        "kind": "split",
        "featureIndex": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "nSamples": n,
        "stat": stat,
    }


def predict_regression(node, x):
    while node["kind"] == "split":
        node = (
            node["left"]
            if x[node["featureIndex"]] <= node["threshold"]
            else node["right"]
        )
    return node["value"]


def prune_regression(node, lam):
    if node["kind"] == "leaf":
        sse = region_sse(node["stat"]["sumT"], node["stat"]["sumTSq"], node["nSamples"])
        return node, sse, 1
    left, left_cost, left_leaves = prune_regression(node["left"], lam)
    right, right_cost, right_leaves = prune_regression(node["right"], lam)
    keep_cost = left_cost + right_cost + lam * (left_leaves + right_leaves)
    collapsed_sse = region_sse(
        node["stat"]["sumT"], node["stat"]["sumTSq"], node["nSamples"]
    )
    collapse_cost = collapsed_sse + lam
    if collapse_cost <= keep_cost:
        leaf = {
            "kind": "leaf",
            "value": node["stat"]["sumT"] / node["nSamples"],
            "nSamples": node["nSamples"],
            "stat": node["stat"],
        }
        return leaf, collapsed_sse, 1
    rebuilt = dict(node)
    rebuilt["left"] = left
    rebuilt["right"] = right
    return rebuilt, left_cost + right_cost, left_leaves + right_leaves


def leaf_count(node):
    return (
        1
        if node["kind"] == "leaf"
        else leaf_count(node["left"]) + leaf_count(node["right"])
    )


def gini(p):
    return float(sum(pk * (1 - pk) for pk in p))


def cross_entropy(p):
    return float(-sum(pk * np.log(pk) for pk in p if pk > 0))


def misclassification(p):
    return float(1 - max(p))


def class_proportions(labels, idx, num_classes):
    counts = [0] * num_classes
    for i in idx:
        counts[labels[i]] += 1
    return [c / len(idx) for c in counts]


def class_counts(labels, idx, num_classes):
    counts = [0] * num_classes
    for i in idx:
        counts[labels[i]] += 1
    return counts


def majority_class(labels, idx, num_classes):
    counts = class_counts(labels, idx, num_classes)
    best = 0
    for k in range(1, num_classes):
        if counts[k] > counts[best]:
            best = k
    return best


def build_classification_node(
    X, labels, num_classes, impurity_fn, idx, depth, min_leaf, max_depth
):
    n = len(idx)
    value = majority_class(labels, idx, num_classes)
    counts = class_counts(labels, idx, num_classes)
    stat = {"counts": counts}

    def leaf():
        return {"kind": "leaf", "value": value, "nSamples": n, "stat": stat}

    if n < 2 * min_leaf or depth >= max_depth:
        return leaf()

    def score(left, right):
        pl = class_proportions(labels, left, num_classes)
        pr = class_proportions(labels, right, num_classes)
        return (len(left) / n) * impurity_fn(pl) + (len(right) / n) * impurity_fn(pr)

    split = best_split(X, idx, min_leaf, score)
    if split is None:
        return leaf()
    feature, threshold = split
    left_idx = [i for i in idx if X[i][feature] <= threshold]
    right_idx = [i for i in idx if X[i][feature] > threshold]
    left = build_classification_node(
        X, labels, num_classes, impurity_fn, left_idx, depth + 1, min_leaf, max_depth
    )
    right = build_classification_node(
        X, labels, num_classes, impurity_fn, right_idx, depth + 1, min_leaf, max_depth
    )
    return {
        "kind": "split",
        "featureIndex": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "nSamples": n,
        "stat": stat,
    }


def predict_classification(node, x):
    while node["kind"] == "split":
        node = (
            node["left"]
            if x[node["featureIndex"]] <= node["threshold"]
            else node["right"]
        )
    return node["value"]


def prune_classification(node, lam):
    if node["kind"] == "leaf":
        n = node["nSamples"]
        proportions = [c / n for c in node["stat"]["counts"]]
        cost = n * misclassification(proportions)
        return node, cost, 1
    left, left_cost, left_leaves = prune_classification(node["left"], lam)
    right, right_cost, right_leaves = prune_classification(node["right"], lam)
    keep_cost = left_cost + right_cost + lam * (left_leaves + right_leaves)
    n = node["nSamples"]
    proportions = [c / n for c in node["stat"]["counts"]]
    collapsed_cost = n * misclassification(proportions)
    collapse_total = collapsed_cost + lam
    if collapse_total <= keep_cost:
        leaf = {
            "kind": "leaf",
            "value": majority_class_from_counts(node["stat"]["counts"]),
            "nSamples": n,
            "stat": node["stat"],
        }
        return leaf, collapsed_cost, 1
    rebuilt = dict(node)
    rebuilt["left"] = left
    rebuilt["right"] = right
    return rebuilt, left_cost + right_cost, left_leaves + right_leaves


def majority_class_from_counts(counts):
    best = 0
    for k in range(1, len(counts)):
        if counts[k] > counts[best]:
            best = k
    return best


rng = np.random.default_rng(20260917)

cases = []

proportion_vectors = [
    [0.5, 0.5],
    [1.0, 0.0, 0.0],
    [0.2, 0.3, 0.5],
    [1.0 / 3, 1.0 / 3, 1.0 / 3],
    [0.0, 1.0],
]
for p in proportion_vectors:
    cases.append({"fn": "giniImpurity", "proportions": p, "expected": gini(p)})
    cases.append(
        {"fn": "crossEntropyImpurity", "proportions": p, "expected": cross_entropy(p)}
    )
    cases.append(
        {
            "fn": "misclassificationImpurity",
            "proportions": p,
            "expected": misclassification(p),
        }
    )

N1 = 24
X0_1 = np.linspace(0.0, 1.0, N1)
X1_1 = rng.uniform(0.0, 1.0, N1)
T1 = np.where(X0_1 < 0.5, 2.0, 9.0) + rng.normal(0.0, 0.15, N1)
X_1 = [[float(X0_1[i]), float(X1_1[i])] for i in range(N1)]
t_1 = [float(v) for v in T1]
tree_1 = build_regression_node(X_1, t_1, list(range(N1)), 0, 2, 2)
query_points_1 = [[0.0, 0.5], [0.3, 0.5], [0.6, 0.5], [1.0, 0.5]]
cases.append(
    {
        "fn": "regressionSplit",
        "name": "unambiguous",
        "X": X_1,
        "t": t_1,
        "options": {"minLeafSize": 2, "maxDepth": 2},
        "expectedTree": tree_1,
        "predictions": [
            {"x": q, "expected": predict_regression(tree_1, q)} for q in query_points_1
        ],
    }
)

# feature1 is an exact copy of feature0, so every threshold on feature1 ties feature0's
# score exactly; the tie-break rule above must resolve this to feature 0.
N2 = 10
X0_2 = np.linspace(-1.0, 1.0, N2)
X1_2 = X0_2.copy()
T2 = np.where(X0_2 < 0.0, -3.0, 3.0)
X_2 = [[float(X0_2[i]), float(X1_2[i])] for i in range(N2)]
t_2 = [float(v) for v in T2]
tree_2 = build_regression_node(X_2, t_2, list(range(N2)), 0, 1, 1)
cases.append(
    {
        "fn": "regressionSplit",
        "name": "tieBreak",
        "X": X_2,
        "t": t_2,
        "options": {"minLeafSize": 1, "maxDepth": 1},
        "expectedTree": tree_2,
        "predictions": [],
    }
)

N3 = 27
X0_3 = np.sort(
    np.concatenate(
        [rng.uniform(0.0, 0.3, 9), rng.uniform(0.3, 0.7, 9), rng.uniform(0.7, 1.0, 9)]
    )
)
labels_3 = [0 if x < 0.3 else (1 if x < 0.7 else 2) for x in X0_3]
X1_3 = rng.uniform(0.0, 1.0, N3)
X_3 = [[float(X0_3[i]), float(X1_3[i])] for i in range(N3)]
query_points_3 = [[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]]
for impurity_name, impurity_fn in (("gini", gini), ("crossEntropy", cross_entropy)):
    tree_3 = build_classification_node(
        X_3, labels_3, 3, impurity_fn, list(range(N3)), 0, 2, 2
    )
    cases.append(
        {
            "fn": "classificationSplit",
            "name": impurity_name,
            "X": X_3,
            "labels": labels_3,
            "numClasses": 3,
            "impurity": impurity_name,
            "options": {"minLeafSize": 2, "maxDepth": 2},
            "expectedTree": tree_3,
            "predictions": [
                {"x": q, "expected": predict_classification(tree_3, q)}
                for q in query_points_3
            ],
        }
    )

# Lambdas were picked by sweeping this exact recursion over the exact tree below (not
# guessed), so that lambda=1e-6 prunes nothing, lambda=25 collapses exactly the root's
# left subtree to one leaf (4 leaves -> 3), and lambda=300 collapses everything to the
# root (1 leaf) - giving one identifiable "which node was pruned" case in the middle.
N4 = 32
X0_4 = np.sort(rng.uniform(0.0, 1.0, N4))
X1_4 = rng.uniform(0.0, 1.0, N4)
T4 = 10.0 * X0_4**2 + rng.normal(0.0, 0.05, N4)
X_4 = [[float(X0_4[i]), float(X1_4[i])] for i in range(N4)]
t_4 = [float(v) for v in T4]
tree_4 = build_regression_node(X_4, t_4, list(range(N4)), 0, 2, 2)
assert leaf_count(tree_4) == 4, f"expected a 4-leaf base tree, got {leaf_count(tree_4)}"

prune_lambdas = [1e-6, 25.0, 300.0]
pruned_cases = []
for lam in prune_lambdas:
    pruned, cost, leaves = prune_regression(tree_4, lam)
    pruned_cases.append(
        {"lambda": lam, "expectedTree": pruned, "expectedLeafCount": leaves}
    )

assert pruned_cases[0]["expectedLeafCount"] == 4
assert pruned_cases[1]["expectedLeafCount"] == 3
assert pruned_cases[1]["expectedTree"]["left"]["kind"] == "leaf"
assert pruned_cases[1]["expectedTree"]["right"]["kind"] == "split"
assert pruned_cases[2]["expectedLeafCount"] == 1

cases.append(
    {
        "fn": "regressionPrune",
        "X": X_4,
        "t": t_4,
        "options": {"minLeafSize": 2, "maxDepth": 2},
        "baseTree": tree_4,
        "pruneCases": pruned_cases,
    }
)

# Classification pruning: four well-separated Gaussian blobs, one class each. Growing
# uses Gini, which keeps finding small, genuine impurity reductions deep in the tree even
# once every region is already dominated by one class (Gini is sensitive to proportions
# in a way misclassification rate is not, PRML's own stated reason for growing on Gini
# rather than misclassification rate); this grows a much bigger tree than the four true
# regions need. Pruning switches to misclassification COUNT as its cost, and because
# most of that deeper structure contributes literally zero reduction in misclassification
# count, cost-complexity pruning collapses it back to exactly the four true regions at a
# lambda arbitrarily close to zero, not just at some moderate value - the mismatch between
# the growing and pruning criteria is what makes that collapse immediate rather than
# gradual. Unlike the regression case above there is no third, partially-collapsed tier
# between "the four true regions" and "one root leaf": with well-separated clusters,
# misclassification cost sees nothing worth keeping in between. Both leaf counts below
# were read off this exact recursion, not guessed.
N5 = 60
centers = [(-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.6, 1.6)]
labels_5 = []
X_5 = []
per_class = N5 // 4
for k, (cx, cy) in enumerate(centers):
    for _ in range(per_class):
        X_5.append([cx + rng.normal(0.0, 0.35), cy + rng.normal(0.0, 0.35)])
        labels_5.append(k)
tree_5 = build_classification_node(X_5, labels_5, 4, gini, list(range(N5)), 0, 2, 6)
base_leaves_5 = leaf_count(tree_5)
assert base_leaves_5 > 4, (
    f"expected Gini growing to overshoot the 4 true regions, got {base_leaves_5} leaves"
)

class_prune_lambdas = [1e-9, 10.0, 20.0]
class_pruned_cases = []
for lam in class_prune_lambdas:
    pruned, cost, leaves = prune_classification(tree_5, lam)
    class_pruned_cases.append(
        {"lambda": lam, "expectedTree": pruned, "expectedLeafCount": leaves}
    )

assert class_pruned_cases[0]["expectedLeafCount"] == 4
assert class_pruned_cases[1]["expectedLeafCount"] == 4
assert class_pruned_cases[2]["expectedLeafCount"] == 1

cases.append(
    {
        "fn": "classificationPrune",
        "X": X_5,
        "labels": labels_5,
        "numClasses": 4,
        "options": {"minLeafSize": 2, "maxDepth": 6},
        "baseTree": tree_5,
        "pruneCases": class_pruned_cases,
    }
)

fixture = {"cases": cases}
(FIXTURES / "cart.json").write_text(json.dumps(fixture, indent=2))
print(f"wrote {len(cases)} cases to cart.json")
