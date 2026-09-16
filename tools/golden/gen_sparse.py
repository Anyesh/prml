"""Golden fixtures for packages/math/src/sparse/.

Follows the shape documented in tools/golden/README.md: {"cases": [...]}, each case
carrying an "fn" discriminant, the inputs, and an "expected" value at full float64
precision.

Equation references: maximum margin dual 7.10-7.18, soft margin 7.20-7.37, hinge and
comparison losses 7.44-7.51 (Figure 7.5), epsilon-insensitive SVM regression 7.50-7.69,
RVM regression 7.76-7.91, sparsity analysis 7.92-7.107, RVM classification 7.108-7.119.

SMO's own working-set heuristic is never replicated here: for both the classification and
regression duals, an approximate solution comes from a general-purpose scipy QP solve,
which is then rounded to the nearest active set (a variable pinned at 0, pinned at its
upper bound, or free) and polished by solving the *linear* KKT stationarity system for
that active set exactly. That system is what a correct SMO fixed point must also satisfy,
so agreement with it is agreement with the true optimum, not with any particular algorithm
for finding it. Each case asserts its own KKT residual is below 1e-10 before being written
out, so a fixture that fails that check never reaches the file.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

# Kernels (7.42, 6.23) and the margin-based losses (Figure 7.5, 7.51).

kernel_pairs = [
    ([1.0, 2.0], [3.0, -1.0]),
    ([0.0, 0.0], [1.0, 1.0]),
    ([2.0, 2.0, 2.0], [1.0, 0.0, -1.0]),
    ([-1.5, 0.5], [-1.5, 0.5]),
]
for a, b in kernel_pairs:
    a_arr, b_arr = np.array(a), np.array(b)
    cases.append(
        {"fn": "linearKernel", "a": a, "b": b, "expected": float(a_arr @ b_arr)}
    )

poly_cases = [
    ([1.0, 2.0], [3.0, -1.0], 2, 1.0),
    ([0.5, -0.5], [2.0, 2.0], 3, 1.0),
    ([1.0, 0.0], [0.0, 1.0], 2, 0.0),
]
for a, b, degree, offset in poly_cases:
    a_arr, b_arr = np.array(a), np.array(b)
    expected = float((offset + a_arr @ b_arr) ** degree)
    cases.append(
        {
            "fn": "polynomialKernel",
            "a": a,
            "b": b,
            "degree": degree,
            "offset": offset,
            "expected": expected,
        }
    )

rbf_cases = [
    ([1.0, 2.0], [3.0, -1.0], 0.5),
    ([0.0, 0.0], [1.0, 1.0], 1.0),
    ([2.0, 2.0, 2.0], [1.0, 0.0, -1.0], 0.1),
]
for a, b, gamma in rbf_cases:
    a_arr, b_arr = np.array(a), np.array(b)
    expected = float(np.exp(-gamma * np.sum((a_arr - b_arr) ** 2)))
    cases.append(
        {"fn": "rbfKernelGamma", "a": a, "b": b, "gamma": gamma, "expected": expected}
    )

margins = [-2.0, -1.0, -0.5, 0.0, 0.3, 1.0, 2.0]
cases.append(
    {
        "fn": "hingeLoss",
        "margins": margins,
        "expected": [float(max(0.0, 1.0 - z)) for z in margins],
    }
)
cases.append(
    {
        "fn": "logisticMarginLoss",
        "margins": margins,
        "expected": [float(np.log1p(np.exp(-z)) / np.log(2.0)) for z in margins],
    }
)
cases.append(
    {
        "fn": "squaredMarginLoss",
        "margins": margins,
        "expected": [float((1.0 - z) ** 2) for z in margins],
    }
)
cases.append(
    {
        "fn": "misclassificationLoss",
        "margins": margins,
        "expected": [0.0 if z > 0 else 1.0 for z in margins],
    }
)

eps_cases = [(0.0, 0.3), (0.2, 0.3), (-0.2, 0.3), (0.5, 0.3), (-0.5, 0.3), (1.2, 0.1)]
cases.append(
    {
        "fn": "epsilonInsensitiveLoss",
        "residuals": [r for r, _ in eps_cases],
        "epsilons": [e for _, e in eps_cases],
        "expected": [float(max(0.0, abs(r) - e)) for r, e in eps_cases],
    }
)

# SMO for classification (7.10-7.18, 7.32-7.37): an independent dual QP solve, polished
# to the exact KKT stationarity point for its rounded active set.


def solve_classification_dual_approx(K, y, C):
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K
    ones = np.ones(n)

    def obj(a):
        return 0.5 * a @ Q @ a - ones @ a

    def grad(a):
        return Q @ a - ones

    cons = [LinearConstraint(y.reshape(1, -1), 0.0, 0.0)]
    bounds = Bounds(0.0, C)
    res = minimize(
        obj,
        np.zeros(n),
        jac=grad,
        hess=lambda a: Q,
        bounds=bounds,
        constraints=cons,
        method="trust-constr",
        options={"gtol": 1e-14, "xtol": 1e-14, "maxiter": 20000},
    )
    return res.x


# Exact KKT solve given a rounded active set: a linear system, independent of SMO.
def polish_classification_active_set(K, y, C, alpha_approx, tol=1e-6):
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K
    at_zero = alpha_approx < tol
    at_c = alpha_approx > C - tol
    free = ~at_zero & ~at_c

    alpha = np.where(at_c, C, 0.0)
    free_idx = np.where(free)[0]
    m = len(free_idx)
    if m > 0:
        # Free rows satisfy stationarity (Qa)_i - 1 + y_i*b = 0, with b the equality
        # constraint's Lagrange multiplier, which is exactly the SVM bias (7.18): at a free
        # support vector, y_i*(sum_j a_j y_j K_ij + b) = 1 expands to (Qa)_i + y_i*b = 1.
        A = np.zeros((m + 1, m + 1))
        rhs = np.zeros(m + 1)
        bound_contribution = (
            Q[np.ix_(free_idx, np.where(at_c)[0])] @ np.full(at_c.sum(), C)
            if at_c.any()
            else 0.0
        )
        A[:m, :m] = Q[np.ix_(free_idx, free_idx)]
        A[:m, m] = y[free_idx]
        rhs[:m] = 1.0 - bound_contribution
        A[m, :m] = y[free_idx]
        rhs[m] = -(y[at_c] @ np.full(at_c.sum(), C) if at_c.any() else 0.0)
        sol = np.linalg.solve(A, rhs)
        alpha[free_idx] = sol[:m]
        b = sol[m]
    else:
        b = 0.0

    grad = Q @ alpha - 1.0
    kkt = np.max(
        np.abs(
            np.concatenate(
                [
                    (grad[free] - (-y[free] * b)) if m > 0 else np.array([0.0]),
                    [y @ alpha],
                ]
            )
        )
    )
    return alpha, b, kkt


def classification_case(name, points, labels, C):
    points = np.array(points)
    labels = np.array(labels, dtype=float)
    K = points @ points.T
    approx = solve_classification_dual_approx(K, labels, C)
    alpha, b, kkt = polish_classification_active_set(K, labels, C, approx)
    assert kkt < 1e-10, f"{name}: KKT residual {kkt} too large"

    probe = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0], [1.0, 0.0], [-1.0, 0.5]])
    decision = (alpha * labels) @ (points @ probe.T) + b

    cases.append(
        {
            "fn": "smoFitClassifier",
            "name": name,
            "points": points.tolist(),
            "labels": labels.tolist(),
            "C": C,
            "probe": probe.tolist(),
            "expected": {
                "alpha": alpha.tolist(),
                "bias": float(b),
                "decision": decision.tolist(),
            },
        }
    )


SEP_POS = [[3.0, 2.0], [4.0, 3.5], [3.5, 0.8], [5.0, 2.5], [2.5, 3.2]]
SEP_NEG = [[-1.0, -2.0], [-2.2, -3.0], [-0.3, -3.5], [-2.8, -1.2], [-3.5, -0.4]]
classification_case("separable", SEP_POS + SEP_NEG, [1.0] * 5 + [-1.0] * 5, C=10.0)

OVERLAP_POS = [[1.0, 1.2], [2.1, 1.8], [1.6, -0.3], [2.8, 1.1], [0.6, -1.4]]
OVERLAP_NEG = [[-0.8, -1.1], [0.3, 0.2], [1.1, -0.2], [-1.9, 0.4], [-0.6, 1.3]]
classification_case(
    "overlapping", OVERLAP_POS + OVERLAP_NEG, [1.0] * 5 + [-1.0] * 5, C=1.0
)

# SMO for regression (7.50-7.69), folded into the same 2N signed-index shape smo.ts uses.


def solve_regression_dual_approx(K, t, C, epsilon):
    n = len(t)
    y = np.concatenate([np.ones(n), -np.ones(n)])
    p = np.concatenate([t - epsilon, -t - epsilon])
    Q = np.block([[K, -K], [-K, K]])

    def obj(v):
        return 0.5 * v @ Q @ v - p @ v

    def grad(v):
        return Q @ v - p

    cons = [LinearConstraint(y.reshape(1, -1), 0.0, 0.0)]
    bounds = Bounds(0.0, C)
    res = minimize(
        obj,
        np.zeros(2 * n),
        jac=grad,
        hess=lambda v: Q,
        bounds=bounds,
        constraints=cons,
        method="trust-constr",
        options={"gtol": 1e-14, "xtol": 1e-14, "maxiter": 20000},
    )
    return res.x, Q, y, p


def polish_regression_active_set(Q, y, p, C, v_approx, tol=1e-6):
    m = len(y)
    at_zero = v_approx < tol
    at_c = v_approx > C - tol
    free = ~at_zero & ~at_c

    v = np.where(at_c, C, 0.0)
    free_idx = np.where(free)[0]
    k = len(free_idx)
    if k > 0:
        A = np.zeros((k + 1, k + 1))
        rhs = np.zeros(k + 1)
        bound_idx = np.where(at_c)[0]
        bound_contribution = (
            Q[np.ix_(free_idx, bound_idx)] @ np.full(len(bound_idx), C)
            if bound_idx.size
            else 0.0
        )
        A[:k, :k] = Q[np.ix_(free_idx, free_idx)]
        A[:k, k] = y[free_idx]
        rhs[:k] = p[free_idx] - bound_contribution
        A[k, :k] = y[free_idx]
        rhs[k] = -(y[bound_idx] @ np.full(len(bound_idx), C) if bound_idx.size else 0.0)
        sol = np.linalg.solve(A, rhs)
        v[free_idx] = sol[:k]
        mu = sol[k]
    else:
        mu = 0.0

    grad = Q @ v - p
    kkt = np.max(
        np.abs(
            np.concatenate(
                [(grad[free] + y[free] * mu) if k > 0 else np.array([0.0]), [y @ v]]
            )
        )
    )
    return v, mu, kkt


def regression_case(name, xs, ts, C, epsilon):
    xs = np.array(xs).reshape(-1, 1)
    ts = np.array(ts)
    n = len(ts)
    K = xs @ xs.T
    approx, Q, y, p = solve_regression_dual_approx(K, ts, C, epsilon)
    v, b, kkt = polish_regression_active_set(Q, y, p, C, approx)
    assert kkt < 1e-10, f"{name}: KKT residual {kkt} too large"

    a = v[:n]
    ahat = v[n:]
    coefficients = a - ahat

    probe = np.array([-2.5, -1.0, 0.0, 1.0, 2.5]).reshape(-1, 1)
    predictions = (coefficients @ (xs @ probe.T)) + b

    cases.append(
        {
            "fn": "smoFitRegression",
            "name": name,
            "points": xs.tolist(),
            "targets": ts.tolist(),
            "C": C,
            "epsilon": epsilon,
            "probe": probe.tolist(),
            "expected": {
                "coefficients": coefficients.tolist(),
                "bias": float(b),
                "predictions": predictions.tolist(),
            },
        }
    )


svr_xs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
svr_ts = [-0.7, -0.42, -0.35, -0.05, 0.1, 0.28, 0.55, 0.6, 0.95]
regression_case("sinusoid_like", svr_xs, svr_ts, C=2.0, epsilon=0.1)

svr_xs2 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
svr_ts2 = [-2.6, -1.9, -0.8, 0.3, 1.1, 1.8, 3.2]
regression_case("wider_tube", svr_xs2, svr_ts2, C=5.0, epsilon=0.4)

# RVM regression (7.82-7.91): direct fixed-point re-estimation, replicated in numpy the
# same way chapter 3's evidence-approximation fixed point is (tools/golden/gen_regression.py),
# because it is the book's own prescribed iteration and not a search algorithm with a
# choice of heuristic to hide a bug behind.

ALPHA_MAX = 1e12


def rvm_regression_fit(phi, t, alpha0, beta0, max_iterations=1000, tol=1e-12):
    n, m = phi.shape
    alpha = np.array(alpha0, dtype=float)
    beta = beta0
    gram = phi.T @ phi
    phiT_t = phi.T @ t
    mean = np.zeros(m)
    for iteration in range(max_iterations):
        precision = np.diag(alpha) + beta * gram
        covariance = np.linalg.inv(precision)
        mean = beta * (covariance @ phiT_t)
        gamma = 1.0 - alpha * np.diag(covariance)
        new_alpha = np.array(
            [
                gamma[i] / (mean[i] ** 2) if mean[i] ** 2 > 0 else ALPHA_MAX
                for i in range(m)
            ]
        )
        new_alpha = np.minimum(new_alpha, ALPHA_MAX)
        residual = phi @ mean - t
        new_beta = (n - gamma.sum()) / (residual @ residual)
        converged = np.all(
            np.abs(new_alpha - alpha)
            <= tol * np.maximum(1, np.minimum(new_alpha, alpha))
        ) and abs(new_beta - beta) <= tol * max(1, beta)
        alpha, beta = new_alpha, new_beta
        if converged:
            break
    precision = np.diag(alpha) + beta * gram
    covariance = np.linalg.inv(precision)
    mean = beta * (covariance @ phiT_t)
    gamma = 1.0 - alpha * np.diag(covariance)
    return alpha, beta, mean, covariance, gamma, iteration + 1, converged


rvm_rng = np.random.default_rng(20260916)
rvm_x = np.linspace(-1.0, 1.0, 11)
rvm_centers = rvm_x.copy()


def rbf_design(x, centers, gamma):
    return np.exp(-gamma * (x[:, None] - centers[None, :]) ** 2)


RVM_GAMMA = 3.0
rvm_phi = np.hstack(
    [np.ones((len(rvm_x), 1)), rbf_design(rvm_x, rvm_centers, RVM_GAMMA)]
)
rvm_true_w = np.zeros(rvm_phi.shape[1])
rvm_true_w[0] = 0.2
rvm_true_w[3] = 1.5
rvm_true_w[7] = -1.2
rvm_t = rvm_phi @ rvm_true_w + 0.05 * rvm_rng.standard_normal(len(rvm_x))

alpha0 = np.ones(rvm_phi.shape[1])
alpha_r, beta_r, mean_r, cov_r, gamma_r, iters_r, converged_r = rvm_regression_fit(
    rvm_phi, rvm_t, alpha0, 1.0
)

cases.append(
    {
        "fn": "rvmRegressionFit",
        "design": rvm_phi.tolist(),
        "targets": rvm_t.tolist(),
        "initialAlpha": alpha0.tolist(),
        "initialBeta": 1.0,
        "expected": {
            "alpha": alpha_r.tolist(),
            "beta": float(beta_r),
            "mean": mean_r.tolist(),
            "covariance": cov_r.tolist(),
            "gamma": gamma_r.tolist(),
            "iterations": int(iters_r),
            "converged": bool(converged_r),
        },
    }
)

rvm_probe_phi = rbf_design(np.array([-0.8, -0.1, 0.4, 0.9]), rvm_centers, RVM_GAMMA)
rvm_probe_phi = np.hstack([np.ones((4, 1)), rvm_probe_phi])
pred_mean = rvm_probe_phi @ mean_r
pred_var = (1.0 / beta_r) + np.einsum(
    "ij,jk,ik->i", rvm_probe_phi, cov_r, rvm_probe_phi
)
cases.append(
    {
        "fn": "rvmRegressionPredictive",
        "phis": rvm_probe_phi.tolist(),
        "mean": mean_r.tolist(),
        "covariance": cov_r.tolist(),
        "beta": float(beta_r),
        "expected": {"mean": pred_mean.tolist(), "variance": pred_var.tolist()},
    }
)

# Sparsity analysis (7.92-7.107): closed-form quantities at a fixed (alpha, beta), no
# iteration to replicate.


def sparsity_statistics(phi, t, alpha, beta):
    n = phi.shape[0]
    C = (1.0 / beta) * np.eye(n) + phi @ np.diag(1.0 / alpha) @ phi.T
    C_inv = np.linalg.inv(C)
    C_inv_t = C_inv @ t
    Q = phi.T @ C_inv_t
    S = np.einsum("ni,nm,mi->i", phi, C_inv, phi)
    q = alpha * Q / (alpha - S)
    s = alpha * S / (alpha - S)
    return Q, S, q, s


sparsity_alpha = np.full(rvm_phi.shape[1], 2.0)
sparsity_alpha[3] = 0.05
sparsity_alpha[7] = 0.05
Q_s, S_s, q_s, s_s = sparsity_statistics(rvm_phi, rvm_t, sparsity_alpha, 5.0)
cases.append(
    {
        "fn": "rvmSparsityStatistics",
        "design": rvm_phi.tolist(),
        "targets": rvm_t.tolist(),
        "alpha": sparsity_alpha.tolist(),
        "beta": 5.0,
        "expected": {
            "Q": Q_s.tolist(),
            "S": S_s.tolist(),
            "q": q_s.tolist(),
            "s": s_s.tolist(),
        },
    }
)

alpha_grid = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
for i, (s_val, q_val) in enumerate([(1.0, 2.0), (2.0, 1.0)]):
    lam = 0.5 * (
        np.log(alpha_grid)
        - np.log(alpha_grid + s_val)
        + (q_val**2) / (alpha_grid + s_val)
    )
    cases.append(
        {
            "fn": "rvmSingleAlphaLogEvidenceTerm",
            "s": s_val,
            "q": q_val,
            "alphaGrid": alpha_grid.tolist(),
            "expected": lam.tolist(),
        }
    )
    optimal = (s_val**2) / (q_val**2 - s_val) if q_val**2 > s_val else None
    cases.append(
        {"fn": "rvmOptimalSingleAlpha", "s": s_val, "q": q_val, "expected": optimal}
    )

# RVM classification (7.108-7.119): reuses chapter 4's Newton-Raphson logistic fit with a
# diagonal ARD prior. Re-estimation formula 7.116 is replicated directly, exactly as the
# regression case above, since it is again the book's own prescribed fixed point.


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def newton_raphson_logistic(
    x, t, prior_precision, initial, max_iterations=100, tol=1e-12
):
    w = np.array(initial, dtype=float)
    hessian = prior_precision
    for _ in range(max_iterations):
        a = x @ w
        y = sigmoid(a)
        r = np.maximum(y * (1 - y), 1e-10)
        hessian = prior_precision + x.T @ (x * r[:, None])
        grad = prior_precision @ w + x.T @ (y - t)
        step = np.linalg.solve(hessian, grad)
        next_w = w - step
        delta = np.linalg.norm(next_w - w)
        w = next_w
        if delta <= tol:
            r_final = np.maximum(sigmoid(x @ w) * (1 - sigmoid(x @ w)), 1e-10)
            hessian = prior_precision + x.T @ (x * r_final[:, None])
            return w, hessian
    return w, hessian


def rvm_classification_fit(x, t, alpha0, max_iterations=200, tol=1e-10):
    m = x.shape[1]
    alpha = np.array(alpha0, dtype=float)
    w = np.zeros(m)
    hessian = np.diag(alpha)
    gamma = np.zeros(m)
    for iteration in range(max_iterations):
        w, hessian = newton_raphson_logistic(x, t, np.diag(alpha), np.zeros(m))
        covariance = np.linalg.inv(hessian)
        gamma = 1.0 - alpha * np.diag(covariance)
        new_alpha = np.array(
            [gamma[i] / (w[i] ** 2) if w[i] ** 2 > 0 else ALPHA_MAX for i in range(m)]
        )
        new_alpha = np.minimum(new_alpha, ALPHA_MAX)
        converged = np.all(
            np.abs(new_alpha - alpha)
            <= tol * np.maximum(1, np.minimum(new_alpha, alpha))
        )
        alpha = new_alpha
        if converged:
            break
    covariance = np.linalg.inv(hessian)
    return alpha, w, covariance, gamma, iteration + 1, converged


rvm_class_rng = np.random.default_rng(20260917)
n_per_class = 12
c0 = rvm_class_rng.normal(loc=[-1.0, -0.3], scale=0.6, size=(n_per_class, 2))
c1 = rvm_class_rng.normal(loc=[1.0, 0.3], scale=0.6, size=(n_per_class, 2))
class_points = np.vstack([c0, c1])
class_targets = np.array([0.0] * n_per_class + [1.0] * n_per_class)


def class_rbf_design(points, centers, gamma):
    diff = points[:, None, :] - centers[None, :, :]
    return np.exp(-gamma * np.sum(diff**2, axis=-1))


CLASS_GAMMA = 0.75
class_design = np.hstack(
    [
        np.ones((len(class_points), 1)),
        class_rbf_design(class_points, class_points, CLASS_GAMMA),
    ]
)
alpha0_class = np.ones(class_design.shape[1])
alpha_c, w_c, cov_c, gamma_c, iters_c, converged_c = rvm_classification_fit(
    class_design, class_targets, alpha0_class
)

cases.append(
    {
        "fn": "rvmClassificationFit",
        "design": class_design.tolist(),
        "targets": class_targets.tolist(),
        "initialAlpha": alpha0_class.tolist(),
        "expected": {
            "alpha": alpha_c.tolist(),
            "mean": w_c.tolist(),
            "covariance": cov_c.tolist(),
            "gamma": gamma_c.tolist(),
            "iterations": int(iters_c),
            "converged": bool(converged_c),
        },
    }
)

payload = {"cases": cases}
out = FIXTURES / "sparse.json"
out.write_text(json.dumps(payload))
print(f"wrote {out} ({len(cases)} cases)")
