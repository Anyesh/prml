"""Golden fixtures for packages/math/src/classification/.

Follows the shape documented in tools/golden/README.md: {"cases": [...]}, each case
carrying an "fn" discriminant, the inputs, and an "expected" value at full float64
precision. Nothing here re-derives the TypeScript algorithm from inside the TypeScript;
every case is either a direct scipy call (expit, norm.cdf, multivariate_normal.logpdf,
log_softmax) or an independent numpy replication of the PRML equation cited in the
TypeScript's own doc comment.

Equation references: least squares 4.16-4.17, perceptron 4.54-4.55, Fisher 4.28/4.30,
generative Gaussian classifier 4.73-4.76 and 4.62-4.63 and 4.66-4.67, logistic regression
cross-entropy 4.90-4.91, IRLS 4.92-4.100, Laplace approximation 4.126-4.135, Bayesian
logistic regression 4.140-4.144, probit and the logistic-Gaussian convolution 4.114 and
4.149-4.155.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit, log_softmax
from scipy.stats import multivariate_normal, norm

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

rng = np.random.default_rng(20260917)

MEAN1 = np.array([-1.5, 0.5])
MEAN2 = np.array([1.5, -0.5])
MEAN3 = np.array([0.0, 3.0])
COV_SHARED = np.array([[1.0, 0.3], [0.3, 0.6]])
COV3 = np.array([[0.4, -0.1], [-0.1, 0.3]])
N_PER_CLASS = 40


def sample(mean, cov, n):
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, 2))
    return mean + z @ chol.T


class1 = sample(MEAN1, COV_SHARED, N_PER_CLASS)
class2 = sample(MEAN2, COV_SHARED, N_PER_CLASS)
class3 = sample(MEAN3, COV3, N_PER_CLASS)


def with_bias(x):
    return np.hstack([np.ones((x.shape[0], 1)), x])


X3 = with_bias(np.vstack([class1, class2, class3]))
T3 = np.zeros((X3.shape[0], 3))
T3[:N_PER_CLASS, 0] = 1.0
T3[N_PER_CLASS : 2 * N_PER_CLASS, 1] = 1.0
T3[2 * N_PER_CLASS :, 2] = 1.0

W_LS = np.linalg.pinv(X3, rcond=1e-12) @ T3
probe_x = np.array(
    [[1.0, -1.5, 0.5], [1.0, 1.5, -0.5], [1.0, 0.0, 3.0], [1.0, 0.0, 0.0]]
)
scores = probe_x @ W_LS

cases.append(
    {
        "fn": "leastSquaresClassifierWeights",
        "design": X3.tolist(),
        "targets": T3.tolist(),
        "expected": W_LS.tolist(),
    }
)
cases.append(
    {
        "fn": "leastSquaresScores",
        "design": probe_x.tolist(),
        "weights": W_LS.tolist(),
        "expected": scores.tolist(),
    }
)

sep_pos = np.array(
    [[2.0, 2.0], [3.0, 1.0], [3.0, 3.0], [4.0, 2.0], [2.0, 3.0], [3.0, 2.0]]
)
sep_neg = np.array(
    [[-2.0, -2.0], [-3.0, -1.0], [-3.0, -3.0], [-4.0, -2.0], [-2.0, -3.0], [-3.0, -2.0]]
)
X_perc = with_bias(np.vstack([sep_pos, sep_neg]))
t_perc = np.array([1.0] * len(sep_pos) + [-1.0] * len(sep_neg))

# One flipped label, so this set is not linearly separable and the maxEpochs cutoff
# below is genuinely exercised rather than converging early like the set above.
non_sep = X_perc.copy()
t_non_sep = t_perc.copy()
t_non_sep[0] = -1.0


def perceptron_train(x, t, initial, max_epochs, lr=1.0):
    w = np.array(initial, dtype=float)
    history = []
    for epoch in range(max_epochs):
        misclassified = 0
        for n in range(x.shape[0]):
            a = w @ x[n]
            if a * t[n] <= 0:
                w = w + lr * x[n] * t[n]
                misclassified += 1
        history.append(misclassified)
        if misclassified == 0:
            return w, epoch + 1, True, history
    return w, max_epochs, False, history


w_sep, epochs_sep, converged_sep, hist_sep = perceptron_train(
    X_perc, t_perc, [0.0, 0.0, 0.0], 1000
)
cases.append(
    {
        "fn": "perceptronTrain",
        "design": X_perc.tolist(),
        "targets": t_perc.tolist(),
        "initialWeights": [0.0, 0.0, 0.0],
        "maxEpochs": 1000,
        "expected": {
            "weights": w_sep.tolist(),
            "epochs": epochs_sep,
            "converged": converged_sep,
            "history": hist_sep,
        },
    }
)

w_ns, epochs_ns, converged_ns, hist_ns = perceptron_train(
    non_sep, t_non_sep, [0.0, 0.0, 0.0], 5
)
cases.append(
    {
        "fn": "perceptronTrain",
        "design": non_sep.tolist(),
        "targets": t_non_sep.tolist(),
        "initialWeights": [0.0, 0.0, 0.0],
        "maxEpochs": 5,
        "expected": {
            "weights": w_ns.tolist(),
            "epochs": epochs_ns,
            "converged": converged_ns,
            "history": hist_ns,
        },
    }
)

mean1, mean2 = class1.mean(axis=0), class2.mean(axis=0)
scatter1 = (class1 - mean1).T @ (class1 - mean1)
scatter2 = (class2 - mean2).T @ (class2 - mean2)
s_w = scatter1 + scatter2
w_fisher = np.linalg.solve(s_w, mean2 - mean1)

cases.append(
    {
        "fn": "withinClassScatter",
        "class1": class1.tolist(),
        "class2": class2.tolist(),
        "mean1": mean1.tolist(),
        "mean2": mean2.tolist(),
        "expected": s_w.tolist(),
    }
)
cases.append(
    {
        "fn": "fisherDirection",
        "mean1": mean1.tolist(),
        "mean2": mean2.tolist(),
        "within": s_w.tolist(),
        "expected": w_fisher.tolist(),
    }
)

class_points = [class1, class2, class3]
means = [c.mean(axis=0) for c in class_points]
total = sum(c.shape[0] for c in class_points)
priors = [c.shape[0] / total for c in class_points]

pooled = np.zeros((2, 2))
for c, m in zip(class_points, means):
    pooled += (c - m).T @ (c - m)
cov_shared_fit = pooled / total

separate_covs = [
    ((c - m).T @ (c - m)) / c.shape[0] for c, m in zip(class_points, means)
]

cases.append(
    {
        "fn": "fitSharedCovarianceGaussian",
        "classPoints": [c.tolist() for c in class_points],
        "expected": {
            "priors": priors,
            "means": [m.tolist() for m in means],
            "covariance": cov_shared_fit.tolist(),
        },
    }
)
cases.append(
    {
        "fn": "fitSeparateCovarianceGaussian",
        "classPoints": [c.tolist() for c in class_points],
        "expected": {
            "priors": priors,
            "means": [m.tolist() for m in means],
            "covariances": [c.tolist() for c in separate_covs],
        },
    }
)

probe_points = [[-1.5, 0.5], [1.5, -0.5], [0.0, 3.0], [0.0, 0.5], [-0.5, 1.5]]
shared_posteriors = []
separate_posteriors = []
for x in probe_points:
    log_shared = np.array(
        [
            multivariate_normal.logpdf(x, mean=means[k], cov=cov_shared_fit)
            + np.log(priors[k])
            for k in range(3)
        ]
    )
    shared_posteriors.append(np.exp(log_softmax(log_shared)).tolist())

    log_sep = np.array(
        [
            multivariate_normal.logpdf(x, mean=means[k], cov=separate_covs[k])
            + np.log(priors[k])
            for k in range(3)
        ]
    )
    separate_posteriors.append(np.exp(log_softmax(log_sep)).tolist())

cases.append(
    {
        "fn": "posteriorSharedCovariance",
        "classPoints": [c.tolist() for c in class_points],
        "points": probe_points,
        "expected": shared_posteriors,
    }
)
cases.append(
    {
        "fn": "posteriorSeparateCovariance",
        "classPoints": [c.tolist() for c in class_points],
        "points": probe_points,
        "expected": separate_posteriors,
    }
)

two_class_points = [class1, class2]
two_means = [class1.mean(axis=0), class2.mean(axis=0)]
two_total = class1.shape[0] + class2.shape[0]
two_priors = [class1.shape[0] / two_total, class2.shape[0] / two_total]
two_pooled = np.zeros((2, 2))
for c, m in zip(two_class_points, two_means):
    two_pooled += (c - m).T @ (c - m)
two_cov = two_pooled / two_total
sigma_inv = np.linalg.inv(two_cov)
w_boundary = sigma_inv @ (two_means[0] - two_means[1])
w0_boundary = (
    -0.5 * two_means[0] @ sigma_inv @ two_means[0]
    + 0.5 * two_means[1] @ sigma_inv @ two_means[1]
    + np.log(two_priors[0] / two_priors[1])
)

cases.append(
    {
        "fn": "sharedCovarianceLinearBoundary",
        "classPoints": [c.tolist() for c in two_class_points],
        "expected": {"w": w_boundary.tolist(), "w0": float(w0_boundary)},
    }
)

logit_rng = np.random.default_rng(20260918)
logit_mean1, logit_mean2 = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
N_LOGIT = 30
logit_class1 = logit_mean1 + logit_rng.standard_normal((N_LOGIT, 2))
logit_class2 = logit_mean2 + logit_rng.standard_normal((N_LOGIT, 2))
X_logit = with_bias(np.vstack([logit_class1, logit_class2]))
t_logit = np.array([1.0] * N_LOGIT + [0.0] * N_LOGIT)

probe_weights = [
    [0.0, 0.0, 0.0],
    [0.1, -0.5, 0.8],
    [-0.3, 1.2, -0.4],
]
for pw in probe_weights:
    w = np.array(pw)
    a = X_logit @ w
    err = float(np.sum(np.logaddexp(0.0, a) - t_logit * a))
    grad = X_logit.T @ (expit(a) - t_logit)
    cases.append(
        {
            "fn": "crossEntropyError",
            "design": X_logit.tolist(),
            "targets": t_logit.tolist(),
            "weights": w.tolist(),
            "expected": err,
        }
    )
    cases.append(
        {
            "fn": "crossEntropyGradient",
            "design": X_logit.tolist(),
            "targets": t_logit.tolist(),
            "weights": w.tolist(),
            "expected": grad.tolist(),
        }
    )

R_FLOOR = 1e-10


def newton_raphson_logistic(
    x, t, prior_mean, prior_precision, initial, max_iterations=50, tol=1e-10
):
    w = np.array(initial, dtype=float)
    history = []

    def evaluate(w):
        a = x @ w
        y = expit(a)
        r = np.maximum(y * (1 - y), R_FLOOR)
        hessian = prior_precision + x.T @ (x * r[:, None])
        grad = prior_precision @ (w - prior_mean) + x.T @ (y - t)
        return hessian, grad

    for iteration in range(max_iterations):
        hessian, grad = evaluate(w)
        step = np.linalg.solve(hessian, grad)
        next_w = w - step
        history.append(next_w.tolist())
        delta = np.linalg.norm(next_w - w)
        w = next_w
        if delta <= tol:
            final_hessian, _ = evaluate(w)
            return w, final_hessian, iteration + 1, True, history

    final_hessian, _ = evaluate(w)
    return w, final_hessian, max_iterations, False, history


dim = X_logit.shape[1]
w_irls, h_irls, iters_irls, converged_irls, hist_irls = newton_raphson_logistic(
    X_logit, t_logit, np.zeros(dim), np.zeros((dim, dim)), np.zeros(dim)
)
cases.append(
    {
        "fn": "irlsFit",
        "design": X_logit.tolist(),
        "targets": t_logit.tolist(),
        "expected": {
            "weights": w_irls.tolist(),
            "precision": h_irls.tolist(),
            "iterations": iters_irls,
            "converged": converged_irls,
            "history": hist_irls,
        },
    }
)


def gamma_log_density_1d(a):
    def gradient(z):
        return np.array([(a - 1) / z[0] - 1.0])

    def hessian(z):
        return np.array([[-(a - 1) / z[0] ** 2]])

    return gradient, hessian


def laplace_approx(gradient, hessian, x0, max_iterations=100, tol=1e-12):
    x = np.array(x0, dtype=float)

    def precision_at(point):
        return -hessian(point)

    for iteration in range(max_iterations):
        g = gradient(x)
        precision = precision_at(x)
        step = np.linalg.solve(precision, g)
        next_x = x + step
        delta = np.linalg.norm(next_x - x)
        x = next_x
        if delta <= tol:
            final_precision = precision_at(x)
            return x, final_precision, iteration + 1, True

    final_precision = precision_at(x)
    return x, final_precision, max_iterations, False


# Gamma-shaped, not Gaussian: a Gaussian's Laplace approximation is exact after one
# step and would not exercise the iteration loop at all.
grad_1d, hess_1d = gamma_log_density_1d(2.0)
mode_1d, prec_1d, iters_1d, converged_1d = laplace_approx(grad_1d, hess_1d, [0.5])
cov_1d = np.linalg.inv(prec_1d)
cases.append(
    {
        "fn": "laplaceApproximation1D",
        "shapeParam": 2.0,
        "x0": [0.5],
        "expected": {
            "mode": mode_1d.tolist(),
            "precision": prec_1d.tolist(),
            "covariance": cov_1d.tolist(),
            "iterations": iters_1d,
            "converged": converged_1d,
        },
    }
)


def gamma_log_density_2d(a1, a2):
    def gradient(z):
        return np.array([(a1 - 1) / z[0] - 1.0, (a2 - 1) / z[1] - 1.0])

    def hessian(z):
        return np.array([[-(a1 - 1) / z[0] ** 2, 0.0], [0.0, -(a2 - 1) / z[1] ** 2]])

    return gradient, hessian


grad_2d, hess_2d = gamma_log_density_2d(2.0, 3.0)
mode_2d, prec_2d, iters_2d, converged_2d = laplace_approx(grad_2d, hess_2d, [0.5, 0.5])
cov_2d = np.linalg.inv(prec_2d)
cases.append(
    {
        "fn": "laplaceApproximation2D",
        "shapeParams": [2.0, 3.0],
        "x0": [0.5, 0.5],
        "expected": {
            "mode": mode_2d.tolist(),
            "precision": prec_2d.tolist(),
            "covariance": cov_2d.tolist(),
            "iterations": iters_2d,
            "converged": converged_2d,
        },
    }
)

prior_mean = np.zeros(dim)
prior_cov = np.eye(dim)
prior_precision = np.linalg.inv(prior_cov)

w_map, h_map, iters_map, converged_map, _hist_map = newton_raphson_logistic(
    X_logit, t_logit, prior_mean, prior_precision, np.zeros(dim)
)
cov_map = np.linalg.inv(h_map)
cases.append(
    {
        "fn": "fitLaplaceLogisticPosterior",
        "design": X_logit.tolist(),
        "targets": t_logit.tolist(),
        "priorMean": prior_mean.tolist(),
        "priorCovariance": prior_cov.tolist(),
        "expected": {
            "mean": w_map.tolist(),
            "covariance": cov_map.tolist(),
            "precision": h_map.tolist(),
            "iterations": iters_map,
            "converged": converged_map,
        },
    }
)

probit_args = [-3.0, -1.0, 0.0, 0.5, 2.0, 5.0]
cases.append(
    {
        "fn": "probit",
        "args": probit_args,
        "expected": [float(norm.cdf(a)) for a in probit_args],
    }
)

kappa_variances = [0.0, 0.1, 1.0, 5.0, 20.0]
cases.append(
    {
        "fn": "kappa",
        "variances": kappa_variances,
        "expected": [
            float(1.0 / np.sqrt(1.0 + np.pi * v / 8.0)) for v in kappa_variances
        ],
    }
)

convolution_points = [(0.0, 0.0), (1.0, 0.5), (-2.0, 2.0), (3.0, 4.0), (0.5, 10.0)]
cases.append(
    {
        "fn": "logisticGaussianConvolution",
        "points": [{"mean": m, "variance": v} for m, v in convolution_points],
        "expected": [
            float(expit((1.0 / np.sqrt(1.0 + np.pi * v / 8.0)) * m))
            for m, v in convolution_points
        ],
    }
)

predictive_phis = [[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, -0.5]]
predictive_expected = []
for phi in predictive_phis:
    phi = np.array(phi)
    mu = float(phi @ w_map)
    var = float(phi @ cov_map @ phi)
    kap = 1.0 / np.sqrt(1.0 + np.pi * var / 8.0)
    predictive_expected.append(float(expit(kap * mu)))

cases.append(
    {
        "fn": "laplaceLogisticPredictive",
        "posteriorMean": w_map.tolist(),
        "posteriorCovariance": cov_map.tolist(),
        "phis": predictive_phis,
        "expected": predictive_expected,
    }
)

payload = {"cases": cases}
out = FIXTURES / "classification.json"
out.write_text(json.dumps(payload))
print(f"wrote {out} ({len(cases)} cases)")
