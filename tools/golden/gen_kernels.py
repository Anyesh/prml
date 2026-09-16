"""Golden fixtures for packages/math/src/kernels/.

Follows tools/golden/README.md: {"cases": [...]}, each case an "fn" discriminant, its
inputs, and an "expected" value computed independently in numpy, never hand-typed.

Kernel functions here operate on vectors (PRML's x, x' are vectors throughout chapter 6),
so a "1-D" case is a vector of length 1. dataset1d below is the shared small training set
used for dual ridge regression, Nadaraya-Watson, and Gaussian process regression; it is
kept to N = 6 points deliberately, because 6.1's widget shows the N x N matrix being
inverted at its actual on-screen size, and a bigger N would defeat that.

Where PRML gives a formula and no independent scipy call exists to check it against (the
kernel-construction rules, the GP log marginal likelihood gradient, the Laplace mode for
GP classification), the expected value is a from-scratch numpy re-implementation of the
same mathematics, in a different language and linear-algebra path than the TypeScript
(numpy's LAPACK bindings rather than ml-matrix's Cholesky), which is the same standard
gen_regression.py already applies to log_evidence and maximise_evidence. Gradients that
have no closed-form fixture partner are additionally checked against central finite
differences with a plain assert in this script, so a wrong formula fails here rather than
silently shipping.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []

rng = np.random.default_rng(20260601)
N1D = 6
X1D = np.sort(rng.uniform(0.0, 1.0, N1D))
NOISE_SD = 0.15
T1D = np.sin(2 * np.pi * X1D) + rng.normal(0.0, NOISE_SD, N1D)
TEST_X1D = np.array([0.05, 0.2, 0.45, 0.6, 0.85, 0.98])

VECTORS_2D = [
    np.array([1.0, 0.5]),
    np.array([-0.3, 0.8]),
    np.array([0.2, -0.6]),
    np.array([0.0, 1.2]),
]
VECTORS_3D = [
    np.array([1.0, -1.0, 0.5]),
    np.array([0.2, 0.3, -0.4]),
    np.array([-0.5, 0.1, 0.9]),
]

for a in VECTORS_2D:
    for b in VECTORS_2D:
        cases.append(
            {
                "fn": "linearKernel",
                "x": a.tolist(),
                "xp": b.tolist(),
                "expected": float(a @ b),
            }
        )
for a in VECTORS_3D:
    for b in VECTORS_3D:
        cases.append(
            {
                "fn": "linearKernel",
                "x": a.tolist(),
                "xp": b.tolist(),
                "expected": float(a @ b),
            }
        )

for degree, c in ((2, 0.0), (2, 1.0), (3, 0.0), (3, 0.5), (4, 2.0)):
    for a in VECTORS_2D:
        for b in VECTORS_2D:
            cases.append(
                {
                    "fn": "polynomialKernel",
                    "degree": degree,
                    "c": c,
                    "x": a.tolist(),
                    "xp": b.tolist(),
                    "expected": float((a @ b + c) ** degree),
                }
            )

for length_scale in (0.1, 0.5, 2.0):
    for a in VECTORS_2D:
        for b in VECTORS_2D:
            d2 = float(np.sum((a - b) ** 2))
            cases.append(
                {
                    "fn": "rbfKernel",
                    "lengthScale": length_scale,
                    "x": a.tolist(),
                    "xp": b.tolist(),
                    "expected": float(np.exp(-0.5 * d2 / length_scale**2)),
                }
            )

ARD_SCALES = [0.3, 5.0]
for a in VECTORS_2D:
    for b in VECTORS_2D:
        d2 = sum(((a[i] - b[i]) / ARD_SCALES[i]) ** 2 for i in range(2))
        cases.append(
            {
                "fn": "rbfKernel",
                "lengthScale": ARD_SCALES,
                "x": a.tolist(),
                "xp": b.tolist(),
                "expected": float(np.exp(-0.5 * d2)),
            }
        )

for theta in (0.5, 2.0, 5.0):
    for a in VECTORS_2D:
        for b in VECTORS_2D:
            cases.append(
                {
                    "fn": "exponentialKernel",
                    "theta": theta,
                    "x": a.tolist(),
                    "xp": b.tolist(),
                    "expected": float(np.exp(-theta * np.linalg.norm(a - b))),
                }
            )

COMPOSITE_PARAM_SETS = [
    (1.0, 4.0, 0.0, 0.0),
    (9.0, 4.0, 0.0, 0.0),
    (1.0, 64.0, 0.0, 0.0),
    (1.0, 0.25, 0.0, 0.0),
    (1.0, 4.0, 10.0, 0.0),
    (1.0, 4.0, 0.0, 5.0),
]


def composite_kernel(params, x, xp):
    theta0, theta1, theta2, theta3 = params
    d2 = float(np.sum((x - xp) ** 2))
    return theta0 * np.exp(-theta1 / 2 * d2) + theta2 + theta3 * float(x @ xp)


PROBE_1D = [
    np.array([-0.8]),
    np.array([-0.2]),
    np.array([0.0]),
    np.array([0.3]),
    np.array([0.9]),
]
for params in COMPOSITE_PARAM_SETS:
    for a in PROBE_1D:
        for b in PROBE_1D:
            cases.append(
                {
                    "fn": "compositeKernel",
                    "params": {
                        "theta0": params[0],
                        "theta1": params[1],
                        "theta2": params[2],
                        "theta3": params[3],
                    },
                    "x": a.tolist(),
                    "xp": b.tolist(),
                    "expected": composite_kernel(params, a, b),
                }
            )


def composite_partials_analytic(params, x, xp):
    theta0, theta1, _theta2, _theta3 = params
    d2 = float(np.sum((x - xp) ** 2))
    gaussian_term = np.exp(-theta1 / 2 * d2)
    return [
        gaussian_term,
        -0.5 * theta0 * d2 * gaussian_term,
        1.0,
        float(x @ xp),
    ]


def composite_kernel_from_vec(theta_vec, x, xp):
    return composite_kernel(theta_vec, x, xp)


for params in COMPOSITE_PARAM_SETS:
    theta_vec = np.array(params)
    for a in PROBE_1D:
        for b in PROBE_1D:
            analytic = composite_partials_analytic(params, a, b)
            h = 1e-6
            for i in range(4):
                bumped = theta_vec.copy()
                bumped[i] += h
                plus = composite_kernel_from_vec(bumped, a, b)
                bumped[i] -= 2 * h
                minus = composite_kernel_from_vec(bumped, a, b)
                findiff = (plus - minus) / (2 * h)
                assert abs(findiff - analytic[i]) < 1e-4, (
                    params,
                    a,
                    b,
                    i,
                    findiff,
                    analytic[i],
                )
            cases.append(
                {
                    "fn": "compositeKernelPartials",
                    "params": {
                        "theta0": params[0],
                        "theta1": params[1],
                        "theta2": params[2],
                        "theta3": params[3],
                    },
                    "x": a.tolist(),
                    "xp": b.tolist(),
                    "expected": [float(v) for v in analytic],
                }
            )

for a_coef, b_coef in ((1.0, 0.0), (0.5, -0.2), (2.0, 1.0)):
    for a in VECTORS_2D:
        for b in VECTORS_2D:
            cases.append(
                {
                    "fn": "sigmoidKernel",
                    "a": a_coef,
                    "b": b_coef,
                    "x": a.tolist(),
                    "xp": b.tolist(),
                    "expected": float(np.tanh(a_coef * (a @ b) + b_coef)),
                }
            )


def rbf(x, xp, length_scale):
    return float(np.exp(-0.5 * np.sum((x - xp) ** 2) / length_scale**2))


def linear(x, xp):
    return float(x @ xp)


gram_points = VECTORS_2D
gram_expected = [[rbf(a, b, 0.7) for b in gram_points] for a in gram_points]
cases.append(
    {
        "fn": "gramMatrix",
        "kernel": "rbf",
        "lengthScale": 0.7,
        "points": [p.tolist() for p in gram_points],
        "expected": gram_expected,
    }
)

for a in VECTORS_2D:
    for b in VECTORS_2D:
        lin = linear(a, b)
        rb = rbf(a, b, 1.0)
        cases.append(
            {"fn": "sumKernel", "x": a.tolist(), "xp": b.tolist(), "expected": lin + rb}
        )
        cases.append(
            {
                "fn": "productKernel",
                "x": a.tolist(),
                "xp": b.tolist(),
                "expected": lin * rb,
            }
        )
        cases.append(
            {
                "fn": "scaleKernel",
                "c": 3.5,
                "x": a.tolist(),
                "xp": b.tolist(),
                "expected": 3.5 * lin,
            }
        )

X1D_VEC = X1D.reshape(-1, 1)


def gram_1d(xs, kernel_fn):
    return np.array(
        [[kernel_fn(np.array([xi]), np.array([xj])) for xj in xs] for xi in xs]
    )


for length_scale, lam in ((0.1, 1e-3), (0.1, 1.0), (0.3, 1e-2)):
    k = gram_1d(X1D, lambda x, xp: rbf(x, xp, length_scale))
    a = np.linalg.solve(k + lam * np.eye(N1D), T1D)
    predictions = []
    for xt in TEST_X1D:
        kx = np.array([rbf(np.array([xt]), np.array([xn]), length_scale) for xn in X1D])
        predictions.append(float(kx @ a))
    cases.append(
        {
            "fn": "dualRidgeCoefficients",
            "kernel": "rbf",
            "lengthScale": length_scale,
            "lambda": lam,
            "expected": a.tolist(),
        }
    )
    cases.append(
        {
            "fn": "dualPredict",
            "kernel": "rbf",
            "lengthScale": length_scale,
            "lambda": lam,
            "testX": TEST_X1D.tolist(),
            "expected": predictions,
        }
    )

for bandwidth in (0.05, 0.15, 0.4):
    weight_rows = []
    predictions = []
    for xt in TEST_X1D:
        raw = np.array([rbf(np.array([xt]), np.array([xn]), bandwidth) for xn in X1D])
        w = raw / raw.sum()
        weight_rows.append(w.tolist())
        predictions.append(float(w @ T1D))
    cases.append(
        {
            "fn": "nadarayaWatson",
            "bandwidth": bandwidth,
            "testX": TEST_X1D.tolist(),
            "expectedWeights": weight_rows,
            "expectedPredictions": predictions,
        }
    )

GP_PARAMS = (1.0, 8.0, 0.1, 0.5)
GP_NOISE_VARIANCE = 1.0 / 25.0

# The TypeScript adds this on top of the noise variance before factorising, because a
# Gram matrix from closely spaced points is near-singular and Cholesky fails on it. The
# fixture must model the same matrix or it is comparing two different problems.
GP_JITTER = 1e-8


def gp_gram(xs, params):
    return np.array(
        [
            [composite_kernel(params, np.array([xi]), np.array([xj])) for xj in xs]
            for xi in xs
        ]
    )


CN = gp_gram(X1D, GP_PARAMS) + (GP_NOISE_VARIANCE + GP_JITTER) * np.eye(N1D)
CN_inv = np.linalg.inv(CN)
alpha = CN_inv @ T1D

gp_predictions = []
for xt in TEST_X1D:
    k_vec = np.array(
        [composite_kernel(GP_PARAMS, np.array([xt]), np.array([xn])) for xn in X1D]
    )
    mean = float(k_vec @ alpha)
    c = composite_kernel(GP_PARAMS, np.array([xt]), np.array([xt])) + GP_NOISE_VARIANCE
    variance = float(c - k_vec @ CN_inv @ k_vec)
    gp_predictions.append({"x": float(xt), "mean": mean, "variance": variance})

cases.append(
    {
        "fn": "gpPredict",
        "params": {
            "theta0": GP_PARAMS[0],
            "theta1": GP_PARAMS[1],
            "theta2": GP_PARAMS[2],
            "theta3": GP_PARAMS[3],
        },
        "noiseVariance": GP_NOISE_VARIANCE,
        "expected": gp_predictions,
    }
)

sign, logdet = np.linalg.slogdet(CN)
assert sign > 0
log_ml = -0.5 * logdet - 0.5 * float(T1D @ alpha) - 0.5 * N1D * np.log(2 * np.pi)
cases.append(
    {
        "fn": "gpLogMarginalLikelihood",
        "params": {
            "theta0": GP_PARAMS[0],
            "theta1": GP_PARAMS[1],
            "theta2": GP_PARAMS[2],
            "theta3": GP_PARAMS[3],
        },
        "noiseVariance": GP_NOISE_VARIANCE,
        "expected": float(log_ml),
    }
)


def log_ml_at(params, noise_variance):
    cn = gp_gram(X1D, params) + (noise_variance + GP_JITTER) * np.eye(N1D)
    s, ld = np.linalg.slogdet(cn)
    a = np.linalg.solve(cn, T1D)
    return -0.5 * ld - 0.5 * float(T1D @ a) - 0.5 * N1D * np.log(2 * np.pi)


def dCN_dtheta(params, xs, index):
    theta0, theta1, _theta2, _theta3 = params
    n = len(xs)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xi, xj = np.array([xs[i]]), np.array([xs[j]])
            d2 = float(np.sum((xi - xj) ** 2))
            gaussian_term = np.exp(-theta1 / 2 * d2)
            if index == 0:
                out[i, j] = gaussian_term
            elif index == 1:
                out[i, j] = -0.5 * theta0 * d2 * gaussian_term
            elif index == 2:
                out[i, j] = 1.0
            else:
                out[i, j] = xi @ xj
    return out


gradient = []
for i in range(4):
    dcn = dCN_dtheta(GP_PARAMS, X1D, i)
    cn_inv_dcn = CN_inv @ dcn
    trace_term = np.trace(cn_inv_dcn)
    quad_term = float(alpha @ dcn @ alpha)
    gradient.append(-0.5 * trace_term + 0.5 * quad_term)

h = 1e-6
for i in range(4):
    bumped = list(GP_PARAMS)
    bumped[i] += h
    plus = log_ml_at(tuple(bumped), GP_NOISE_VARIANCE)
    bumped[i] -= 2 * h
    minus = log_ml_at(tuple(bumped), GP_NOISE_VARIANCE)
    findiff = (plus - minus) / (2 * h)
    assert abs(findiff - gradient[i]) < 1e-3, (i, findiff, gradient[i])

cases.append(
    {
        "fn": "gpLogMarginalLikelihoodGradient",
        "params": {
            "theta0": GP_PARAMS[0],
            "theta1": GP_PARAMS[1],
            "theta2": GP_PARAMS[2],
            "theta3": GP_PARAMS[3],
        },
        "noiseVariance": GP_NOISE_VARIANCE,
        "expected": [float(g) for g in gradient],
    }
)

crng = np.random.default_rng(7)
N_PER_CLASS = 6
class0 = crng.normal(loc=[-1.0, -1.0], scale=0.6, size=(N_PER_CLASS, 2))
class1 = crng.normal(loc=[1.0, 1.0], scale=0.6, size=(N_PER_CLASS, 2))
CLASS_X = np.vstack([class0, class1])
CLASS_T = np.array([0.0] * N_PER_CLASS + [1.0] * N_PER_CLASS)

CLS_LENGTH_SCALE = 1.2
NU = 1e-6


def cls_kernel(x, xp):
    return rbf(x, xp, CLS_LENGTH_SCALE)


CN_cls = np.array([[cls_kernel(a, b) for b in CLASS_X] for a in CLASS_X]) + NU * np.eye(
    len(CLASS_X)
)
CN_cls_inv = np.linalg.inv(CN_cls)

a_mode = np.zeros(len(CLASS_X))
for _ in range(100):
    sigma = 1.0 / (1.0 + np.exp(-a_mode))
    w = sigma * (1 - sigma)
    grad = CLASS_T - sigma - CN_cls_inv @ a_mode
    hessian = -(np.diag(w) + CN_cls_inv)
    step = np.linalg.solve(-hessian, grad)
    a_new = a_mode + step
    if np.linalg.norm(a_new - a_mode) < 1e-14:
        a_mode = a_new
        break
    a_mode = a_new

sigma_star = 1.0 / (1.0 + np.exp(-a_mode))
residual = CLASS_T - sigma_star
w_star = sigma_star * (1 - sigma_star)

cases.append(
    {
        "fn": "gpClassificationMode",
        "lengthScale": CLS_LENGTH_SCALE,
        "nu": NU,
        "points": CLASS_X.tolist(),
        "targets": CLASS_T.tolist(),
        "expected": a_mode.tolist(),
    }
)

TEST_POINTS_2D = [
    np.array([0.0, 0.0]),
    np.array([-1.0, -1.0]),
    np.array([1.2, 0.8]),
    np.array([2.0, -2.0]),
]
cls_predictions = []
for xt in TEST_POINTS_2D:
    k_vec = np.array([cls_kernel(xt, xn) for xn in CLASS_X])
    mean_a = float(k_vec @ residual)
    c = cls_kernel(xt, xt) + NU
    w_inv_plus_cn = np.diag(1.0 / w_star) + CN_cls
    variance_a = float(c - k_vec @ np.linalg.solve(w_inv_plus_cn, k_vec))
    variance_a = max(0.0, variance_a)
    kappa = 1.0 / np.sqrt(1.0 + np.pi * variance_a / 8.0)
    prob = 1.0 / (1.0 + np.exp(-kappa * mean_a))
    cls_predictions.append(
        {
            "x": xt.tolist(),
            "meanA": mean_a,
            "varianceA": variance_a,
            "probability": float(prob),
        }
    )

cases.append(
    {
        "fn": "gpClassificationPredict",
        "lengthScale": CLS_LENGTH_SCALE,
        "nu": NU,
        "points": CLASS_X.tolist(),
        "targets": CLASS_T.tolist(),
        "testPoints": [p.tolist() for p in TEST_POINTS_2D],
        "expected": cls_predictions,
    }
)

payload = {
    "dataset1d": {
        "x": X1D.tolist(),
        "t": T1D.tolist(),
        "noiseSd": NOISE_SD,
        "testX": TEST_X1D.tolist(),
    },
    "classificationDataset": {"points": CLASS_X.tolist(), "targets": CLASS_T.tolist()},
    "cases": cases,
}

out = FIXTURES / "kernels.json"
out.write_text(json.dumps(payload))
print(f"wrote {out} ({len(cases)} cases)")
