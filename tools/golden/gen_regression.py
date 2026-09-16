"""Golden fixtures for packages/math/src/regression/.

Follows the shape documented in tools/golden/README.md: {"cases": [...]}, each case
carrying an "fn" discriminant, the inputs, and an "expected" value at full float64
precision.

Everything here is computed from the PRML equations independently of the TypeScript, in
the parameterisation the TS signatures declare: alpha is the isotropic prior precision
over the weights (3.52) and beta the noise precision (3.8), never variances.

Equation references for the helpers below: `posterior` is 3.53-3.54 for the isotropic
prior, `log_evidence` is 3.86, `maximise_evidence` is the 3.91-3.95 fixed point.

The sinusoidal dataset is the book's own running example for chapter 3 (figures 3.7-3.9):
targets are sin(2*pi*x) plus Gaussian noise. The draw is pinned with a fixed numpy seed so
the fixture is reproducible, but the TypeScript never reproduces the draw itself - the x
and t arrays travel in the fixture.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260916)
N = 25
X = np.sort(rng.uniform(0.0, 1.0, N))
NOISE_SD = 0.2
T = np.sin(2 * np.pi * X) + rng.normal(0.0, NOISE_SD, N)

CENTRES = np.linspace(0.0, 1.0, 9)
SCALE = 0.1


def polynomial_design(x, degree, bias=True):
    powers = np.arange(0 if bias else 1, degree + 1)
    return x[:, None] ** powers[None, :]


def gaussian_design(x, centres, scale, bias=True):
    bumps = np.exp(-((x[:, None] - centres[None, :]) ** 2) / (2 * scale**2))
    if not bias:
        return bumps
    return np.hstack([np.ones((x.size, 1)), bumps])


def sigmoidal_design(x, centres, scale, bias=True):
    a = (x[:, None] - centres[None, :]) / scale
    bumps = 1.0 / (1.0 + np.exp(-a))
    if not bias:
        return bumps
    return np.hstack([np.ones((x.size, 1)), bumps])


def posterior(phi, t, alpha, beta):
    precision = alpha * np.eye(phi.shape[1]) + beta * phi.T @ phi
    cov = np.linalg.inv(precision)
    mean = beta * cov @ phi.T @ t
    return mean, cov, precision


def log_evidence(phi, t, alpha, beta):
    n, m = phi.shape
    mean, _, precision = posterior(phi, t, alpha, beta)
    residual = t - phi @ mean
    e_mn = 0.5 * beta * residual @ residual + 0.5 * alpha * mean @ mean
    sign, logdet = np.linalg.slogdet(precision)
    assert sign > 0
    return (
        0.5 * m * np.log(alpha)
        + 0.5 * n * np.log(beta)
        - e_mn
        - 0.5 * logdet
        - 0.5 * n * np.log(2 * np.pi)
    )


def maximise_evidence(phi, t, alpha, beta, iterations=500, tol=1e-12):
    n, _ = phi.shape
    eig = np.linalg.eigvalsh(phi.T @ phi)
    used = 0
    for used in range(1, iterations + 1):
        mean, _, _ = posterior(phi, t, alpha, beta)
        # Rescaled every sweep because these eigenvalues scale with beta, which the
        # iteration is itself changing.
        lam = beta * eig
        gamma = np.sum(lam / (alpha + lam))
        new_alpha = gamma / (mean @ mean)
        residual = t - phi @ mean
        new_beta = (n - gamma) / (residual @ residual)
        converged = abs(new_alpha - alpha) <= tol * abs(alpha) and abs(
            new_beta - beta
        ) <= tol * abs(beta)
        alpha, beta = new_alpha, new_beta
        if converged:
            break
    mean, _, _ = posterior(phi, t, alpha, beta)
    lam = beta * eig
    gamma = float(np.sum(lam / (alpha + lam)))
    return alpha, beta, gamma, used


cases = []

for degree in (0, 1, 3, 9):
    for bias in (True, False):
        for x in (0.0, 0.25, 0.7, 1.0):
            if not bias and degree == 0:
                continue
            row = polynomial_design(np.array([x]), degree, bias)[0]
            cases.append(
                {
                    "fn": "polynomialBasis",
                    "degree": degree,
                    "bias": bias,
                    "x": x,
                    "expected": row.tolist(),
                }
            )

for x in (0.0, 0.33, 0.9):
    cases.append(
        {
            "fn": "gaussianBasis",
            "centres": CENTRES.tolist(),
            "scale": SCALE,
            "bias": True,
            "x": x,
            "expected": gaussian_design(np.array([x]), CENTRES, SCALE)[0].tolist(),
        }
    )
    cases.append(
        {
            "fn": "sigmoidalBasis",
            "centres": CENTRES.tolist(),
            "scale": SCALE,
            "bias": True,
            "x": x,
            "expected": sigmoidal_design(np.array([x]), CENTRES, SCALE)[0].tolist(),
        }
    )

# Degree 9 on this many points is deliberately near-singular: it is the book's own
# over-fitting figure. cond(Phi) is 7.3e7 but cond(Phi^T Phi) is 5.3e15, at the edge of
# double precision, so the normal equations cannot determine these coefficients at all:
# two correct solvers disagree by thousands while producing the same residual. The
# unregularised fit must therefore go through an SVD least-squares solve on Phi directly,
# which never squares the condition number. The ridge cases below are fine either way,
# because lambda lifts the spectrum away from zero.
for degree in (1, 3, 9):
    phi = polynomial_design(X, degree)
    w, *_ = np.linalg.lstsq(phi, T, rcond=None)
    cases.append(
        {"fn": "maximumLikelihoodWeights", "degree": degree, "expected": w.tolist()}
    )
    cases.append(
        {
            "fn": "meanSquaredError",
            "degree": degree,
            "weights": w.tolist(),
            "expected": float(np.mean((phi @ w - T) ** 2)),
        }
    )
    for lam in (1e-6, 0.01, 1.0):
        wr = np.linalg.solve(lam * np.eye(phi.shape[1]) + phi.T @ phi, phi.T @ T)
        cases.append(
            {
                "fn": "regularisedWeights",
                "degree": degree,
                "lambda": lam,
                "expected": wr.tolist(),
            }
        )

for degree, alpha, beta in ((3, 2e-3, 25.0), (9, 5.0, 11.1), (1, 1.0, 1.0)):
    phi = polynomial_design(X, degree)
    mean, cov, precision = posterior(phi, T, alpha, beta)
    cases.append(
        {
            "fn": "weightPosterior",
            "degree": degree,
            "alpha": alpha,
            "beta": beta,
            "expected": {
                "mean": mean.tolist(),
                "cov": cov.tolist(),
                "precision": precision.tolist(),
            },
        }
    )
    for x in (0.0, 0.35, 1.0, 1.4):
        phi_x = polynomial_design(np.array([x]), degree)[0]
        cases.append(
            {
                "fn": "predictive",
                "degree": degree,
                "alpha": alpha,
                "beta": beta,
                "x": x,
                "expected": {
                    "mean": float(phi_x @ mean),
                    "variance": float(1.0 / beta + phi_x @ cov @ phi_x),
                },
            }
        )

# The sequential update must land exactly where a batch fit over the same points lands;
# that equality is the whole claim of the sequential-learning figure.
for prefix in (1, 2, 5, N):
    phi = polynomial_design(X[:prefix], 3)
    mean, cov, _ = posterior(phi, T[:prefix], 2e-3, 25.0)
    cases.append(
        {
            "fn": "updatePosterior",
            "degree": 3,
            "alpha": 2e-3,
            "beta": 25.0,
            "count": prefix,
            "expected": {"mean": mean.tolist(), "cov": cov.tolist()},
        }
    )

phi9 = gaussian_design(X, CENTRES, SCALE)
_, cov9, _ = posterior(phi9, T, 2e-3, 25.0)
kernel_rows = []
probe = np.linspace(0.0, 1.0, 11)
for xa in (0.25, 0.5, 0.75):
    phi_a = gaussian_design(np.array([xa]), CENTRES, SCALE)[0]
    row = [
        float(25.0 * phi_a @ cov9 @ gaussian_design(np.array([xb]), CENTRES, SCALE)[0])
        for xb in probe
    ]
    kernel_rows.append({"x": xa, "probe": probe.tolist(), "expected": row})
cases.append(
    {
        "fn": "equivalentKernel",
        "centres": CENTRES.tolist(),
        "scale": SCALE,
        "alpha": 2e-3,
        "beta": 25.0,
        "rows": kernel_rows,
    }
)

for degree in (0, 1, 2, 3, 5, 9):
    phi = polynomial_design(X, degree)
    cases.append(
        {
            "fn": "logEvidence",
            "degree": degree,
            "alpha": 5e-3,
            "beta": 11.1,
            "expected": float(log_evidence(phi, T, 5e-3, 11.1)),
        }
    )

for degree in (3, 9):
    phi = polynomial_design(X, degree)
    a, b, gamma, iters = maximise_evidence(phi, T, 1.0, 1.0)
    cases.append(
        {
            "fn": "maximiseEvidence",
            "degree": degree,
            "initialAlpha": 1.0,
            "initialBeta": 1.0,
            "expected": {
                "alpha": float(a),
                "beta": float(b),
                "effectiveParameters": gamma,
                "logEvidence": float(log_evidence(phi, T, a, b)),
            },
            "iterations": iters,
        }
    )

# Twelve datasets of the book's sinusoid, each fitted with a regularised gaussian basis and
# evaluated on a shared test grid: the ensemble of figure 3.5.
ens_rng = np.random.default_rng(31415)
test_x = np.linspace(0.0, 1.0, 40)
truth = np.sin(2 * np.pi * test_x)
for lam in (1e-4, 1.0, 100.0):
    preds = []
    for _ in range(12):
        xs = np.sort(ens_rng.uniform(0.0, 1.0, 25))
        ts = np.sin(2 * np.pi * xs) + ens_rng.normal(0.0, NOISE_SD, 25)
        phi = gaussian_design(xs, CENTRES, SCALE)
        w = np.linalg.solve(lam * np.eye(phi.shape[1]) + phi.T @ phi, phi.T @ ts)
        preds.append((gaussian_design(test_x, CENTRES, SCALE) @ w).tolist())
    P = np.array(preds)
    avg = P.mean(axis=0)
    bias2 = float(np.mean((avg - truth) ** 2))
    variance = float(np.mean(np.mean((P - avg[None, :]) ** 2, axis=0)))
    cases.append(
        {
            "fn": "biasVarianceDecomposition",
            "lambda": lam,
            "predictions": preds,
            "truth": truth.tolist(),
            "expected": {
                "bias2": bias2,
                "variance": variance,
                "total": bias2 + variance,
            },
        }
    )

payload = {
    "dataset": {"x": X.tolist(), "t": T.tolist(), "noiseSd": NOISE_SD},
    "centres": CENTRES.tolist(),
    "scale": SCALE,
    "cases": cases,
}

out = FIXTURES / "regression.json"
out.write_text(json.dumps(payload))
print(f"wrote {out} ({len(cases)} cases)")
