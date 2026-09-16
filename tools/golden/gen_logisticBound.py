"""Golden fixtures for packages/math/src/variational/logisticBound.ts.

Covers the local variational bound on the logistic sigmoid (PRML 10.144-10.148) and its
use in variational logistic regression (PRML 10.151-10.164). Every quantity is derived
fresh in numpy from the closed forms; nothing here calls the TypeScript module under test.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lam(xi):
    xi = np.asarray(xi, dtype=float)
    out = np.where(np.abs(xi) < 1e-8, 0.125, (sigmoid(xi) - 0.5) / (2 * xi))
    return out


def local_bound(x, xi):
    return sigmoid(xi) * np.exp((x - xi) / 2 - lam(xi) * (x**2 - xi**2))


cases = []

xi_grid = [-6.0, -2.0, -0.5, 0.0, 0.001, 0.5, 2.0, 6.0]
cases.append(
    {"fn": "logisticLambda", "xi": xi_grid, "expected": lam(np.array(xi_grid)).tolist()}
)

x_grid = np.linspace(-8, 8, 25)
for xi in [0.5, 1.5, 3.0]:
    cases.append(
        {
            "fn": "logisticLocalBound",
            "x": x_grid.tolist(),
            "xi": xi,
            "expected": local_bound(x_grid, xi).tolist(),
        }
    )
    # The bound must touch the true sigmoid exactly at x = xi and x = -xi (PRML 10.144).
    cases.append(
        {
            "fn": "logisticLocalBound_touchPoints",
            "xi": xi,
            "x": [xi, -xi],
            "expected": [float(sigmoid(xi)), float(sigmoid(-xi))],
        }
    )

M = 3
N = 20
rng = np.random.default_rng(20260917)
design = rng.normal(size=(N, M))
design[:, 0] = 1.0
true_w = np.array([0.4, 1.1, -0.8])
probs = sigmoid(design @ true_w)
targets = (rng.uniform(size=N) < probs).astype(float)

m0 = np.zeros(M)
S0 = 100.0 * np.eye(M)
S0inv = np.linalg.inv(S0)


def q_w(design, targets, S0inv, m0, xi):
    l = lam(xi)
    precision = S0inv + 2 * (design * l[:, None]).T @ design
    S = np.linalg.inv(precision)
    rhs = S0inv @ m0 + (design * (targets - 0.5)[:, None]).sum(axis=0)
    m = S @ rhs
    return m, S


def update_xi(design, m, S):
    quad = np.einsum("ni,ij,nj->n", design, S, design) + (design @ m) ** 2
    return np.sqrt(quad)


def lower_bound(design, targets, S0inv, m0, S0, m, S, xi):
    n = len(targets)
    sign0, logdet0 = np.linalg.slogdet(S0)
    signN, logdetN = np.linalg.slogdet(S)
    quad_m = m @ np.linalg.solve(S, m)
    quad_m0 = m0 @ S0inv @ m0
    total = 0.5 * (logdetN - logdet0) + 0.5 * quad_m - 0.5 * quad_m0
    l = lam(xi)
    total += float(np.sum(np.log(sigmoid(xi)) - xi / 2 + l * xi**2))
    return float(total)


xi0 = np.full(N, 0.7)
m1, S1 = q_w(design, targets, S0inv, m0, xi0)
xi1 = update_xi(design, m1, S1)

cases.append(
    {
        "fn": "variationalLogisticUpdate",
        "design": design.tolist(),
        "targets": targets.tolist(),
        "prior": {"mean": m0.tolist(), "cov": S0.tolist()},
        "xi": xi0.tolist(),
        "expected": {"mean": m1.tolist(), "cov": S1.tolist()},
    }
)
cases.append(
    {
        "fn": "updateXi",
        "design": design.tolist(),
        "mean": m1.tolist(),
        "cov": S1.tolist(),
        "expected": xi1.tolist(),
    }
)

lb1 = lower_bound(design, targets, S0inv, m0, S0, m1, S1, xi1)
cases.append(
    {
        "fn": "variationalLogisticLowerBound",
        "design": design.tolist(),
        "targets": targets.tolist(),
        "prior": {"mean": m0.tolist(), "cov": S0.tolist()},
        "posterior": {"mean": m1.tolist(), "cov": S1.tolist()},
        "xi": xi1.tolist(),
        "expected": lb1,
    }
)

# Full fit trace: alternate q(w) and xi updates for several rounds, from a fixed xi0.
xi = xi0.copy()
trace = []
for _ in range(5):
    m, S = q_w(design, targets, S0inv, m0, xi)
    xi = update_xi(design, m, S)
    trace.append(
        {
            "mean": m.tolist(),
            "cov": S.tolist(),
            "xi": xi.tolist(),
            "lowerBound": lower_bound(design, targets, S0inv, m0, S0, m, S, xi),
        }
    )

cases.append(
    {
        "fn": "variationalLogisticFit",
        "design": design.tolist(),
        "targets": targets.tolist(),
        "prior": {"mean": m0.tolist(), "cov": S0.tolist()},
        "xiInit": xi0.tolist(),
        "rounds": 5,
        "expectedTrace": trace,
    }
)


# updateLogisticAlpha: the evidence-style point-estimate re-estimation of an isotropic
# prior precision from a converged q(w), gamma = M - alpha * Tr(SN), alpha_new = gamma / (mN^T mN)
# (PRML's evidence-approximation form, eq 3.92, applied here to the variational posterior).
posterior_for_alpha = {"mean": trace[-1]["mean"], "cov": trace[-1]["cov"]}
alpha_current = 2.0
mean_arr = np.array(posterior_for_alpha["mean"])
cov_arr = np.array(posterior_for_alpha["cov"])
trace_term = float(np.trace(cov_arr))
gamma = len(mean_arr) - alpha_current * trace_term
wtw = float(mean_arr @ mean_arr)
alpha_new = gamma / wtw

cases.append(
    {
        "fn": "updateLogisticAlpha",
        "posterior": posterior_for_alpha,
        "alpha": alpha_current,
        "expected": alpha_new,
    }
)

(FIXTURES / "logisticBound.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'logisticBound.json'}")
