"""Golden fixtures for packages/math/src/latent/factorAnalysis.ts.

Factor analysis (PRML 12.2.4) replaces PPCA's isotropic noise sigma^2 I with a diagonal
Psi, so it is the natural contrast case: anisotropic per-axis noise that isotropic PPCA
cannot represent. There is no closed-form maximum-likelihood solution (unlike PPCA), so
only the EM iterate trace is golden-tested here, following the same
E-step-then-M-step-by-hand pattern as tools/golden/gen_ppca.py and gen_em.py.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260916)
n = 10
D = 3
M = 1
true_w = np.array([[1.0], [0.5], [-0.8]])
true_psi = np.array([0.05, 0.5, 0.1])
true_mean = np.array([0.2, -0.1, 0.3])
z_true = rng.normal(size=(n, M))
noise = rng.normal(size=(n, D)) * np.sqrt(true_psi)
data = z_true @ true_w.T + true_mean + noise
mean = data.mean(axis=0)


def e_step(X, mean, W, psi):
    psi_inv = np.diag(1.0 / psi)
    m = W.shape[1]
    G = np.linalg.inv(np.eye(m) + W.T @ psi_inv @ W)
    centered = X - mean
    ez = centered @ psi_inv @ W @ G.T
    ezz = np.array([G + np.outer(z, z) for z in ez])
    return ez, ezz


def m_step(X, mean, ez, ezz):
    centered = X - mean
    n_ = X.shape[0]
    d_ = X.shape[1]
    sum_xz = centered.T @ ez
    sum_zz = ezz.sum(axis=0)
    w_new = sum_xz @ np.linalg.inv(sum_zz)
    S = (centered.T @ centered) / n_
    psi_new = np.diag(S - (w_new @ (ez.T @ centered) / n_))
    return w_new, np.maximum(psi_new, 1e-8)


init_w = np.array([[0.5], [0.5], [0.5]])
init_psi = np.array([1.0, 1.0, 1.0])

w_cur, psi_cur = init_w.copy(), init_psi.copy()
trace = []
for _ in range(3):
    ez, ezz = e_step(data, mean, w_cur, psi_cur)
    w_cur, psi_cur = m_step(data, mean, ez, ezz)
    trace.append(
        {
            "ez": ez.tolist(),
            "wAfterMStep": w_cur.tolist(),
            "psiAfterMStep": psi_cur.tolist(),
        }
    )

C = true_w @ true_w.T + np.diag(true_psi)

cases = [
    {
        "fn": "faMarginalCov",
        "w": true_w.tolist(),
        "psi": true_psi.tolist(),
        "expected": C.tolist(),
    },
    {
        "fn": "faEmTrace",
        "data": data.tolist(),
        "mean": mean.tolist(),
        "initialW": init_w.tolist(),
        "initialPsi": init_psi.tolist(),
        "steps": 3,
        "expectedTrace": trace,
    },
]

(FIXTURES / "factorAnalysis.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'factorAnalysis.json'}")
