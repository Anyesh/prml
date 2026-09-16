"""Golden fixtures for packages/math/src/latent/ppca.ts.

Covers PRML 12.2: the marginal N(x | mu, C) with C = W W^T + sigma^2 I (12.31-12.36),
the latent posterior (12.41-12.42), the closed-form maximum-likelihood solution
(12.45-12.46, with the arbitrary rotation R fixed to the identity, a convention this
module documents rather than hides), and one full hand-rolled EM iterate trace
(12.54-12.57) so ppcaEStep/ppcaMStep/ppcaFitEM can be checked iterate-by-iterate,
exactly as tools/golden/gen_em.py does for the Gaussian-mixture EM in chapter 9.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def eigh_descending_sign_fixed(a: np.ndarray):
    vals, vecs = np.linalg.eigh(a)
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    rows = vecs.T.copy()
    for i in range(rows.shape[0]):
        row = rows[i]
        j = int(np.argmax(np.abs(row)))
        if row[j] < 0:
            rows[i] = -row
    return vals, rows


rng = np.random.default_rng(20260916)
n = 80
D = 4
M = 2
true_w = rng.normal(size=(D, M))
true_sigma2 = 0.3
true_mean = np.array([1.0, -0.5, 0.2, 0.0])
z_true = rng.normal(size=(n, M))
noise = rng.normal(size=(n, D)) * np.sqrt(true_sigma2)
data = z_true @ true_w.T + true_mean + noise
data_list = data.tolist()

mean = data.mean(axis=0)
centered = data - mean
cov = (centered.T @ centered) / n
eigvals, eigvecs_rows = eigh_descending_sign_fixed(cov)

w_fixed = np.array([[1.0, 0.3], [0.2, -0.8], [0.5, 0.1], [-0.3, 0.4]])
sigma2_fixed = 0.2
C = w_fixed @ w_fixed.T + sigma2_fixed * np.eye(D)
x_query = np.array([0.5, -0.2, 0.1, 0.3])
marginal_logpdf = float(multivariate_normal.logpdf(x_query, mean=mean, cov=C))

Mmat = w_fixed.T @ w_fixed + sigma2_fixed * np.eye(M)
Minv = np.linalg.inv(Mmat)
post_mean = Minv @ w_fixed.T @ (x_query - mean)
post_cov = sigma2_fixed * Minv

lambda_m = eigvals[:M]
u_m = eigvecs_rows[:M]
sigma2_ml = float(np.mean(eigvals[M:]))
w_ml = u_m.T * np.sqrt(np.maximum(lambda_m - sigma2_ml, 0.0))

n_small = 12
data_small = data[:n_small]
mean_small = data_small.mean(axis=0)


def e_step(X, mean, W, sigma2):
    Mm = W.T @ W + sigma2 * np.eye(W.shape[1])
    Minv_ = np.linalg.inv(Mm)
    centered_ = X - mean
    ez = centered_ @ W @ Minv_.T
    ezz = []
    for i in range(X.shape[0]):
        ezz.append(sigma2 * Minv_ + np.outer(ez[i], ez[i]))
    return ez, np.array(ezz)


def m_step(X, mean, ez, ezz):
    centered_ = X - mean
    n_ = X.shape[0]
    sum_xz = centered_.T @ ez
    sum_zz = ezz.sum(axis=0)
    w_new = sum_xz @ np.linalg.inv(sum_zz)
    total = 0.0
    for i in range(n_):
        xi = centered_[i]
        total += xi @ xi
        total -= 2 * ez[i] @ w_new.T @ xi
        total += np.trace(ezz[i] @ w_new.T @ w_new)
    sigma2_new = total / (n_ * X.shape[1])
    return w_new, float(sigma2_new)


def marginal_log_likelihood(X, mean, W, sigma2):
    C_ = W @ W.T + sigma2 * np.eye(W.shape[0])
    total = 0.0
    for x in X:
        total += multivariate_normal.logpdf(x, mean=mean, cov=C_)
    return float(total)


init_w = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.0], [0.0, 0.5]])
init_sigma2 = 1.0

w_cur, sigma2_cur = init_w.copy(), init_sigma2
trace = []
for _ in range(4):
    ez, ezz = e_step(data_small, mean_small, w_cur, sigma2_cur)
    ll_before = marginal_log_likelihood(data_small, mean_small, w_cur, sigma2_cur)
    w_cur, sigma2_cur = m_step(data_small, mean_small, ez, ezz)
    trace.append(
        {
            "ez": ez.tolist(),
            "logLikelihoodBeforeMStep": ll_before,
            "wAfterMStep": w_cur.tolist(),
            "sigma2AfterMStep": sigma2_cur,
        }
    )

cases = [
    {
        "fn": "ppcaMarginalLogPdf",
        "x": x_query.tolist(),
        "mean": mean.tolist(),
        "w": w_fixed.tolist(),
        "sigma2": sigma2_fixed,
        "expected": marginal_logpdf,
    },
    {
        "fn": "ppcaLatentPosterior",
        "x": x_query.tolist(),
        "mean": mean.tolist(),
        "w": w_fixed.tolist(),
        "sigma2": sigma2_fixed,
        "expected": {"mean": post_mean.tolist(), "cov": post_cov.tolist()},
    },
    {
        "fn": "ppcaMLE",
        "data": data.tolist(),
        "latentDim": M,
        "expected": {
            "mean": mean.tolist(),
            "w": w_ml.tolist(),
            "sigma2": sigma2_ml,
        },
        "note": "R fixed to the identity; W is only comparable up to right-multiplication by an orthogonal matrix",
    },
    {
        "fn": "ppcaEmTrace",
        "data": data_small.tolist(),
        "mean": mean_small.tolist(),
        "initialW": init_w.tolist(),
        "initialSigma2": init_sigma2,
        "steps": 4,
        "expectedTrace": trace,
    },
]

(FIXTURES / "ppca.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'ppca.json'}")
