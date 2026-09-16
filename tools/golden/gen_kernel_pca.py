"""Golden fixtures for packages/math/src/latent/kernelPca.ts.

PRML 12.3: the feature-space eigenproblem reduces to `K_tilde a_i = lambda_i N a_i`
(12.80) on the centred Gram matrix (12.83-12.85), normalised so `lambda_i N a_i^T a_i = 1`
(12.81), with the projection of a point onto component `i` given by `y_i(x) = sum_n
a_in k(x, x_n)` (12.82). Data here is two concentric rings, the standard case where
kernel PCA separates structure that linear PCA (a single scalar projection) cannot.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260916)
n_per = 10
theta1 = rng.uniform(0, 2 * np.pi, n_per)
theta2 = rng.uniform(0, 2 * np.pi, n_per)
inner = np.stack([np.cos(theta1), np.sin(theta1)], axis=1) * 1.0
outer = np.stack([np.cos(theta2), np.sin(theta2)], axis=1) * 3.0
data = np.concatenate([inner, outer]) + rng.normal(scale=0.05, size=(2 * n_per, 2))
n = data.shape[0]

gamma = 10.0


def rbf(a, b):
    diff = a - b
    return np.exp(-gamma * float(diff @ diff))


K = np.array([[rbf(data[i], data[j]) for j in range(n)] for i in range(n)])

ones_n = np.full((n, n), 1.0 / n)
K_centered = K - ones_n @ K - K @ ones_n + ones_n @ K @ ones_n

eigvals, eigvecs = np.linalg.eigh(K_centered)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

num_components = 3
kept_vals = []
kept_vecs = []
for i in range(len(eigvals)):
    if eigvals[i] > 1e-8 and len(kept_vals) < num_components:
        v = eigvecs[:, i]
        norm_factor = 1.0 / np.sqrt(eigvals[i] * n * (v @ v))
        v_normed = v * norm_factor
        j = int(np.argmax(np.abs(v_normed)))
        if v_normed[j] < 0:
            v_normed = -v_normed
        kept_vals.append(float(eigvals[i]))
        kept_vecs.append(v_normed)

alphas = np.array(kept_vecs)

query = data[3]
row_means = K.mean(axis=1)
grand_mean = float(K.mean())
k_query = np.array([rbf(query, data[j]) for j in range(n)])
k_query_centered = k_query - row_means - k_query.mean() + grand_mean
projection_query = alphas @ k_query_centered

train_projections = alphas @ K_centered

cases = [
    {
        "fn": "gramMatrix",
        "data": data.tolist(),
        "gamma": gamma,
        "expected": K.tolist(),
    },
    {
        "fn": "centerGramMatrix",
        "k": K.tolist(),
        "expected": K_centered.tolist(),
    },
    {
        "fn": "kernelPcaFit",
        "data": data.tolist(),
        "gamma": gamma,
        "numComponents": num_components,
        "expected": {
            "eigenvalues": kept_vals,
            "alphas": alphas.tolist(),
        },
    },
    {
        "fn": "kernelPcaTrainingProjections",
        "data": data.tolist(),
        "gamma": gamma,
        "numComponents": num_components,
        "expected": train_projections.T.tolist(),
    },
    {
        "fn": "kernelPcaProject",
        "data": data.tolist(),
        "gamma": gamma,
        "numComponents": num_components,
        "query": query.tolist(),
        "expected": projection_query.tolist(),
    },
]

(FIXTURES / "kernelPca.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'kernelPca.json'}")
