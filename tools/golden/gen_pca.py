"""Golden fixtures for packages/math/src/latent/pca.ts.

eigSym (packages/math/src/linalg/decompose.ts) returns eigenvalues descending with
eigenvectors as rows, sign-fixed so each row's largest-magnitude entry is positive
(see tools/golden/gen_linalg.py's `eigh_descending_sign_fixed`). pcaFitCov/pcaFitSvd
must agree with that same convention, so the same transform is applied here before
comparing, and a near-degenerate eigenvalue pair is included deliberately (PRML 12.1's
own two-point argument for why an eigenvector can be individually unstable even though
the subspace it spans is not).
"""

import json
from pathlib import Path

import numpy as np

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


# A well-separated 3D dataset: variance concentrated along two roughly orthogonal
# directions plus a small third, so the top-2 subspace is well defined.
rng = np.random.default_rng(20260916)
n = 60
basis = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.2], [0.1, 0.1, 1.0]])
basis /= np.linalg.norm(basis, axis=1, keepdims=True)
latent = rng.normal(size=(n, 3)) * np.array([3.0, 1.5, 0.3])
data = latent @ basis + np.array([2.0, -1.0, 0.5])

mean = data.mean(axis=0)
centered = data - mean
cov = (centered.T @ centered) / n
eigvals, eigvecs_rows = eigh_descending_sign_fixed(cov)

# SVD of the centred data: singular values s relate to eigenvalues by s^2 / N.
u, s, vt = np.linalg.svd(centered, full_matrices=False)
svd_eigvals = (s**2) / n
# Fix sign the same way (rows of V^T = right singular vectors as rows).
vt_rows = vt.copy()
for i in range(vt_rows.shape[0]):
    row = vt_rows[i]
    j = int(np.argmax(np.abs(row)))
    if row[j] < 0:
        vt_rows[i] = -row

degenerate_basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
degenerate_latent = rng.normal(size=(n, 3)) * np.array([2.0, 1.995, 0.05])
degenerate_data = degenerate_latent @ degenerate_basis
deg_mean = degenerate_data.mean(axis=0)
deg_centered = degenerate_data - deg_mean
deg_cov = (deg_centered.T @ deg_centered) / n
deg_eigvals, deg_eigvecs_rows = eigh_descending_sign_fixed(deg_cov)

M = 2
components_M = eigvecs_rows[:M]
scores = centered @ components_M.T
reconstruction = scores @ components_M + mean
recon_error = float(np.mean(np.sum((data - reconstruction) ** 2, axis=1)))
discarded_sum = float(np.sum(eigvals[M:]))

whitened = centered @ eigvecs_rows.T @ np.diag(1.0 / np.sqrt(eigvals))
whitened_cov = (whitened.T @ whitened) / n

cases = [
    {"fn": "dataMean", "data": data.tolist(), "expected": mean.tolist()},
    {
        "fn": "covarianceMatrix",
        "data": data.tolist(),
        "mean": mean.tolist(),
        "expected": cov.tolist(),
    },
    {
        "fn": "pcaFitCov",
        "data": data.tolist(),
        "expected": {
            "mean": mean.tolist(),
            "eigenvalues": eigvals.tolist(),
            "components": eigvecs_rows.tolist(),
        },
    },
    {
        "fn": "pcaFitSvd_eigenvalues",
        "data": data.tolist(),
        "expected": svd_eigvals.tolist(),
    },
    {
        "fn": "pcaFitCov_degenerate",
        "data": degenerate_data.tolist(),
        "expected": {
            "mean": deg_mean.tolist(),
            "eigenvalues": deg_eigvals.tolist(),
        },
        "note": "the two largest population variances (4.0 and 3.980025) are within 1% of each other by construction; at N=60 the sample eigenvalues separate far more than that, which is the instability this fixture exists to cover",
    },
    {
        "fn": "pcaProject",
        "data": data.tolist(),
        "mean": mean.tolist(),
        "components": components_M.tolist(),
        "numComponents": M,
        "expected": scores.tolist(),
    },
    {
        "fn": "pcaReconstruct",
        "scores": scores.tolist(),
        "mean": mean.tolist(),
        "components": components_M.tolist(),
        "expected": reconstruction.tolist(),
    },
    {
        "fn": "pcaReconstructionError",
        "data": data.tolist(),
        "mean": mean.tolist(),
        "components": components_M.tolist(),
        "numComponents": M,
        "expected": recon_error,
    },
    {
        "fn": "discardedEigenvalueSum",
        "eigenvalues": eigvals.tolist(),
        "numComponents": M,
        "expected": discarded_sum,
    },
    {
        "fn": "reconstructionErrorEqualsDiscardedSum",
        "expectedReconError": recon_error,
        "expectedDiscardedSum": discarded_sum,
    },
    {
        "fn": "whiten",
        "data": data.tolist(),
        "mean": mean.tolist(),
        "components": eigvecs_rows.tolist(),
        "eigenvalues": eigvals.tolist(),
        "expected": whitened.tolist(),
        "expectedCov": whitened_cov.tolist(),
    },
]

(FIXTURES / "pca.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'pca.json'}")
