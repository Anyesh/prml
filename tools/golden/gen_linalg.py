"""Golden fixtures for packages/math/src/linalg/{core,decompose}.ts.

Follows the shape documented in tools/golden/README.md: {"cases": [...]}, each case
carrying an "fn" discriminant, whatever inputs that function needs, and an "expected"
value at full float64 precision.

svd() has no documented sign convention for its singular vectors (unlike eigSym,
which the source pins down explicitly), and distinct valid SVD implementations can
flip the sign of a (u_i, v_i) pair independently. Only the singular values and the
reconstruction/orthonormality properties are therefore golden-comparable; U and V
themselves are checked in the TypeScript test via round-trip, not elementwise.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

S = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]])  # SPD
B = np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]])  # general invertible
R = np.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0], [2.0, 1.0, 0.0]]
)  # 4x3
A_asym = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

x = np.array([1.0, -2.0, 0.5])
y = np.array([2.0, 1.0, -1.0])
b = np.array([1.0, 2.0, 3.0])
Bmat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


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


eig_vals, eig_vecs_rows = eigh_descending_sign_fixed(S)

L = np.linalg.cholesky(S)
logdet_S = np.linalg.slogdet(S)[1]

svd_u, svd_s, svd_vt = np.linalg.svd(R, full_matrices=False)
pinv_R = np.linalg.pinv(R)

cases = [
    {"fn": "matmul", "a": S.tolist(), "b": B.tolist(), "expected": (S @ B).tolist()},
    {"fn": "matvec", "a": S.tolist(), "x": x.tolist(), "expected": (S @ x).tolist()},
    {
        "fn": "quadForm",
        "x": x.tolist(),
        "a": S.tolist(),
        "y": y.tolist(),
        "expected": float(x @ S @ y),
    },
    {"fn": "matAdd", "a": S.tolist(), "b": B.tolist(), "expected": (S + B).tolist()},
    {"fn": "matSub", "a": S.tolist(), "b": B.tolist(), "expected": (S - B).tolist()},
    {"fn": "matScale", "a": S.tolist(), "s": 2.5, "expected": (S * 2.5).tolist()},
    {"fn": "vecAdd", "a": x.tolist(), "b": y.tolist(), "expected": (x + y).tolist()},
    {"fn": "vecSub", "a": x.tolist(), "b": y.tolist(), "expected": (x - y).tolist()},
    {"fn": "vecScale", "a": x.tolist(), "s": -1.5, "expected": (x * -1.5).tolist()},
    {"fn": "dot", "a": x.tolist(), "b": y.tolist(), "expected": float(x @ y)},
    {
        "fn": "outer",
        "a": x.tolist(),
        "b": y.tolist(),
        "expected": np.outer(x, y).tolist(),
    },
    {"fn": "norm", "a": x.tolist(), "expected": float(np.linalg.norm(x))},
    {"fn": "trace", "a": S.tolist(), "expected": float(np.trace(S))},
    {"fn": "transpose", "a": B.tolist(), "expected": B.T.tolist()},
    {"fn": "diag", "values": x.tolist(), "expected": np.diag(x).tolist()},
    {"fn": "diagOf", "a": S.tolist(), "expected": np.diag(S).tolist()},
    {
        "fn": "submatrix",
        "a": S.tolist(),
        "rows": [0, 2],
        "cols": [1, 2],
        "expected": S[np.ix_([0, 2], [1, 2])].tolist(),
    },
    {
        "fn": "subvector",
        "x": x.tolist(),
        "idx": [2, 0],
        "expected": x[[2, 0]].tolist(),
    },
    {
        "fn": "symmetrise",
        "a": A_asym.tolist(),
        "expected": ((A_asym + A_asym.T) / 2).tolist(),
    },
    {"fn": "cholesky", "a": S.tolist(), "expected": L.tolist()},
    {"fn": "logDet", "a": S.tolist(), "expected": float(logdet_S)},
    {"fn": "det", "a": B.tolist(), "expected": float(np.linalg.det(B))},
    {"fn": "inverse", "a": B.tolist(), "expected": np.linalg.inv(B).tolist()},
    {
        "fn": "solve",
        "a": B.tolist(),
        "b": b.tolist(),
        "expected": np.linalg.solve(B, b).tolist(),
    },
    {
        "fn": "solveMat",
        "a": B.tolist(),
        "b": Bmat.tolist(),
        "expected": np.linalg.solve(B, Bmat).tolist(),
    },
    {
        "fn": "solveCholesky",
        "l": L.tolist(),
        "b": b.tolist(),
        "expected": np.linalg.solve(S, b).tolist(),
    },
    {
        "fn": "eigSym",
        "a": S.tolist(),
        "expected": {"values": eig_vals.tolist(), "vectors": eig_vecs_rows.tolist()},
    },
    {
        "fn": "svd",
        "a": R.tolist(),
        "expected": {"s": svd_s.tolist()},
    },
    {
        "fn": "pinv",
        "a": R.tolist(),
        "expected": pinv_R.tolist(),
    },
]

(FIXTURES / "linalg.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {FIXTURES / 'linalg.json'}")
