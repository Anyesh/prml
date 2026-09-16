"""Golden fixtures for packages/math/src/sequential/viterbi.ts.

Plain numpy max-sum in log space (PRML 13.68-13.71), computed directly from the
recursion's definition rather than by re-deriving the TypeScript implementation.
scipy.stats.multivariate_normal supplies the Gaussian emission matrix, as in
gen_hmm_forward_backward.py.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

K = 3
pi = np.array([0.5, 0.3, 0.2])
A = np.array(
    [
        [0.85, 0.1, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.15, 0.8],
    ]
)
means = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
covs = np.array([np.eye(2) * 0.6, np.eye(2) * 0.6, np.eye(2) * 0.6])

rng = np.random.default_rng(20260917)
# A path that visibly switches states, so the Viterbi path is a nontrivial sequence
# rather than one state repeated throughout.
true_states = [0, 0, 0, 1, 1, 1, 2, 2, 0, 0]
data = np.array([rng.multivariate_normal(means[s], covs[s]) for s in true_states])
N = len(true_states)


def emission_matrix(x):
    return np.array([multivariate_normal.pdf(x, means[k], covs[k]) for k in range(K)]).T


B = emission_matrix(data)
logB = np.log(B)
logA = np.log(A)
logPi = np.log(pi)


def viterbi():
    omega = np.zeros((N, K))
    psi = np.full((N, K), -1, dtype=int)
    omega[0] = logPi + logB[0]
    for n in range(1, N):
        for k in range(K):
            scores = omega[n - 1] + logA[:, k]
            j = int(np.argmax(scores))
            psi[n, k] = j
            omega[n, k] = logB[n, k] + scores[j]
    final_state = int(np.argmax(omega[N - 1]))
    log_prob = float(omega[N - 1, final_state])
    path = [0] * N
    path[N - 1] = final_state
    for n in range(N - 2, -1, -1):
        path[n] = int(psi[n + 1, path[n + 1]])
    return omega, psi, path, log_prob


omega, psi, path, log_prob = viterbi()

# Sanity: the recovered path should recover most of the ground-truth generating states,
# not merely run without error.
agreement = sum(1 for a, b in zip(path, true_states) if a == b) / N
assert agreement >= 0.7, (
    f"Viterbi path barely resembles the generating states ({agreement:.2f})"
)

cases = [
    {
        "fn": "hmmViterbi",
        "pi": pi.tolist(),
        "A": A.tolist(),
        "B": B.tolist(),
        "expected": {
            "omega": omega.tolist(),
            "psi": psi.tolist(),
            "path": path,
            "logProb": log_prob,
        },
    }
]

(FIXTURES / "hmm_viterbi.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'hmm_viterbi.json'}")
