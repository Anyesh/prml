"""Golden fixtures for packages/math/src/sequential/markov.ts.

Plain numpy: the log-likelihood of a path through a homogeneous first-order Markov
chain (PRML 13.2) is just a sum of logs, computed independently of the TypeScript
recursion.
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

pi = np.array([0.5, 0.3, 0.2])
A = np.array(
    [
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ]
)


def chain_log_likelihood(states):
    total = np.log(pi[states[0]])
    for t in range(1, len(states)):
        total += np.log(A[states[t - 1], states[t]])
    return float(total)


cases = []
for states in (
    [0],
    [0, 0, 0, 0],
    [0, 1, 2, 1, 0],
    [2, 2, 2, 0, 1, 1, 2],
):
    cases.append(
        {
            "fn": "markovChainLogLikelihood",
            "pi": pi.tolist(),
            "A": A.tolist(),
            "states": states,
            "expected": chain_log_likelihood(states),
        }
    )

# A path that uses a structurally zero transition (left-to-right style chain): log
# likelihood must come back as -inf, not NaN, and json has no token for -inf so this
# case is checked separately by asserting the value in TS rather than round-tripping it
# through the fixture. Left as a comment for anyone regenerating this file:
# left-to-right A = [[0.9,0.1,0],[0,0.9,0.1],[0,0,1]], states [2,0] -> log(0) = -inf.

(FIXTURES / "markov.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'markov.json'}")
