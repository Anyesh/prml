"""Golden fixtures for packages/math/src/ensemble/mixtureExperts.ts.

PRML 14.5.3, general K: a softmax gating network `pi(x) = softmax(V phi_gate(x))`
(4.104-4.105) selects among K linear regression experts sharing a noise precision `beta`.
The synthetic set below has three genuine regimes along one input dimension `x` (not two):
a distinct linear function generates the data on each of three intervals, same noise
standard deviation throughout, so the gate's job is to recover two boundaries and each of
three experts fits its own regime. K=3 is exercised end to end (not just K=2) because the
gate's identifiability fix (class 0 pinned to zero, PRML 4.104-4.109's redundancy) only
bites for K >= 3: at K=2 there is only one free row regardless of how it is fixed.

All reference numbers are computed independently of the eventual TypeScript:
responsibilities and log-likelihood directly from `scipy.special.softmax` and
`scipy.stats.norm.pdf`, the expert M-step via weighted normal equations, and the gate
M-step via a fresh Newton-Raphson loop written here (not imported from
gen_weightedSoftmax.py, though it uses the identical reference-class-zero convention and
ridge so the two are checked against the same regularised objective).
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import softmax as scipy_softmax
from scipy.stats import norm

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

RIDGE = 1e-10
rng = np.random.default_rng(20260920)


def design_matrix(x):
    return np.stack([np.ones_like(x), x], axis=1)


def gate_probabilities(phi_gate, gate_weights):
    return scipy_softmax(phi_gate @ gate_weights.T, axis=1)


def responsibilities_and_terms(
    phi_experts, phi_gate, t, expert_weights, beta, gate_weights
):
    k = expert_weights.shape[0]
    pis = gate_probabilities(phi_gate, gate_weights)
    means = np.stack([phi_experts @ expert_weights[c] for c in range(k)], axis=1)
    log_terms = np.log(pis) + norm.logpdf(
        t[:, None], loc=means, scale=1.0 / np.sqrt(beta)
    )
    m = log_terms.max(axis=1, keepdims=True)
    resp = np.exp(log_terms - m)
    resp = resp / resp.sum(axis=1, keepdims=True)
    ll = float(np.sum(m[:, 0] + np.log(np.sum(np.exp(log_terms - m), axis=1))))
    return resp, ll


def weighted_normal_equations(phi, t, w):
    wmat = np.diag(w)
    gram = phi.T @ wmat @ phi
    rhs = phi.T @ wmat @ t
    return np.linalg.solve(gram, rhs)


def gate_newton_raphson(phi, soft_targets, tol=1e-13, max_iter=200):
    """Fits a K-class softmax gate to `soft_targets` (N x K, rows summing to 1), pinning
    class 0's weights to zero, exactly as gen_weightedSoftmax.py and weightedSoftmax.ts do.
    """
    n, d = phi.shape
    k = soft_targets.shape[1]
    free = k - 1
    w = np.zeros((free, d))

    def full(w_free):
        return np.vstack([np.zeros((1, d)), w_free])

    def probs(w_free):
        return scipy_softmax(phi @ full(w_free).T, axis=1)

    for _ in range(max_iter):
        y = probs(w)
        gradient = np.zeros(free * d)
        hessian = np.zeros((free * d, free * d))
        for kk in range(1, k):
            fk = kk - 1
            gk = y[:, kk] - soft_targets[:, kk]
            gradient[fk * d : (fk + 1) * d] = phi.T @ gk
        for kk in range(1, k):
            fk = kk - 1
            for ll in range(1, k):
                fl = ll - 1
                indicator = 1.0 if kk == ll else 0.0
                curvature = y[:, kk] * (indicator - y[:, ll])
                hessian[fk * d : (fk + 1) * d, fl * d : (fl + 1) * d] = (
                    phi * curvature[:, None]
                ).T @ phi
        hessian = hessian + RIDGE * np.eye(free * d)
        step = np.linalg.solve(hessian, gradient)
        w_next = w - step.reshape(free, d)
        delta = np.sqrt(np.sum((w_next - w) ** 2))
        w = w_next
        if delta <= tol:
            break
    return full(w)


def experts_m_step(phi_experts, phi_gate, t, resp):
    k = resp.shape[1]
    weights = np.stack(
        [weighted_normal_equations(phi_experts, t, resp[:, c]) for c in range(k)]
    )
    means = np.stack([phi_experts @ weights[c] for c in range(k)], axis=1)
    weighted_sq_error = float(np.sum(resp * (t[:, None] - means) ** 2))
    beta = phi_experts.shape[0] / weighted_sq_error
    gate_weights = gate_newton_raphson(phi_gate, resp)
    return weights, beta, gate_weights


# Three regimes along x: rising, falling, rising again, each with the same noise std.
N = 60
NOISE_STD = 0.15
X = np.sort(rng.uniform(-3.0, 3.0, N))
TRUE_A = np.array([1.0, 0.9])  # x < -1
TRUE_B = np.array([-1.5, -0.7])  # -1 <= x < 1
TRUE_C = np.array([0.5, 1.1])  # x >= 1
MEAN = np.select(
    [X < -1.0, X < 1.0],
    [TRUE_A[0] + TRUE_A[1] * X, TRUE_B[0] + TRUE_B[1] * X],
    default=TRUE_C[0] + TRUE_C[1] * X,
)
T = MEAN + rng.normal(0.0, NOISE_STD, N)

PHI = design_matrix(X)
K = 3

cases = []

# Hand-chosen, plausible but not converged.
HAND_EXPERT_WEIGHTS = np.array([[0.8, 0.7], [-1.2, -0.5], [0.3, 0.9]])
HAND_BETA = 20.0
HAND_GATE_WEIGHTS = np.array([[0.0, 0.0], [1.0, 2.5], [1.5, -2.0]])
hand_params = {
    "expertWeights": HAND_EXPERT_WEIGHTS.tolist(),
    "beta": HAND_BETA,
    "gateWeights": HAND_GATE_WEIGHTS.tolist(),
}

hand_resp, hand_ll = responsibilities_and_terms(
    PHI, PHI, T, HAND_EXPERT_WEIGHTS, HAND_BETA, HAND_GATE_WEIGHTS
)

cases.append(
    {
        "fn": "mixtureExpertsResponsibilities",
        "designExperts": PHI.tolist(),
        "designGate": PHI.tolist(),
        "targets": T.tolist(),
        "params": hand_params,
        "expected": hand_resp.tolist(),
    }
)
cases.append(
    {
        "fn": "mixtureExpertsLogLikelihood",
        "designExperts": PHI.tolist(),
        "designGate": PHI.tolist(),
        "targets": T.tolist(),
        "params": hand_params,
        "expected": hand_ll,
    }
)

mstep_weights, mstep_beta, mstep_gate = experts_m_step(PHI, PHI, T, hand_resp)
cases.append(
    {
        "fn": "mixtureExpertsMStep",
        "designExperts": PHI.tolist(),
        "designGate": PHI.tolist(),
        "targets": T.tolist(),
        "responsibilities": hand_resp.tolist(),
        "expected": {
            "expertWeights": mstep_weights.tolist(),
            "beta": float(mstep_beta),
            "gateWeights": mstep_gate.tolist(),
        },
    }
)

# A short EM trace from a fresh initial guess, hand-rolled E-step-then-M-step, 4 rounds.
init_expert_weights = np.array([[0.0, 0.6], [0.0, -0.6], [0.0, 0.6]])
init_beta = 5.0
init_gate_weights = np.array([[0.0, 0.0], [0.5, 1.0], [0.5, -1.0]])

expert_weights, beta, gate_weights = (
    init_expert_weights.copy(),
    init_beta,
    init_gate_weights.copy(),
)
trace = []
for _ in range(4):
    resp, ll_before = responsibilities_and_terms(
        PHI, PHI, T, expert_weights, beta, gate_weights
    )
    expert_weights, beta, gate_weights = experts_m_step(PHI, PHI, T, resp)
    trace.append(
        {
            "responsibilities": resp.tolist(),
            "logLikelihoodBeforeMStep": ll_before,
            "paramsAfterMStep": {
                "expertWeights": expert_weights.tolist(),
                "beta": float(beta),
                "gateWeights": gate_weights.tolist(),
            },
        }
    )

cases.append(
    {
        "fn": "mixtureExpertsEmTrace",
        "designExperts": PHI.tolist(),
        "designGate": PHI.tolist(),
        "targets": T.tolist(),
        "initialParams": {
            "expertWeights": init_expert_weights.tolist(),
            "beta": init_beta,
            "gateWeights": init_gate_weights.tolist(),
        },
        "steps": 4,
        "expectedTrace": trace,
    }
)

fixture = {"cases": cases}
(FIXTURES / "mixtureExperts.json").write_text(json.dumps(fixture, indent=2))
print(f"wrote {len(cases)} cases to mixtureExperts.json")
