"""Golden fixtures for packages/math/src/neural/.

Follows tools/golden/README.md's shape: {"cases": [...]}. Two independent references are
used, deliberately never the same algebra as the TypeScript being tested:

- Backpropagation (PRML 5.53-5.57) is checked against a central difference of the error
  function itself (5.69), not against a from-scratch numpy backprop, because a finite
  difference exercises the whole forward-and-backward algorithm as a black box, where an
  independent analytic re-derivation would only prove the algebra agrees with itself.
- The exact Hessian (5.4.5) is checked against a central difference of an independently
  written numpy analytic gradient, which is 5.91's own suggested method ("central
  differences on the first derivatives... gives the Hessian in O(W^2)"), while the
  TypeScript computes it instead via Pearlmutter's R-operator (5.96-5.111): a genuinely
  different algorithm arriving at the same matrix.

At the optimal step size for a float64 central difference (h scaled near the cube root of
machine epsilon), both references land within 1e-10 to 1e-11 of the true value, inside
this repo's 1e-9 assertion tolerance; the diagnostic fields below report the actual gap
measured on this run rather than assuming the error-order formula holds.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

cases = []
rng = np.random.default_rng(20260916)

EPS_SCALE = 1e-5


def make_weights(sizes):
    return [
        rng.normal(scale=0.6, size=(sizes[l + 1], sizes[l] + 1))
        for l in range(len(sizes) - 1)
    ]


def weights_to_json(weights):
    return [w.tolist() for w in weights]


def output_activation(a, kind):
    if kind == "linear":
        return a
    if kind == "logistic":
        return 1.0 / (1.0 + np.exp(-a))
    return softmax(a)


def forward(x, weights, output_kind):
    z = np.asarray(x, dtype=float)
    zs = [z]
    as_ = [None]
    for l, w in enumerate(weights):
        aug = np.concatenate([[1.0], z])
        a = w @ aug
        is_output = l == len(weights) - 1
        z = output_activation(a, output_kind) if is_output else np.tanh(a)
        as_.append(a)
        zs.append(z)
    return zs, as_


def sum_squared_error(y, t):
    return 0.5 * float(np.sum((y - t) ** 2))


def cross_entropy_binary_error(a_out, t):
    return float(np.sum(np.logaddexp(0.0, a_out) - t * a_out))


def cross_entropy_softmax_error(y, t):
    return float(-np.sum(t * np.log(np.clip(y, 1e-300, None))))


def error_of(zs, as_, t, kind):
    if kind == "sumSquared":
        return sum_squared_error(zs[-1], t)
    if kind == "crossEntropyBinary":
        return cross_entropy_binary_error(as_[-1], t)
    return cross_entropy_softmax_error(zs[-1], t)


OUTPUT_FOR_KIND = {
    "sumSquared": "linear",
    "crossEntropyBinary": "logistic",
    "crossEntropySoftmax": "softmax",
}


def flatten(weights):
    return np.concatenate([w.flatten() for w in weights])


def unflatten(flat, sizes):
    out = []
    cursor = 0
    for l in range(len(sizes) - 1):
        n_out, n_in = sizes[l + 1], sizes[l] + 1
        out.append(flat[cursor : cursor + n_out * n_in].reshape(n_out, n_in))
        cursor += n_out * n_in
    return out


def central_diff_grad(f, x0, eps_scale=EPS_SCALE):
    grad = np.zeros_like(x0, dtype=float)
    for i in range(len(x0)):
        eps = eps_scale * max(1.0, abs(x0[i]))
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


def backprop_deltas(weights, zs, terminal_delta):
    deltas = [None] * (len(weights) + 1)
    deltas[-1] = terminal_delta
    for l in range(len(weights) - 1, 0, -1):
        z = zs[l]
        w_next = weights[l]
        d_next = deltas[l + 1]
        back_sum = w_next[:, 1:].T @ d_next
        deltas[l] = (1 - z**2) * back_sum
    return deltas


def gradient_from_deltas(weights, zs, deltas):
    grad = []
    for l in range(len(weights)):
        z_prev_aug = np.concatenate([[1.0], zs[l]])
        grad.append(np.outer(deltas[l + 1], z_prev_aug))
    return grad


def analytic_gradient(weights, x, t, kind):
    zs, _ = forward(x, weights, OUTPUT_FOR_KIND[kind])
    terminal = zs[-1] - t
    deltas = backprop_deltas(weights, zs, terminal)
    return gradient_from_deltas(weights, zs, deltas)


def output_weight_gradient(weights, x, output_kind):
    zs, _ = forward(x, weights, output_kind)
    deltas = backprop_deltas(weights, zs, np.array([1.0]))
    return flatten(gradient_from_deltas(weights, zs, deltas))


SIZES_REG = [2, 3, 1]
W_REG = make_weights(SIZES_REG)
SIZES_CLS = [2, 3, 1]
W_CLS = make_weights(SIZES_CLS)
SIZES_MULTI = [2, 4, 3]
W_MULTI = make_weights(SIZES_MULTI)

X_PROBE = np.array([0.4, -0.7])
T_REG = np.array([0.85])
T_CLS = np.array([1.0])
T_MULTI = np.array([0.0, 1.0, 0.0])

DATASET_INPUTS = [
    np.array([0.4, -0.7]),
    np.array([-0.3, 0.9]),
    np.array([1.1, 0.2]),
    np.array([-0.8, -0.5]),
]
DATASET_TARGETS_REG = [
    np.array([0.85]),
    np.array([-0.2]),
    np.array([0.5]),
    np.array([-0.9]),
]
DATASET_TARGETS_CLS = [
    np.array([1.0]),
    np.array([0.0]),
    np.array([1.0]),
    np.array([0.0]),
]
DATASET_TARGETS_MULTI = [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 0.0]),
]


def spec_json(sizes, output_kind):
    return {
        "layerSizes": sizes,
        "hiddenActivation": "tanh",
        "outputActivation": output_kind,
    }


NETWORKS = [
    ("sumSquared", SIZES_REG, W_REG, X_PROBE, T_REG),
    ("crossEntropyBinary", SIZES_CLS, W_CLS, X_PROBE, T_CLS),
    ("crossEntropySoftmax", SIZES_MULTI, W_MULTI, X_PROBE, T_MULTI),
]

for kind, sizes, weights, x, _t in NETWORKS:
    zs, as_ = forward(x, weights, OUTPUT_FOR_KIND[kind])
    cases.append(
        {
            "fn": "forwardPass",
            "spec": spec_json(sizes, OUTPUT_FOR_KIND[kind]),
            "weights": weights_to_json(weights),
            "input": x.tolist(),
            "expected": {
                "activations": [z.tolist() for z in zs],
                "preActivations": [([] if a is None else a.tolist()) for a in as_],
                "output": zs[-1].tolist(),
            },
        }
    )

probe_pred = np.array([0.3, -0.6, 0.9])
probe_target = np.array([0.1, -0.4, 1.2])
cases.append(
    {
        "fn": "sumSquaredErrorSingle",
        "predicted": probe_pred.tolist(),
        "target": probe_target.tolist(),
        "expected": sum_squared_error(probe_pred, probe_target),
    }
)

probe_a = np.array([1.4, -2.1, 0.05])
probe_t_bin = np.array([1.0, 0.0, 1.0])
cases.append(
    {
        "fn": "crossEntropyBinarySingleFromActivations",
        "preActivations": probe_a.tolist(),
        "target": probe_t_bin.tolist(),
        "expected": cross_entropy_binary_error(probe_a, probe_t_bin),
    }
)

probe_y = softmax(np.array([1.2, -0.3, 0.5]))
probe_t_multi = np.array([0.0, 1.0, 0.0])
cases.append(
    {
        "fn": "crossEntropySoftmaxSingle",
        "predicted": probe_y.tolist(),
        "target": probe_t_multi.tolist(),
        "expected": cross_entropy_softmax_error(probe_y, probe_t_multi),
    }
)

DATASET_TARGETS = {
    "sumSquared": DATASET_TARGETS_REG,
    "crossEntropyBinary": DATASET_TARGETS_CLS,
    "crossEntropySoftmax": DATASET_TARGETS_MULTI,
}

for kind, sizes, weights, _x, _t in NETWORKS:
    total = 0.0
    for xi, ti in zip(DATASET_INPUTS, DATASET_TARGETS[kind]):
        zs, as_ = forward(xi, weights, OUTPUT_FOR_KIND[kind])
        total += error_of(zs, as_, ti, kind)
    cases.append(
        {
            "fn": "networkError",
            "spec": spec_json(sizes, OUTPUT_FOR_KIND[kind]),
            "weights": weights_to_json(weights),
            "inputs": [xi.tolist() for xi in DATASET_INPUTS],
            "targets": [ti.tolist() for ti in DATASET_TARGETS[kind]],
            "kind": kind,
            "expected": total,
        }
    )

for kind, sizes, weights, x, t in NETWORKS:
    flat0 = flatten(weights)

    def error_at(flat, sizes=sizes, x=x, t=t, kind=kind):
        ws = unflatten(flat, sizes)
        zs, as_ = forward(x, ws, OUTPUT_FOR_KIND[kind])
        return error_of(zs, as_, t, kind)

    expected_grad = central_diff_grad(error_at, flat0)
    analytic_grad_flat = flatten(analytic_gradient(weights, x, t, kind))
    max_gap = float(
        np.max(
            np.abs(expected_grad - analytic_grad_flat)
            / np.maximum(1.0, np.abs(analytic_grad_flat))
        )
    )
    cases.append(
        {
            "fn": "backpropGradientSingle",
            "spec": spec_json(sizes, OUTPUT_FOR_KIND[kind]),
            "weights": weights_to_json(weights),
            "input": x.tolist(),
            "target": t.tolist(),
            "kind": kind,
            "expected": expected_grad.tolist(),
            "_diagnostic_central_diff_vs_analytic_max_relative_gap": max_gap,
        }
    )

for kind, sizes, weights, _x, _t in [NETWORKS[0], NETWORKS[1]]:
    flat0 = flatten(weights)
    targets = DATASET_TARGETS[kind]

    def batch_error_at(flat, sizes=sizes, kind=kind, targets=targets):
        ws = unflatten(flat, sizes)
        total = 0.0
        for xi, ti in zip(DATASET_INPUTS, targets):
            zs, as_ = forward(xi, ws, OUTPUT_FOR_KIND[kind])
            total += error_of(zs, as_, ti, kind)
        return total

    expected_grad = central_diff_grad(batch_error_at, flat0)
    cases.append(
        {
            "fn": "backpropGradientBatch",
            "spec": spec_json(sizes, OUTPUT_FOR_KIND[kind]),
            "weights": weights_to_json(weights),
            "inputs": [xi.tolist() for xi in DATASET_INPUTS],
            "targets": [ti.tolist() for ti in targets],
            "kind": kind,
            "expected": expected_grad.tolist(),
        }
    )


def probe_fn(x):
    return np.sin(x[0]) + x[0] * x[1] ** 2 + np.exp(x[2]) + x[1] * x[3]


x0 = np.array([0.6, -0.3, 0.2, 1.1])
expected_numgrad = central_diff_grad(probe_fn, x0)
analytic_probe_grad = np.array(
    [np.cos(x0[0]) + x0[1] ** 2, 2 * x0[0] * x0[1] + x0[3], np.exp(x0[2]), x0[1]]
)
cases.append(
    {
        "fn": "numericalGradient",
        "x0": x0.tolist(),
        "epsScale": EPS_SCALE,
        "expected": expected_numgrad.tolist(),
        "_diagnostic_vs_true_analytic_max_abs_gap": float(
            np.max(np.abs(expected_numgrad - analytic_probe_grad))
        ),
    }
)

flat_decay = flatten(W_REG)
lam = 0.3
cases.append(
    {
        "fn": "weightDecayPenalty",
        "flatWeights": flat_decay.tolist(),
        "lambda": lam,
        "expected": float(0.5 * lam * np.sum(flat_decay**2)),
    }
)
cases.append(
    {
        "fn": "weightDecayGradient",
        "flatWeights": flat_decay.tolist(),
        "lambda": lam,
        "expected": (lam * flat_decay).tolist(),
    }
)

for kind, sizes, weights in [
    ("sumSquared", SIZES_REG, W_REG),
    ("crossEntropyBinary", SIZES_CLS, W_CLS),
]:
    w_count = len(flatten(weights))
    h = np.zeros((w_count, w_count))
    for xi in DATASET_INPUTS:
        b = output_weight_gradient(weights, xi, OUTPUT_FOR_KIND[kind])
        if kind == "sumSquared":
            scale = 1.0
        else:
            zs, _ = forward(xi, weights, OUTPUT_FOR_KIND[kind])
            y0 = float(zs[-1][0])
            scale = y0 * (1 - y0)
        h += scale * np.outer(b, b)
    cases.append(
        {
            "fn": "outerProductHessian",
            "spec": spec_json(sizes, OUTPUT_FOR_KIND[kind]),
            "weights": weights_to_json(weights),
            "inputs": [xi.tolist() for xi in DATASET_INPUTS],
            "kind": kind,
            "expected": h.tolist(),
        }
    )

for kind, sizes, weights in [
    ("sumSquared", SIZES_REG, W_REG),
    ("crossEntropyBinary", SIZES_CLS, W_CLS),
]:
    flat0 = flatten(weights)
    w_count = len(flat0)
    targets_for = DATASET_TARGETS[kind][:2]
    inputs_for = DATASET_INPUTS[:2]

    def grad_at(
        flat, sizes=sizes, kind=kind, inputs_for=inputs_for, targets_for=targets_for
    ):
        ws = unflatten(flat, sizes)
        total = np.zeros_like(flat)
        for xi, ti in zip(inputs_for, targets_for):
            total += flatten(analytic_gradient(ws, xi, ti, kind))
        return total

    h = np.zeros((w_count, w_count))
    for b in range(w_count):
        eps = EPS_SCALE * max(1.0, abs(flat0[b]))
        fp, fm = flat0.copy(), flat0.copy()
        fp[b] += eps
        fm[b] -= eps
        h[:, b] = (grad_at(fp) - grad_at(fm)) / (2 * eps)
    h = 0.5 * (h + h.T)
    cases.append(
        {
            "fn": "exactHessian",
            "spec": spec_json(sizes, OUTPUT_FOR_KIND[kind]),
            "weights": weights_to_json(weights),
            "inputs": [xi.tolist() for xi in inputs_for],
            "targets": [ti.tolist() for ti in targets_for],
            "kind": kind,
            "expected": h.tolist(),
        }
    )

K, L = 3, 1
raw_mdn = rng.normal(scale=0.8, size=K * (2 + L))
t_mdn = np.array([0.6])


def mdn_params(raw, k=K, l=L):
    mixing = softmax(raw[:k])
    sigma = np.exp(raw[k : 2 * k])
    means = raw[2 * k : 2 * k + k * l].reshape(k, l)
    return mixing, sigma, means


def mdn_component_log_pdf(t, mean, sigma, l):
    return (
        -0.5 * l * np.log(2 * np.pi)
        - l * np.log(sigma)
        - float(np.sum((t - mean) ** 2)) / (2 * sigma**2)
    )


def mdn_error(raw, t, k=K, l=L):
    mixing, sigma, means = mdn_params(raw, k, l)
    log_terms = np.array(
        [
            np.log(mixing[j]) + mdn_component_log_pdf(t, means[j], sigma[j], l)
            for j in range(k)
        ]
    )
    return -logsumexp(log_terms)


def mdn_responsibilities(raw, t, k=K, l=L):
    mixing, sigma, means = mdn_params(raw, k, l)
    log_terms = np.array(
        [
            np.log(mixing[j]) + mdn_component_log_pdf(t, means[j], sigma[j], l)
            for j in range(k)
        ]
    )
    return np.exp(log_terms - logsumexp(log_terms))


mixing0, sigma0, means0 = mdn_params(raw_mdn)
cases.append(
    {
        "fn": "mdnParamsFromOutput",
        "raw": raw_mdn.tolist(),
        "numComponents": K,
        "targetDim": L,
        "expected": {
            "mixing": mixing0.tolist(),
            "sigma": sigma0.tolist(),
            "means": means0.tolist(),
        },
    }
)
cases.append(
    {
        "fn": "mdnErrorSingle",
        "raw": raw_mdn.tolist(),
        "numComponents": K,
        "targetDim": L,
        "target": t_mdn.tolist(),
        "expected": mdn_error(raw_mdn, t_mdn),
    }
)
gamma0 = mdn_responsibilities(raw_mdn, t_mdn)
cases.append(
    {
        "fn": "mdnResponsibilities",
        "raw": raw_mdn.tolist(),
        "numComponents": K,
        "targetDim": L,
        "target": t_mdn.tolist(),
        "expected": gamma0.tolist(),
    }
)

expected_output_grad = central_diff_grad(lambda r: mdn_error(r, t_mdn), raw_mdn)
cases.append(
    {
        "fn": "mdnOutputGradients",
        "raw": raw_mdn.tolist(),
        "numComponents": K,
        "targetDim": L,
        "target": t_mdn.tolist(),
        "expected": {
            "dMixing": expected_output_grad[:K].tolist(),
            "dSigma": expected_output_grad[K : 2 * K].tolist(),
            "dMeans": expected_output_grad[2 * K :].reshape(K, L).tolist(),
        },
    }
)

pred_mean = np.sum(mixing0[:, None] * means0, axis=0)
pred_var = float(
    np.sum(
        mixing0 * (L * sigma0**2 + np.sum((means0 - pred_mean[None, :]) ** 2, axis=1))
    )
)
cases.append(
    {
        "fn": "mdnPredictiveMean",
        "raw": raw_mdn.tolist(),
        "numComponents": K,
        "targetDim": L,
        "expected": pred_mean.tolist(),
    }
)
cases.append(
    {
        "fn": "mdnPredictiveVariance",
        "raw": raw_mdn.tolist(),
        "numComponents": K,
        "targetDim": L,
        "expected": pred_var,
    }
)

K2, L2 = 2, 2
raw_mdn2 = rng.normal(scale=0.7, size=K2 * (2 + L2))
t_mdn2 = np.array([0.3, -0.2])
mixing2, sigma2, means2 = mdn_params(raw_mdn2, K2, L2)
cases.append(
    {
        "fn": "mdnParamsFromOutput",
        "raw": raw_mdn2.tolist(),
        "numComponents": K2,
        "targetDim": L2,
        "expected": {
            "mixing": mixing2.tolist(),
            "sigma": sigma2.tolist(),
            "means": means2.tolist(),
        },
    }
)
cases.append(
    {
        "fn": "mdnErrorSingle",
        "raw": raw_mdn2.tolist(),
        "numComponents": K2,
        "targetDim": L2,
        "target": t_mdn2.tolist(),
        "expected": mdn_error(raw_mdn2, t_mdn2, K2, L2),
    }
)
pred_mean2 = np.sum(mixing2[:, None] * means2, axis=0)
pred_var2 = float(
    np.sum(
        mixing2 * (L2 * sigma2**2 + np.sum((means2 - pred_mean2[None, :]) ** 2, axis=1))
    )
)
cases.append(
    {
        "fn": "mdnPredictiveMean",
        "raw": raw_mdn2.tolist(),
        "numComponents": K2,
        "targetDim": L2,
        "expected": pred_mean2.tolist(),
    }
)
cases.append(
    {
        "fn": "mdnPredictiveVariance",
        "raw": raw_mdn2.tolist(),
        "numComponents": K2,
        "targetDim": L2,
        "expected": pred_var2,
    }
)

payload = {"cases": cases}
out = FIXTURES / "neural.json"
out.write_text(json.dumps(payload))
print(f"wrote {out} ({len(cases)} cases)")

diag_gaps = [
    c["_diagnostic_central_diff_vs_analytic_max_relative_gap"]
    for c in cases
    if "_diagnostic_central_diff_vs_analytic_max_relative_gap" in c
]
print(
    f"[diagnostic] backprop central-diff vs analytic-gradient max relative gap: {max(diag_gaps):.3e}"
)
numgrad_gaps = [
    c["_diagnostic_vs_true_analytic_max_abs_gap"]
    for c in cases
    if "_diagnostic_vs_true_analytic_max_abs_gap" in c
]
print(
    f"[diagnostic] numericalGradient central-diff vs true analytic max abs gap: {max(numgrad_gaps):.3e}"
)
