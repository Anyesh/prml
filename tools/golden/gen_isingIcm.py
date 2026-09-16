"""Golden fixtures for packages/math/src/graphical/ising.ts.

Independent numpy re-implementation of the Ising de-noising model from PRML 8.3.3
(energy 8.42) and iterated conditional modes: a fixed 4-neighbourhood energy

    E(x, y) = h * sum_i x_i - beta * sum_{(i,j) neighbours} x_i * x_j - eta * sum_i x_i * y_i

with x, y in {-1, +1}, and ICM sweeping pixels in raster order, each pixel set to
whichever of {-1, +1} minimises its local energy given every other pixel's *current*
value (including neighbours already re-visited earlier in the same sweep).
"""

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def energy(x, y, h, beta, eta):
    rows, cols = x.shape
    total = h * x.sum() - eta * (x * y).sum()
    coupling = 0.0
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                coupling += x[i, j] * x[i, j + 1]
            if i + 1 < rows:
                coupling += x[i, j] * x[i + 1, j]
    total -= beta * coupling
    return float(total)


def icm_sweep(x, y, h, beta, eta):
    rows, cols = x.shape
    x = x.copy()
    for i in range(rows):
        for j in range(cols):
            s = 0.0
            if i > 0:
                s += x[i - 1, j]
            if i + 1 < rows:
                s += x[i + 1, j]
            if j > 0:
                s += x[i, j - 1]
            if j + 1 < cols:
                s += x[i, j + 1]
            field = beta * s + eta * y[i, j] - h
            x[i, j] = 1.0 if field >= 0 else -1.0
    return x


# A 5x5 noisy image: a clean 3x3 block of +1 in a field of -1, with three pixels
# flipped by noise (one inside the block, two outside), so both interior and
# boundary corrections are exercised.
clean = -np.ones((5, 5))
clean[1:4, 1:4] = 1.0
y = clean.copy()
y[2, 2] = -1.0  # flipped inside the block
y[0, 0] = 1.0  # flipped in the background
y[4, 4] = 1.0  # flipped in the background, far corner

cases = []

for h, beta, eta in [(0.0, 1.5, 2.1), (0.3, 1.5, 2.1)]:
    cases.append(
        {
            "fn": "isingEnergy",
            "x": clean.tolist(),
            "y": y.tolist(),
            "params": {"h": h, "beta": beta, "eta": eta},
            "expected": energy(clean, y, h, beta, eta),
        }
    )
    cases.append(
        {
            "fn": "isingEnergy_noisy",
            "x": y.tolist(),
            "y": y.tolist(),
            "params": {"h": h, "beta": beta, "eta": eta},
            "expected": energy(y, y, h, beta, eta),
        }
    )

params = {"h": 0.0, "beta": 1.5, "eta": 2.1}
cases.append(
    {
        "fn": "icmSweep",
        "x": y.tolist(),
        "y": y.tolist(),
        "params": params,
        "expected": icm_sweep(y, y, **params).tolist(),
    }
)

x = y.copy()
history = [x.tolist()]
energyHistory = [energy(x, y, **params)]
for _ in range(4):
    x = icm_sweep(x, y, **params)
    history.append(x.tolist())
    energyHistory.append(energy(x, y, **params))

cases.append(
    {
        "fn": "icmDenoise",
        "initialX": y.tolist(),
        "y": y.tolist(),
        "params": params,
        "sweeps": 4,
        "expectedHistory": history,
        "expectedEnergyHistory": energyHistory,
    }
)

(FIXTURES / "isingIcm.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {len(cases)} cases to {FIXTURES / 'isingIcm.json'}")
