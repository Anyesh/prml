import json
import os

import scipy.special as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "special.json")

cases = []


def add(fn, args, expected):
    cases.append({"fn": fn, "args": list(args), "expected": expected})


for x in [0.001, 1e-10, 0.5, 1.0, 2.0, 5.0, 10.0, 170.0, 1e6]:
    add("logGamma", [x], float(sp.gammaln(x)))

for x in [0.001, 0.01, 0.5, 1.0, 2.0, 10.0, 100.0, 1e6]:
    add("digamma", [x], float(sp.psi(x)))

for a, b in [
    (0.5, 0.5),
    (1.0, 1.0),
    (2.0, 3.0),
    (0.001, 0.001),
    (1e6, 1e6),
    (1e-3, 1e6),
    (170.0, 170.0),
]:
    add("logBeta", [a, b], float(sp.betaln(a, b)))

for x, d in [(5.0, 1), (3.0, 2), (10.0, 5), (0.6, 1), (100.0, 10), (50.5, 3)]:
    add("logMultivariateGamma", [x, d], float(sp.multigammaln(x, d)))

for x in [0.0, 0.001, -0.0001, 0.5, 1.0, 2.0, -3.0, 5.0, 10.0]:
    add("erf", [x], float(sp.erf(x)))

for x in [0.0, 0.5, 1.0, -0.0001, -3.0, -5.0, -8.0, 3.0, 5.0, 8.0, 15.0]:
    add("erfc", [x], float(sp.erfc(x)))

for a, x in [
    (1.0, 1.0),
    (0.5, 0.5),
    (5.0, 0.001),
    (5.0, 100.0),
    (0.001, 0.001),
    (100.0, 50.0),
    (100.0, 200.0),
    (1e-3, 1e3),
    (1000.0, 1000.0),
]:
    add("gammaincLower", [a, x], float(sp.gammainc(a, x)))

for x, a, b in [
    (0.5, 0.5, 0.5),
    (0.5, 1.0, 1.0),
    (0.1, 2.0, 5.0),
    (0.999, 5.0, 5.0),
    (0.001, 0.001, 0.001),
    (0.5, 1000.0, 1000.0),
    (1e-6, 2.0, 3.0),
    (0.999999, 2.0, 3.0),
]:
    add("betainc", [x, a, b], float(sp.betainc(a, b, x)))

for nu, x in [
    (0.0, 0.0),
    (0.0, 0.001),
    (0.0, 1.0),
    (0.0, 10.0),
    (0.0, 50.0),
    (0.0, 700.0),
    (1.0, 0.0),
    (1.0, 0.001),
    (1.0, 1.0),
    (1.0, 50.0),
    (1.0, 700.0),
]:
    add("besselI", [nu, x], float(sp.iv(nu, x)))

with open(OUT, "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"wrote {len(cases)} cases to {OUT}")
