# Golden fixtures

Every expected numeric value used by `@prml/math`'s tests is generated here, by a
Python script that calls numpy/scipy. No test file hand-writes an expected number:
that would only prove the implementation agrees with whoever typed the literal, which
for a Bessel ratio or an incomplete beta is worth nothing.

## Regenerating

```sh
node tools/golden/generate.mjs            # regenerate every fixture
node tools/golden/generate.mjs normal     # regenerate one (matches gen_normal.py)
```

Each invocation shells out to `uv run --with numpy --with scipy python tools/golden/gen_<name>.py`.
`uv` resolves numpy/scipy on the fly; nothing is installed into the JS toolchain. The
script writes `tools/golden/fixtures/<name>.json` itself; `generate.mjs` only invokes it
and reports failure.

## Fixture shape

Every fixture is `{ "cases": [...] }`. Each case carries a `fn` discriminant naming the
function under test, whatever inputs that function needs (`x`, `params`, `theta`, `args`,
...), and an `expected` value computed by scipy at full float64 precision (Python's
`json.dump` uses `repr`-equivalent formatting, so no precision is lost rounding to
text). Test files filter `cases` by `fn` and feed the recorded inputs straight to the
TypeScript implementation.

Values that are not finite (`inf`, `nan`) are never emitted, because standard JSON has
no token for them and `JSON.parse` would reject the file; hard cases are chosen to sit
just short of that boundary (e.g. `besselI` at `kappa = 700`, not `kappa = 2000`).

## What each fixture covers

- **special.json**: `logGamma`, `digamma`, `logBeta`, `logMultivariateGamma`, `erf`,
  `erfc`, `gammaincLower`, `betainc`, `besselI`. Covers tiny arguments (`1e-10`, `0.001`),
  huge arguments (`1e6`), values near the support boundary, `a = b = 0.5` for `betainc`
  (the arcsine case), `erfc` at `x = 5` and `x = 8` (where `1 - erf(x)` loses all
  significance), and `besselI` at `kappa = 700` (where the direct power series overflows
  before the factorial denominator catches up).
- **normal.json**: `logPdf`/`pdf` in the far tail and at extreme variance, `cdf` out to
  `|z| = 8`, `quantile` at `q` within `1e-6` of 0 and 1 (tail inversion accuracy).
- **beta.json**: `logPdf`/`pdf` near both simplex boundaries with `a` or `b` below 1
  (density diverges) and exactly 1 (the `(a - 1) * log(x)` guard at `x = 0`), `a = b =
  0.5` (arcsine), `cdf`, `mean`/`variance` at extreme shape, and `posterior` (pure
  addition, PRML 2.18 - no scipy call needed, just the same arithmetic the fixture
  asserts against).
- **gamma.json**: shape below 1 (density diverges at 0), shape exactly 1 (the guard at
  `x = 0`), extreme rate/shape combinations, `cdf` via the regularised incomplete gamma.
- **studentT.json**: `nu = 1` (Cauchy), `nu = 1e6` (approaches Gaussian), tail `cdf`
  values, and `multivariateLogPdf` against an independent numpy/scipy reference
  (`gammaln` plus `np.linalg.inv`/`slogdet`, not a call into the TypeScript algorithm's
  own dependencies).
- **dirichlet.json**: concentrated and near-uniform `alpha`, a near-vertex `x`, extreme
  component ratios (`[1000, 1, 1]`, `[0.001, 1, 1000]`) for `mean` and `expectedLog`.
- **vonMises.json**: `kappa = 0` (uniform) and `kappa = 700` (the same hard Bessel
  region as special.json), plus `fit` cases where the ground truth is an independent
  `scipy.optimize.brentq` solve of `I1(kappa)/I0(kappa) = R` (not a re-derivation of the
  bisection the TypeScript implementation performs), including one exactly-symmetric
  angle set where the true resultant length is 0.
- **discrete.json**: `binomialPmf`/`logBinomialCoefficient` at `n = 1000` and `n = 170`
  (where `n!` overflows a naive factorial), `multinomialLogPmf` with a zero-count
  category, `bernoulliLogPmf` near both boundaries of `mu`.

Sampler functions (`normalSample`, `betaSample`, `gammaSample`, `dirichletSample`,
`studentTSample`, `vonMisesSample`, `bernoulliSample`, `multinomialSample`) have no
fixture: there is no scipy call whose *exact* output a JS PRNG should reproduce. Their
tests instead assert domain constraints (support bounds, counts summing to `n`) and
that a large sample's mean tracks the distribution's own `mean`/`variance` functions
within a statistical tolerance - never a hand-typed numeric target.

## The rule

If a test needs a new number, write (or extend) the `gen_<name>.py` that produces it.
Never type a `toBeCloseTo(<literal>)` or edit a fixture's `expected` field by hand.
