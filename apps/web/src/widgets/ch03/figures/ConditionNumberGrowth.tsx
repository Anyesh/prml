import { designMatrix, linspace, matmul, polynomialBasis, svd, transpose } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DEGREES = Array.from({ length: 12 }, (_, i) => i + 1);
const DOUBLE_PRECISION_LOG10 = Math.log10(1e16);

/**
 * The same 25 seeded-uniform x-values as tools/golden/fixtures/regression.json (numpy
 * `default_rng(20260916)`, sorted), so this figure's degree-9 numbers land on the 7.3e7 /
 * 5.3e15 the prose and the golden tests both cite. Copied as a literal rather than redrawn
 * with `pcg32` under the same seed number, because PCG32 and numpy's PCG64 are different
 * generators: a `pcg32(20260916)` draw sorts to a visibly different clustering and lands
 * cond(Φ) at degree 9 around 1.0e7, not 7.3e7.
 */
const CLUSTERED_X = [
  0.05400707754585976, 0.14820390210047552, 0.1687239241716425, 0.16956034539506615, 0.17009231230220867,
  0.18265825972462246, 0.19749246151256095, 0.22301378830236118, 0.22452139524305803, 0.3274697208809966,
  0.3319895697577765, 0.46108580828055623, 0.48567882802442386, 0.5026537839623075, 0.5186622377431134,
  0.5419055968264217, 0.5899605320357354, 0.654191207189241, 0.6576002726905469, 0.6662230853482721,
  0.683782222456604, 0.68698037267959, 0.7029355158101181, 0.7275866225405225, 0.9227716068373659,
];

const EVEN_X = linspace(0, 1, 25);

interface CondPoint {
  readonly degree: number;
  readonly logCondPhi: number;
  readonly logCondGram: number;
}

function conditionAt(xs: readonly number[], degree: number): { condPhi: number; condGram: number } {
  const design = designMatrix(xs, polynomialBasis(degree, { bias: true }));
  const sPhi = svd(design).s;
  const condPhi = sPhi[0]! / sPhi[sPhi.length - 1]!;
  const gram = matmul(transpose(design), design);
  const sGram = svd(gram).s;
  const condGram = sGram[0]! / sGram[sGram.length - 1]!;
  return { condPhi, condGram };
}

/** Both series are read off an actual SVD at each degree, never restated from the prose. */
function computeCurve(xs: readonly number[]): CondPoint[] {
  return DEGREES.map((degree) => {
    const { condPhi, condGram } = conditionAt(xs, degree);
    return { degree, logCondPhi: Math.log10(condPhi), logCondGram: Math.log10(condGram) };
  });
}

const CLUSTERED_CURVE = computeCurve(CLUSTERED_X);
const EVEN_CURVE = computeCurve(EVEN_X);
const AT_DEGREE_9_CLUSTERED = CLUSTERED_CURVE.find((p) => p.degree === 9)!;
const AT_DEGREE_9_EVEN = EVEN_CURVE.find((p) => p.degree === 9)!;
const Y_MAX =
  Math.max(DOUBLE_PRECISION_LOG10, ...CLUSTERED_CURVE.map((p) => p.logCondGram), ...EVEN_CURVE.map((p) => p.logCondGram)) + 1;

export default function ConditionNumberGrowth() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot
        height={220}
        xDomain={[1, 12]}
        yDomain={[0, Y_MAX]}
        label="Condition number of Phi and of Phi transpose Phi against polynomial degree, for a clustered dataset and an evenly-spaced one"
      >
        <Axes
          x={{ label: 'degree', tickCount: 12 }}
          y={{ label: 'condition number', ticks: [0, 4, 8, 12, 16], format: (v) => `1e${Math.round(v)}` }}
          grid
        />
        <Rule y={DOUBLE_PRECISION_LOG10} color={tokens.color.danger} label="double-precision limit" />
        <Curve points={CLUSTERED_CURVE.map((p) => [p.degree, p.logCondPhi] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={CLUSTERED_CURVE.map((p) => [p.degree, p.logCondGram] as const)} color={tokens.series[1]!} width={2} />
        <Curve points={EVEN_CURVE.map((p) => [p.degree, p.logCondPhi] as const)} color={tokens.series[0]!} dash="dashed" width={1.5} />
        <Curve points={EVEN_CURVE.map((p) => [p.degree, p.logCondGram] as const)} color={tokens.series[1]!} dash="dashed" width={1.5} />
        <Legend
          entries={[
            { label: 'cond(Φ), clustered x (book dataset)', color: tokens.series[0]!, mark: 'line' },
            { label: 'cond(ΦᵀΦ), clustered x', color: tokens.series[1]!, mark: 'line' },
            { label: 'cond(Φ), evenly spaced x', color: tokens.series[0]!, mark: 'dashed-line' },
            { label: 'cond(ΦᵀΦ), evenly spaced x', color: tokens.series[1]!, mark: 'dashed-line' },
          ]}
          placement="top-left"
        />
      </Plot>
      <p className="widget-readout">
        {`At degree 9: the book's clustered 25 points give cond(Φ) = ${(10 ** AT_DEGREE_9_CLUSTERED.logCondPhi).toExponential(2)}, cond(ΦᵀΦ) = ${(10 ** AT_DEGREE_9_CLUSTERED.logCondGram).toExponential(2)}. The same degree on 25 evenly spaced points gives cond(Φ) = ${(10 ** AT_DEGREE_9_EVEN.logCondPhi).toExponential(2)}, cond(ΦᵀΦ) = ${(10 ** AT_DEGREE_9_EVEN.logCondGram).toExponential(2)}, roughly twenty times better. Where the inputs sit matters as much as the model order: a few points crowded together near x = 0.17 is most of what drives the book's number past the evenly spaced one.`}
      </p>
    </div>
  );
}
