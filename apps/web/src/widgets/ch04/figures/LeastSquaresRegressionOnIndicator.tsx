import { designMatrix, maximumLikelihoodWeights, pcg32, polynomialBasis, standardNormal } from '@prml/math';
import { Axes, Curve, FunctionCurve, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const rng = pcg32(20260412);
const XS_A = Array.from({ length: 12 }, () => 2 + 1.4 * standardNormal(rng));
const XS_B = Array.from({ length: 12 }, () => 7 + 1.4 * standardNormal(rng));
const XS = [...XS_A, ...XS_B];
const TS = [...XS_A.map(() => 0), ...XS_B.map(() => 1)];

const PHI = polynomialBasis(1);
const WEIGHTS = maximumLikelihoodWeights(designMatrix(XS, PHI), TS);
const DOMAIN: readonly [number, number] = [-2, 12];

export default function LeastSquaresRegressionOnIndicator() {
  const tokens = useResolvedTokens();
  const fit = (x: number) => WEIGHTS[0]! + WEIGHTS[1]! * x;

  return (
    <Plot height={240} xDomain={DOMAIN} yDomain={[-0.6, 1.6]} label="Least squares fit directly onto a 0/1 class indicator">
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      <Rule y={0} color={tokens.color.inkFaint} dash />
      <Rule y={1} color={tokens.color.inkFaint} dash />
      <Curve
        points={[
          [-2, 0],
          [4.5, 0],
          [4.5, 1],
          [12, 1],
        ]}
        color={tokens.color.inkMuted}
        width={1.5}
        dash="dotted"
      />
      <FunctionCurve f={fit} domain={DOMAIN} color={tokens.color.danger} width={2} />
      <ScatterField points={XS_A.map((x, i) => ({ x, y: 0, id: `a${i}` }))} color={tokens.series[0]} size={3.5} />
      <ScatterField points={XS_B.map((x, i) => ({ x, y: 1, id: `b${i}` }))} color={tokens.series[1]} size={3.5} />
    </Plot>
  );
}
