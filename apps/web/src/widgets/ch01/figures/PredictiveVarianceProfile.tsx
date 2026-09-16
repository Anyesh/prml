import {
  designMatrix,
  linspace,
  pcg32,
  polynomialBasis,
  standardNormal,
  weightPosterior,
  predictive,
} from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260101;
const NOISE_STD = 0.2;
const N = 10;
const ALPHA = 2e-3;
const BETA = 1 / (NOISE_STD * NOISE_STD);
const BASIS = polynomialBasis(9);

const rng = pcg32(SEED, 1);
const XS = Array.from({ length: N }, (_, i) => i / (N - 1));
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));
const POSTERIOR = weightPosterior(designMatrix(XS, BASIS), TS, { alpha: ALPHA, beta: BETA });

const GRID = linspace(-0.5, 1.5, 161);
const STD_CURVE = GRID.map((x) => [x, Math.sqrt(predictive(BASIS(x), POSTERIOR, BETA).variance)] as const);

export default function PredictiveVarianceProfile() {
  const tokens = useResolvedTokens();
  const insideMax = Math.max(...STD_CURVE.filter(([x]) => x >= 0 && x <= 1).map(([, s]) => s));

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[-0.5, 1.5]} yDomain={[0, 3]} label="Predictive standard deviation against x, training inputs confined to [0,1]">
        <Axes x={{ label: 'x' }} y={{ label: 'predictive std' }} grid />
        <Rule x={0} color={tokens.color.inkFaint} />
        <Rule x={1} color={tokens.color.inkFaint} label="training range" />
        <Rule y={NOISE_STD} color={tokens.color.danger} label="noise floor" />
        <Curve points={STD_CURVE} color={tokens.color.accent} width={2} />
      </Plot>
      <p className="widget-readout">
        {`Inside [0,1] the predictive std stays near ${insideMax.toFixed(2)}, just above the noise floor of ${NOISE_STD}. At x=1.3, past the last training point, it has grown to ${Math.sqrt(predictive(BASIS(1.3), POSTERIOR, BETA).variance).toFixed(1)}: the model knows it is guessing.`}
      </p>
    </div>
  );
}
