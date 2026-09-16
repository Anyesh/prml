import { compositeKernel, fitGPRegression, gpPredict, linspace, pcg32, standardNormal } from '@prml/math';
import { Axes, Curve, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const rng = pcg32(20260916);
const XS = Array.from({ length: 10 }, () => 0.3 + 0.4 * rng.next()).sort((a, b) => a - b);
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + 0.15 * standardNormal(rng));
const KERNEL = compositeKernel({ theta0: 1, theta1: 20, theta2: 0, theta3: 0 });
const NOISE_VARIANCE = 1 / 25;
const MODEL = fitGPRegression(KERNEL, XS.map((x) => [x]), TS, NOISE_VARIANCE);
const GRID = linspace(-0.5, 1.5, 200);
const PREDICTIONS = GRID.map((x) => gpPredict(MODEL, [x]));

export default function PredictiveVarianceGrowsAway() {
  const tokens = useResolvedTokens();
  const maxVar = Math.max(...PREDICTIONS.map((p) => p.variance));

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[-0.5, 1.5]} yDomain={[0, maxVar * 1.05]} label="Predictive variance across and beyond the training range">
        <Axes x={{ label: 'x' }} y={{ label: 'predictive variance' }} grid />
        <Rule x={XS[0]!} color={tokens.color.inkFaint} />
        <Rule x={XS[XS.length - 1]!} color={tokens.color.inkFaint} />
        <Curve points={GRID.map((x, i) => [x, PREDICTIONS[i]!.variance] as const)} color={tokens.color.accent} width={2} />
        <ScatterField points={XS.map((x) => ({ x, y: NOISE_VARIANCE, id: x }))} color={tokens.color.ink} size={3} />
      </Plot>
      <p className="widget-readout">
        {`Training points sit between the two grey lines. Inside that range the variance stays near the noise floor of ${NOISE_VARIANCE.toFixed(3)}; past either edge it climbs toward theta0 = 1, the prior variance, because nothing observed constrains the function out there.`}
      </p>
    </div>
  );
}
