import { compositeKernel, fitGPRegression, gpPredict, linspace, pcg32, standardNormal } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const rng = pcg32(555);
const XS = [0.1, 0.3, 0.5, 0.7, 0.9];
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + 0.1 * standardNormal(rng));
const KERNEL = compositeKernel({ theta0: 1, theta1: 4, theta2: 0, theta3: 0 });
const NOISE_VARIANCE = 1 / 25;
const MODEL = fitGPRegression(KERNEL, XS.map((x) => [x]), TS, NOISE_VARIANCE);
const QUERY_POINTS = [0.5, 1.2];
const GRID = linspace(0, 1.4, 150);
const PREDICTIONS = GRID.map((x) => gpPredict(MODEL, [x]));

export default function GPRegressionWorkedNumbers() {
  const tokens = useResolvedTokens();
  const upper = GRID.map((x, i) => [x, PREDICTIONS[i]!.mean + 2 * Math.sqrt(PREDICTIONS[i]!.variance)] as const);
  const lower = GRID.map((x, i) => [x, PREDICTIONS[i]!.mean - 2 * Math.sqrt(PREDICTIONS[i]!.variance)] as const);

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, 1.4]} yDomain={[-2, 2]} label="GP posterior mean and band, with the two queried points marked">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={upper} color={tokens.color.inkFaint} dash="dashed" width={1} />
        <Curve points={lower} color={tokens.color.inkFaint} dash="dashed" width={1} />
        <Curve points={GRID.map((x, i) => [x, PREDICTIONS[i]!.mean] as const)} color={tokens.color.accent} width={2.5} />
        <ScatterField points={XS.map((x, i) => ({ x, y: TS[i]!, id: i }))} color={tokens.color.ink} size={4} />
        <ScatterField points={QUERY_POINTS.map((x, i) => ({ x, y: gpPredict(MODEL, [x]).mean, id: `q${i}` }))} color={tokens.color.danger} size={6} />
      </Plot>
      <p className="widget-readout">
        {`theta0 = 1, theta1 = 4 (PRML's own Figure 6.5 setting), five points. At x = 0.5, inside the data: mean ${gpPredict(MODEL, [0.5]).mean.toFixed(4)}, variance ${gpPredict(MODEL, [0.5]).variance.toFixed(4)}. At x = 1.2, past every training point: mean ${gpPredict(MODEL, [1.2]).mean.toFixed(4)}, variance ${gpPredict(MODEL, [1.2]).variance.toFixed(4)}, nearly five times as uncertain.`}
      </p>
    </div>
  );
}
