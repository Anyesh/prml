import { useMemo, useState } from 'react';
import { compositeKernel, fitGPRegression, gpPredict, gpPriorSample, linspace, pcg32 } from '@prml/math';
import { Axes, ClickSurface, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

const GRID = linspace(0, 1, 90);
const GRID_VECS = GRID.map((x) => [x]);
const NOISE_VARIANCE = 1 / 25;
const PRIOR_SEED = 20260916;
const SAMPLE_COUNT = 4;

interface Observation {
  readonly x: number;
  readonly t: number;
}

export default function GaussianProcessExplorer() {
  const [theta0, setTheta0] = useState(1.0);
  const [theta1, setTheta1] = useState(20);
  const [theta2, setTheta2] = useState(0);
  const [theta3, setTheta3] = useState(0);
  const [points, setPoints] = useState<Observation[]>([]);
  const tokens = useResolvedTokens();

  const kernel = useMemo(() => compositeKernel({ theta0, theta1, theta2, theta3 }), [theta0, theta1, theta2, theta3]);

  const priorSamples = useMemo(() => {
    if (points.length > 0) return [];
    const rng = pcg32(PRIOR_SEED);
    return Array.from({ length: SAMPLE_COUNT }, (_, i) => gpPriorSample(rng.fork(i + 1), kernel, GRID_VECS));
  }, [kernel, points.length]);

  const posterior = useMemo(() => {
    if (points.length === 0) return null;
    const model = fitGPRegression(kernel, points.map((p) => [p.x]), points.map((p) => p.t), NOISE_VARIANCE);
    return GRID.map((x) => gpPredict(model, [x]));
  }, [kernel, points]);

  const priorBand = GRID.map((x) => 2 * Math.sqrt(kernel([x], [x])));

  return (
    <div className="widget-grid">
      <Plot height={320} xDomain={[0, 1]} yDomain={[-4, 4]} label="Gaussian process: prior samples before any data, posterior mean and band after">
        <Axes x={{ label: 'x' }} y={{ label: 'y' }} grid zeroLine />
        <ClickSurface onClick={(x, t) => setPoints((current) => [...current, { x, t }])} cursor="copy" />
        {points.length === 0 && (
          <Curve points={GRID.map((x, i) => [x, priorBand[i]!] as const)} color={tokens.color.inkFaint} dash="dashed" width={1} />
        )}
        {points.length === 0 && (
          <Curve points={GRID.map((x, i) => [x, -priorBand[i]!] as const)} color={tokens.color.inkFaint} dash="dashed" width={1} />
        )}
        {priorSamples.map((sample, i) => (
          <Curve key={i} points={GRID.map((x, j) => [x, sample[j]!] as const)} color={tokens.series[i % tokens.series.length]!} width={1.25} />
        ))}
        {posterior && (
          <Curve points={GRID.map((x, i) => [x, posterior[i]!.mean + 2 * Math.sqrt(posterior[i]!.variance)] as const)} color={tokens.color.accent} dash="dashed" width={1} />
        )}
        {posterior && (
          <Curve points={GRID.map((x, i) => [x, posterior[i]!.mean - 2 * Math.sqrt(posterior[i]!.variance)] as const)} color={tokens.color.accent} dash="dashed" width={1} />
        )}
        {posterior && <Curve points={GRID.map((x, i) => [x, posterior[i]!.mean] as const)} color={tokens.color.accent} width={2.5} />}
        <ScatterField points={points.map((p, i) => ({ x: p.x, y: p.t, id: i }))} color={tokens.color.ink} size={5} />
      </Plot>

      <Panel columns={2} dense>
        <Slider label="theta0 (amplitude)" value={theta0} onChange={setTheta0} min={0.1} max={5} step={0.05} />
        <Slider label="theta1 (inverse length scale squared)" value={theta1} onChange={setTheta1} min={1} max={100} step={1} scale="log" />
        <Slider label="theta2 (constant)" value={theta2} onChange={setTheta2} min={0} max={2} step={0.05} />
        <Slider label="theta3 (linear term)" value={theta3} onChange={setTheta3} min={0} max={3} step={0.05} />
        <p className="widget-readout">
          {points.length === 0
            ? 'No points yet: click inside the plot to place one. The four sliders are the (6.63) hyperparameters, and the dashed lines are the prior standard deviation band.'
            : `${points.length} point${points.length === 1 ? '' : 's'} clicked. The band is the posterior mean plus and minus two standard deviations (6.66-6.67).`}
        </p>
        <button type="button" className="widget-reset" onClick={() => setPoints([])}>
          Clear points
        </button>
      </Panel>
    </div>
  );
}
