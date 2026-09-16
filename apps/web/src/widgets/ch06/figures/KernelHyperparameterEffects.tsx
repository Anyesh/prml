import { compositeKernel, gpPriorSample, linspace, pcg32 } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const GRID = linspace(0, 1, 80);
const GRID_VECS = GRID.map((x) => [x]);
const SAMPLE_COUNT = 4;
const SEED = 20260916;

function samplesFor(theta0: number, theta1: number) {
  const kernel = compositeKernel({ theta0, theta1, theta2: 0, theta3: 0 });
  const rng = pcg32(SEED);
  return Array.from({ length: SAMPLE_COUNT }, (_, i) => gpPriorSample(rng.fork(i + 1), kernel, GRID_VECS));
}

const THETA1_SWEEP = [4, 20, 100];
const THETA0_SWEEP = [0.3, 1, 3];

export default function KernelHyperparameterEffects() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      {THETA1_SWEEP.map((theta1) => (
        <Plot key={`t1-${theta1}`} height={160} xDomain={[0, 1]} yDomain={[-3, 3]} label={`Prior samples at theta1 = ${theta1}, theta0 = 1`}>
          <Axes x={{ label: 'x' }} y={{ label: 'y' }} grid />
          {samplesFor(1, theta1).map((curve, i) => (
            <Curve key={i} points={GRID.map((x, j) => [x, curve[j]!] as const)} color={tokens.series[0]!} width={1.25} />
          ))}
        </Plot>
      ))}
      {THETA0_SWEEP.map((theta0) => (
        <Plot key={`t0-${theta0}`} height={160} xDomain={[0, 1]} yDomain={[-6, 6]} label={`Prior samples at theta0 = ${theta0}, theta1 = 20`}>
          <Axes x={{ label: 'x' }} y={{ label: 'y' }} grid />
          {samplesFor(theta0, 20).map((curve, i) => (
            <Curve key={i} points={GRID.map((x, j) => [x, curve[j]!] as const)} color={tokens.series[1]!} width={1.25} />
          ))}
        </Plot>
      ))}
      <p className="widget-readout">
        Top row: raising theta1 sharpens the squared-exponential falloff, so functions wiggle
        faster over the same interval. Bottom row: theta0 sets the prior variance directly,
        stretching the same shapes taller without changing how fast they wiggle.
      </p>
    </div>
  );
}
