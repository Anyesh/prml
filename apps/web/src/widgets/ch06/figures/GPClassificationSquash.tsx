import { compositeKernel, gpPriorSample, linspace, pcg32, sigmoid } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const GRID = linspace(-1, 1, 200);
const GRID_VECS = GRID.map((x) => [x]);
const KERNEL = compositeKernel({ theta0: 9, theta1: 15, theta2: 0, theta3: 0 });
const LATENT = gpPriorSample(pcg32(6), KERNEL, GRID_VECS);
const SQUASHED = LATENT.map(sigmoid);

export default function GPClassificationSquash() {
  const tokens = useResolvedTokens();
  const maxAbs = Math.max(...LATENT.map(Math.abs));

  return (
    <div className="widget-grid">
      <Plot height={180} xDomain={[-1, 1]} yDomain={[-maxAbs * 1.1, maxAbs * 1.1]} label="One sample from a Gaussian process prior over the latent function a(x)">
        <Axes x={{ label: 'x' }} y={{ label: 'a(x)' }} grid zeroLine />
        <Curve points={GRID.map((x, i) => [x, LATENT[i]!] as const)} color={tokens.series[0]!} width={2} />
      </Plot>
      <Plot height={180} xDomain={[-1, 1]} yDomain={[0, 1]} label="The same sample after passing through the logistic sigmoid">
        <Axes x={{ label: 'x' }} y={{ label: 'sigma(a(x))' }} grid />
        <Curve points={GRID.map((x, i) => [x, SQUASHED[i]!] as const)} color={tokens.color.accent} width={2} />
      </Plot>
      <p className="widget-readout">
        Wherever a(x) crosses zero, sigma(a(x)) crosses 0.5: the sigmoid turns a Gaussian
        process, which ranges over the whole real line, into a process over class
        probabilities that never leaves (0, 1), exactly (6.73)'s Bernoulli parameter.
      </p>
    </div>
  );
}
