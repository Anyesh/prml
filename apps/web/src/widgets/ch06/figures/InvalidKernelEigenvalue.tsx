import { eigSym, gramMatrix, linspace, sigmoidKernel } from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const POINTS = linspace(-1, 1, 10).map((x) => [x]);
const A_VALUES = linspace(0.1, 4, 60);
const B = 0.5;

const MIN_EIGENVALUES = A_VALUES.map((a) => {
  const gram = gramMatrix(sigmoidKernel(a, B), POINTS);
  return Math.min(...eigSym(gram).values);
});

const FIRST_NEGATIVE = A_VALUES.find((_, i) => MIN_EIGENVALUES[i]! < 0);

export default function InvalidKernelEigenvalue() {
  const tokens = useResolvedTokens();
  const curve = A_VALUES.map((a, i) => [a, MIN_EIGENVALUES[i]!] as const);
  const range = Math.max(...MIN_EIGENVALUES.map(Math.abs));

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[A_VALUES[0]!, A_VALUES[A_VALUES.length - 1]!]} yDomain={[-range * 1.1, range * 1.1]} label="Smallest Gram-matrix eigenvalue for the sigmoid kernel, against its slope parameter a">
        <Axes x={{ label: "a in tanh(a x^Tx' + b)" }} y={{ label: 'smallest eigenvalue of K' }} grid zeroLine />
        <Rule y={0} color={tokens.color.inkFaint} />
        <Curve points={curve} color={tokens.color.accent} width={2} />
      </Plot>
      <p className="widget-readout">
        {`Ten fixed points, b = ${B}. Past a = ${FIRST_NEGATIVE?.toFixed(2) ?? 'n/a'} the smallest eigenvalue of K turns negative, exactly the failure PRML warns about for (6.37): sum, product and scale never do this to a valid kernel, but tanh was never guaranteed to be one.`}
      </p>
    </div>
  );
}
