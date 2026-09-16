import { gaussianBasis, linspace, polynomialBasis, sigmoidalBasis } from '@prml/math';
import { Axes, FunctionCurve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DOMAIN: readonly [number, number] = [-1.3, 1.3];
const FIT_RANGE: readonly [number, number] = [-1, 1];
const CENTRES = linspace(FIT_RANGE[0], FIT_RANGE[1], 5);
const GAUSSIAN_SCALE = 0.25;
const SIGMOID_SCALE = 0.2;
const PANEL_HEIGHT = 130;

interface Family {
  readonly title: string;
  readonly functions: readonly ((x: number) => number)[];
  readonly yDomain: readonly [number, number];
}

const POLY = polynomialBasis(5, { bias: false });
const GAUSS = gaussianBasis(CENTRES, GAUSSIAN_SCALE, { bias: false });
const SIGMOID = sigmoidalBasis(CENTRES, SIGMOID_SCALE, { bias: false });

const FAMILIES: readonly Family[] = [
  {
    title: 'Polynomial: x, x squared, up to x to the fifth, unbounded once x leaves the fitted range',
    functions: [0, 1, 2, 3, 4].map((i) => (x: number) => POLY(x)[i]!),
    yDomain: [-4, 4],
  },
  {
    title: 'Gaussian: a compact bump around each centre',
    functions: CENTRES.map((_, i) => (x: number) => GAUSS(x)[i]!),
    yDomain: [0, 1.05],
  },
  {
    title: 'Sigmoidal: saturating to 0 or 1 away from its centre',
    functions: CENTRES.map((_, i) => (x: number) => SIGMOID(x)[i]!),
    yDomain: [0, 1.05],
  },
];

export default function BasisFamilies() {
  const tokens = useResolvedTokens();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--prml-space-3)' }}>
      {FAMILIES.map((family) => (
        <div key={family.title}>
          <p className="widget-readout">{family.title}</p>
          <Plot height={PANEL_HEIGHT} xDomain={DOMAIN} yDomain={family.yDomain} label={family.title}>
            <Axes x={{ label: 'x', tickCount: 4 }} y={{ tickCount: 3 }} grid zeroLine />
            {family.functions.map((f, i) => (
              <FunctionCurve key={i} f={f} domain={DOMAIN} color={tokens.series[i % tokens.series.length]!} width={1.5} />
            ))}
          </Plot>
        </div>
      ))}
    </div>
  );
}
