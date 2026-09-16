import { linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { DATA, DOMAIN_HI, DOMAIN_LO, truePdf } from '../nonparametricData';
import '../../widgets.css';

const N = DATA.length;
const SAMPLE_MEAN = DATA.reduce((a, b) => a + b, 0) / N;
const SAMPLE_VAR = DATA.reduce((a, x) => a + (x - SAMPLE_MEAN) ** 2, 0) / N;
const GRID = linspace(DOMAIN_LO, DOMAIN_HI, 200);

export default function GaussianMixtureFit() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[DOMAIN_LO, DOMAIN_HI]} yDomain={[0, 0.5]} label="Best single Gaussian against the true two-component mixture, on bimodal data">
        <Axes x={{ label: 'x' }} y={{ label: 'density' }} grid />
        <Curve points={GRID.map((x) => [x, normalPdf(x, { mu: SAMPLE_MEAN, sigma2: SAMPLE_VAR })] as const)} color={tokens.color.danger} width={2} />
        <Curve points={GRID.map((x) => [x, truePdf(x)] as const)} color={tokens.color.accent} width={2} />
        <ScatterField points={DATA.map((x) => ({ x, y: 0.01, size: 2, color: tokens.color.ink }))} />
        <Legend
          entries={[
            { label: 'single Gaussian (ML)', color: tokens.color.danger, mark: 'line' },
            { label: 'true 2-component mixture', color: tokens.color.accent, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <p className="widget-readout">
        {`One Gaussian fit by ML, mean ${SAMPLE_MEAN.toFixed(2)} variance ${SAMPLE_VAR.toFixed(2)} (eq. 2.121-2.122), sits in the `}
        {'valley between the two real modes, matching neither. Equation 2.188 with two components has the right shape by construction; fitting it is chapter 9.'}
      </p>
    </div>
  );
}
