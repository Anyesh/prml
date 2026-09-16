import { covarianceMatrix, dataMean, faFitEM, faMarginalCov, mvnCovarianceEllipse, ppcaMLE, ppcaMarginalCov, submatrix, type Ellipse } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { anisotropicNoiseData } from '../data.js';

import '../../widgets.css';

const DATA = anisotropicNoiseData();
const MEAN = dataMean(DATA);
const EMPIRICAL_COV = covarianceMatrix(DATA, MEAN);

const PPCA = ppcaMLE(DATA, 1);
const FA_FIT = faFitEM(DATA, { mean: MEAN, w: [[0.5], [0.5], [0.5]], psi: [1, 1, 1] }, 40);
const FA = FA_FIT.paramsHistory[FA_FIT.paramsHistory.length - 1]!;

const PLANE = [0, 1] as const;
const DOMAIN: readonly [number, number] = [-4, 4];

function ellipsePoints(ellipse: Ellipse, samples = 64): (readonly [number, number])[] {
  const cosA = Math.cos(ellipse.angle);
  const sinA = Math.sin(ellipse.angle);
  return Array.from({ length: samples + 1 }, (_, i) => {
    const t = (i / samples) * 2 * Math.PI;
    const x = ellipse.rx * Math.cos(t);
    const y = ellipse.ry * Math.sin(t);
    return [ellipse.cx + x * cosA - y * sinA, ellipse.cy + x * sinA + y * cosA] as const;
  });
}

const planeMean = [MEAN[PLANE[0]]!, MEAN[PLANE[1]]!];
const EMPIRICAL_ELLIPSE = mvnCovarianceEllipse({ mean: planeMean, cov: submatrix(EMPIRICAL_COV, PLANE, PLANE) }, 0.95);
const PPCA_ELLIPSE = mvnCovarianceEllipse(
  { mean: planeMean, cov: submatrix(ppcaMarginalCov(PPCA.w, PPCA.sigma2), PLANE, PLANE) },
  0.95,
);
const FA_ELLIPSE = mvnCovarianceEllipse({ mean: planeMean, cov: submatrix(faMarginalCov(FA.w, FA.psi), PLANE, PLANE) }, 0.95);

export default function PpcaVsFactorAnalysis() {
  const tokens = useResolvedTokens();

  return (
    <Plot
      height={280}
      xDomain={DOMAIN}
      yDomain={DOMAIN}
      equalAspect
      label="The quiet axis against a noisy one: empirical spread against what PPCA and factor analysis each fit to it"
    >
      <Axes x={{ label: 'x1 (low noise)' }} y={{ label: 'x2 (high noise)' }} grid zeroLine />
      <ScatterField points={DATA.map((p) => ({ x: p[PLANE[0]]!, y: p[PLANE[1]]! }))} color={tokens.color.inkMuted} size={2} opacity={0.5} />
      <Curve points={ellipsePoints(EMPIRICAL_ELLIPSE)} color={tokens.color.ink} width={1.5} />
      <Curve points={ellipsePoints(PPCA_ELLIPSE)} color={tokens.series[0]!} width={2} dash="dashed" />
      <Curve points={ellipsePoints(FA_ELLIPSE)} color={tokens.series[1]!} width={2} />
      <Legend
        entries={[
          { label: 'empirical 95%', color: tokens.color.ink, mark: 'line' },
          { label: 'PPCA fit', color: tokens.series[0]!, mark: 'dashed-line' },
          { label: 'factor analysis fit', color: tokens.series[1]!, mark: 'line' },
        ]}
      />
    </Plot>
  );
}
