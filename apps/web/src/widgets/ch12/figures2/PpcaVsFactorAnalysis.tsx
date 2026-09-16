import { covarianceMatrix, dataMean, faFitEM, faMarginalCov, ppcaMLE, ppcaMarginalCov, submatrix } from '@prml/math';
import { Axes, CovarianceEllipse, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
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

const MASS = 0.95;
const planeMean = [MEAN[PLANE[0]]!, MEAN[PLANE[1]]!];
const EMPIRICAL_PLANE_COV = submatrix(EMPIRICAL_COV, PLANE, PLANE);
const PPCA_PLANE_COV = submatrix(ppcaMarginalCov(PPCA.w, PPCA.sigma2), PLANE, PLANE);
const FA_PLANE_COV = submatrix(faMarginalCov(FA.w, FA.psi), PLANE, PLANE);

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
      <CovarianceEllipse mean={planeMean} cov={EMPIRICAL_PLANE_COV} levels={[MASS]} color={tokens.color.ink} width={1.5} />
      <CovarianceEllipse mean={planeMean} cov={PPCA_PLANE_COV} levels={[MASS]} color={tokens.series[0]!} width={2} dash="dashed" />
      <CovarianceEllipse mean={planeMean} cov={FA_PLANE_COV} levels={[MASS]} color={tokens.series[1]!} width={2} />
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
