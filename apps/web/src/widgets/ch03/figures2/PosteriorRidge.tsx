import { useMemo } from 'react';
import {
  eigSym,
  evalGrid,
  isotropicPrior,
  linspace,
  mvnLogPdf,
  polynomialBasis,
  updatePosterior,
} from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, VectorField, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const ALPHA = 2;
const BETA = 25;
const OBS_X = 0.9;
const OBS_T = 0.1;
const PHI = polynomialBasis(1);
const GRID = linspace(-2, 2, 60);
const DOMAIN: readonly [number, number] = [-2, 2];
const ARROW_BASE_PX = 44;

const PRIOR = isotropicPrior(2, ALPHA);
const POSTERIOR = updatePosterior(PRIOR, PHI(OBS_X), OBS_T, BETA);

// Eigenvalues of the posterior *precision*, not the covariance: PRML's worked example
// reports how sharply each direction is pinned down, and precision is what grows large
// in the direction the single observation actually constrains.
const PRECISION_EIGEN = eigSym(POSTERIOR.precision);
const [LAMBDA_1, LAMBDA_2] = PRECISION_EIGEN.values as [number, number];
const [EIGVEC_1, EIGVEC_2] = PRECISION_EIGEN.vectors as [number[], number[]];
const MEAN_POINT = [POSTERIOR.mean[0]!, POSTERIOR.mean[1]!] as const;

const PRIOR_DENSITY = evalGrid(GRID, GRID, (w0, w1) => Math.exp(mvnLogPdf([w0, w1], PRIOR)));
const POSTERIOR_DENSITY = evalGrid(GRID, GRID, (w0, w1) => Math.exp(mvnLogPdf([w0, w1], POSTERIOR)));

export default function PosteriorRidge() {
  const tokens = useResolvedTokens();

  const fill = useMemo(() => {
    let peak = 0;
    for (const row of POSTERIOR_DENSITY.values) for (const v of row) if (v > peak) peak = v;
    return sequentialScale([0, peak], tokens.sequential);
  }, [tokens.sequential]);

  return (
    <div>
      <Plot
        height={260}
        xDomain={DOMAIN}
        yDomain={DOMAIN}
        equalAspect
        label="Prior contours and the posterior ridge after one observation, with the precision eigenvectors"
      >
        <ContourField data={PRIOR_DENSITY} levelCount={4} color={tokens.color.inkFaint} opacity={0.7} z={-2} />
        <ContourField data={POSTERIOR_DENSITY} levelCount={7} color={tokens.color.accent} fill={fill} z={-1} />
        <Axes x={{ label: 'w₀' }} y={{ label: 'w₁' }} grid zeroLine />
        <VectorField
          origins={[MEAN_POINT]}
          field={() => [EIGVEC_1[0]!, EIGVEC_1[1]!] as const}
          color={tokens.color.danger}
          maxLength={ARROW_BASE_PX}
        />
        <VectorField
          origins={[MEAN_POINT]}
          field={() => [EIGVEC_2[0]!, EIGVEC_2[1]!] as const}
          color={tokens.color.danger}
          maxLength={ARROW_BASE_PX * (LAMBDA_2 / LAMBDA_1)}
        />
        <ScatterField points={[{ x: MEAN_POINT[0], y: MEAN_POINT[1], color: tokens.color.ink, size: 3 }]} />
      </Plot>
      <p className="widget-readout">
        {`One point at x=${OBS_X}, t=${OBS_T} turns the circular prior into a ridge. The posterior precision's `}
        {`two eigenvalues are λ₁=${LAMBDA_1.toFixed(2)} and λ₂=${LAMBDA_2.toFixed(2)}: tightly pinned along the `}
        {'long arrow, barely constrained along the short one, which is why the contour above is elongated rather than circular.'}
      </p>
    </div>
  );
}
