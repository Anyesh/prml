import { useMemo, useState } from 'react';
import {
  eigSym,
  exactHessian,
  forwardPass,
  inverse,
  mvnCovarianceEllipse,
  networkError,
  outerProductHessian,
  submatrix,
  symmetrise,
  unflattenWeights,
} from '@prml/math';
import { Axes, CovarianceEllipse, Plot, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { HESSIAN_DATASET, HESSIAN_SPEC, trainHessianToy } from './hessianToyNetwork';
import '../widgets.css';

const CHECKPOINTS = [0, 15, 150];
const PAIR: readonly [number, number] = [0, 1];
const MASS = 0.5;

function frobeniusRelativeGap(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a.length; j++) {
      const d = a[i]![j]! - b[i]![j]!;
      num += d * d;
      den += a[i]![j]! * a[i]![j]!;
    }
  }
  return Math.sqrt(num) / Math.sqrt(den);
}

function stageStats(steps: number) {
  const flat = trainHessianToy(steps);
  const weights = unflattenWeights(HESSIAN_SPEC, flat);
  const exact = exactHessian(HESSIAN_SPEC, weights, HESSIAN_DATASET, 'sumSquared');
  const approx = outerProductHessian(HESSIAN_SPEC, weights, HESSIAN_DATASET, 'sumSquared');
  const maxResidual = Math.max(...HESSIAN_DATASET.inputs.map((x, i) => Math.abs(forwardPass(HESSIAN_SPEC, weights, x).output[0]! - HESSIAN_DATASET.targets[i]![0]!)));
  const gap = frobeniusRelativeGap(exact, approx);
  const centre: [number, number] = [flat[PAIR[0]]!, flat[PAIR[1]]!];

  const exactSub = symmetrise(submatrix(exact, [...PAIR], [...PAIR]));
  const approxSub = symmetrise(submatrix(approx, [...PAIR], [...PAIR]));

  // Away from a minimum the exact Hessian is indefinite, so this block has no ellipse at
  // all: its contours are hyperbolae. The outer-product form is a sum of outer products
  // and so always positive semi-definite, which is the comparison this figure is making.
  const exactEigenvalues = eigSym(exactSub).values;
  const exactIsDefinite = exactEigenvalues.every((v) => v > 0);
  const exactCov = exactIsDefinite ? inverse(exactSub) : null;
  const approxCov = inverse(approxSub);

  return {
    maxResidual,
    gap,
    centre,
    exactCov,
    approxCov,
    exactEigenvalues,
    exactEllipse: exactCov ? mvnCovarianceEllipse({ mean: centre, cov: exactCov }, MASS) : null,
    approxEllipse: mvnCovarianceEllipse({ mean: centre, cov: approxCov }, MASS),
    error: networkError(HESSIAN_SPEC, weights, HESSIAN_DATASET, 'sumSquared'),
  };
}

export default function HessianEllipseExplorer() {
  const [stage, setStage] = useState(0);
  const tokens = useResolvedTokens();

  const stats = useMemo(() => CHECKPOINTS.map(stageStats), []);
  const current = stats[stage]!;
  const radii = [current.approxEllipse.rx, current.approxEllipse.ry];
  if (current.exactEllipse) radii.push(current.exactEllipse.rx, current.exactEllipse.ry);
  const bound = Math.max(...radii) * 1.6;
  const [cx, cy] = current.centre;

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[cx - bound, cx + bound]} yDomain={[cy - bound, cy + bound]} equalAspect label="Exact Hessian ellipse against the outer-product approximation">
        <Axes x={{ label: 'w (index 0)' }} y={{ label: 'w (index 1)' }} grid />
        {current.exactCov ? (
          <CovarianceEllipse mean={current.centre} cov={current.exactCov} levels={[MASS]} color={tokens.color.accent} width={2.5} />
        ) : null}
        <CovarianceEllipse mean={current.centre} cov={current.approxCov} levels={[MASS]} color={tokens.color.danger} width={2} dash="dashed" />
      </Plot>
      <StepThrough step={stage} stepCount={CHECKPOINTS.length} onStep={setStage} interval={900} />
      <p className="widget-readout">
        {`After ${CHECKPOINTS[stage]} training steps: error = ${current.error.toFixed(3)}, largest residual = ${current.maxResidual.toFixed(3)}. Relative gap between the exact Hessian and the outer-product approximation (full 13×13 matrix): ${(current.gap * 100).toFixed(1)}%.`}
        {current.exactCov
          ? ' Both curves are drawn: the exact Hessian is positive definite here, so it has an ellipse to compare against.'
          : ` Only the dashed approximation is drawn, because the exact Hessian's eigenvalues in this block are ${current.exactEigenvalues.map((v) => v.toFixed(4)).join(' and ')}. One is negative, so this is a saddle and the exact contours are hyperbolae, not an ellipse. The outer-product form is a sum of outer products and stays positive semi-definite everywhere, which is exactly why it is the one used away from a minimum.`}
      </p>
    </div>
  );
}
