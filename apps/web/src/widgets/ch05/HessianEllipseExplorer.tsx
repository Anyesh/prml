import { useMemo, useState } from 'react';
import {
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
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { HESSIAN_DATASET, HESSIAN_SPEC, trainHessianToy } from './hessianToyNetwork';
import '../widgets.css';

const CHECKPOINTS = [0, 15, 150];
const PAIR: readonly [number, number] = [0, 1];

function ellipsePoints(cx: number, cy: number, rx: number, ry: number, angle: number): (readonly [number, number])[] {
  const n = 64;
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    return [cx + x * Math.cos(angle) - y * Math.sin(angle), cy + x * Math.sin(angle) + y * Math.cos(angle)] as const;
  });
}

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
  const exactEllipse = mvnCovarianceEllipse({ mean: centre, cov: inverse(exactSub) }, 0.5);
  const approxEllipse = mvnCovarianceEllipse({ mean: centre, cov: inverse(approxSub) }, 0.5);

  return { maxResidual, gap, exactEllipse, approxEllipse, error: networkError(HESSIAN_SPEC, weights, HESSIAN_DATASET, 'sumSquared') };
}

export default function HessianEllipseExplorer() {
  const [stage, setStage] = useState(0);
  const tokens = useResolvedTokens();

  const stats = useMemo(() => CHECKPOINTS.map(stageStats), []);
  const current = stats[stage]!;
  const bound = Math.max(current.exactEllipse.rx, current.exactEllipse.ry, current.approxEllipse.rx, current.approxEllipse.ry) * 1.6;
  const cx = current.exactEllipse.cx;
  const cy = current.exactEllipse.cy;

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[cx - bound, cx + bound]} yDomain={[cy - bound, cy + bound]} equalAspect label="Exact Hessian ellipse against the outer-product approximation">
        <Axes x={{ label: 'w (index 0)' }} y={{ label: 'w (index 1)' }} grid />
        <Curve
          points={ellipsePoints(current.exactEllipse.cx, current.exactEllipse.cy, current.exactEllipse.rx, current.exactEllipse.ry, current.exactEllipse.angle)}
          color={tokens.color.accent}
          width={2.5}
        />
        <Curve
          points={ellipsePoints(current.approxEllipse.cx, current.approxEllipse.cy, current.approxEllipse.rx, current.approxEllipse.ry, current.approxEllipse.angle)}
          color={tokens.color.danger}
          width={2}
          dash="dashed"
        />
      </Plot>
      <StepThrough step={stage} stepCount={CHECKPOINTS.length} onStep={setStage} interval={900} />
      <p className="widget-readout">
        {`After ${CHECKPOINTS[stage]} training steps: error = ${current.error.toFixed(3)}, largest residual = ${current.maxResidual.toFixed(3)}. Relative gap between the exact Hessian and the outer-product approximation (full 13×13 matrix): ${(current.gap * 100).toFixed(1)}%.`}
      </p>
    </div>
  );
}
