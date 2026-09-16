import { covarianceMatrix, dataMean, pcaFitCov, trace } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { baseCloud2D } from '../data.js';

import '../../widgets.css';

const RAW = baseCloud2D();
const DATA = RAW.map(([x, y]) => [2.2 * x! - 0.5 * y!, 0.6 * x! + 0.9 * y!]);
const MEAN = dataMean(DATA);
const COV = covarianceMatrix(DATA, MEAN);
const TOTAL_VARIANCE = trace(COV);
const PCA = pcaFitCov(DATA);
const BEST_ANGLE = (Math.atan2(PCA.components[0]![1]!, PCA.components[0]![0]!) * 180) / Math.PI;

function varianceAt(angleDeg: number): number {
  const theta = (angleDeg * Math.PI) / 180;
  const u = [Math.cos(theta), Math.sin(theta)];
  return u[0]! * (COV[0]![0]! * u[0]! + COV[0]![1]! * u[1]!) + u[1]! * (COV[1]![0]! * u[0]! + COV[1]![1]! * u[1]!);
}

const ANGLES = Array.from({ length: 181 }, (_, i) => i - 90);
const VARIANCE_CURVE = ANGLES.map((a) => [a, varianceAt(a)] as const);
const ERROR_CURVE = ANGLES.map((a) => [a, TOTAL_VARIANCE - varianceAt(a)] as const);

export default function VarianceReconstructionDuality() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={240} xDomain={[-90, 90]} yDomain={[0, TOTAL_VARIANCE]} label="Variance captured and reconstruction error, as a single direction rotates through every angle">
      <Axes x={{ label: 'projection angle (degrees)' }} y={{ label: 'sum of squares' }} grid />
      <Curve points={VARIANCE_CURVE} color={tokens.series[0]!} width={2} />
      <Curve points={ERROR_CURVE} color={tokens.series[1]!} width={2} dash="dashed" />
      <Rule x={BEST_ANGLE} color={tokens.color.borderStrong} dash label="PC1" />
      <Legend
        entries={[
          { label: 'variance captured', color: tokens.series[0]!, mark: 'line' },
          { label: 'reconstruction error', color: tokens.series[1]!, mark: 'dashed-line' },
        ]}
      />
    </Plot>
  );
}
