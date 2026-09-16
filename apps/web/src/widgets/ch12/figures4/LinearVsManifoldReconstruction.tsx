import { pcaFitCov, pcaReconstructionError, principalCurveFit, projectToPolyline } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import { spiralData } from '../data.js';

import '../../widgets.css';

const { data: DATA } = spiralData();
const PCA = pcaFitCov(DATA);
const LINEAR_ERROR = pcaReconstructionError(DATA, PCA.mean, PCA.components, 1);

const INITIAL_CURVE = Array.from({ length: 8 }, (_, i) => {
  const t = -6 + (12 * i) / 7;
  return [PCA.mean[0]! + t * PCA.components[0]![0]!, PCA.mean[1]! + t * PCA.components[0]![1]!];
});
const FIT = principalCurveFit(DATA, INITIAL_CURVE, 6, 1.4);
const FINAL_CURVE = FIT.curveHistory[FIT.curveHistory.length - 1]!;
const CURVE_ERROR =
  DATA.reduce((sum, x) => sum + projectToPolyline(x, FINAL_CURVE).distance ** 2, 0) / DATA.length;

export default function LinearVsManifoldReconstruction() {
  const tokens = useResolvedTokens();
  const maxError = Math.max(LINEAR_ERROR, CURVE_ERROR) * 1.15;

  return (
    <Plot height={200} xDomain={[-0.7, 1.7]} yDomain={[0, maxError]} label="Mean squared reconstruction error, one straight line against the fitted curve">
      <Axes x={{ label: '', ticks: [0, 1] }} y={{ label: 'error' }} grid />
      <Bars
        bars={[
          { at: 0, value: LINEAR_ERROR, color: tokens.series[0]! },
          { at: 1, value: CURVE_ERROR, color: tokens.series[1]! },
        ]}
        thickness={0.6}
      />
    </Plot>
  );
}
