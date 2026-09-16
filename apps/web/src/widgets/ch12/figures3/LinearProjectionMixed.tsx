import { pcaFitCov, pcaProject } from '@prml/math';
import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { ringsData } from '../data.js';

import '../../widgets.css';

const DATA = ringsData();
const PCA = pcaFitCov(DATA);
const PROJECTION = pcaProject(DATA, PCA.mean, PCA.components, 1).map((row) => row[0]!);
const N_PER_RING = 24;

export default function LinearProjectionMixed() {
  const tokens = useResolvedTokens();
  const points = PROJECTION.map((value, i) => ({
    x: value,
    y: i < N_PER_RING ? 0 : 1,
    color: i < N_PER_RING ? tokens.series[0]! : tokens.series[1]!,
  }));

  return (
    <Plot height={140} xDomain={[-3.5, 3.5]} yDomain={[-0.6, 1.6]} label="The first linear principal component's value for every point, split by which ring it belongs to">
      <Axes x={{ label: 'linear PC1' }} y={{ label: '', ticks: [0, 1] }} grid zeroLine />
      <ScatterField points={points} size={4} opacity={0.75} />
    </Plot>
  );
}
