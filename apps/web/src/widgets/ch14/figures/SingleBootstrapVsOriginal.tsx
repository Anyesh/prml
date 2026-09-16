import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { bootstrapIndices, committeeMemberRng, committeeSinusoidData } from '../data.js';
import '../../widgets.css';

const { x, t } = committeeSinusoidData();
const RESAMPLE = bootstrapIndices(committeeMemberRng(0), x.length);

function countOf(i: number): number {
  return RESAMPLE.filter((j) => j === i).length;
}

export default function SingleBootstrapVsOriginal() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[0, 1]} yDomain={[-1.6, 1.6]} label="Original data set against one bootstrap resample of it">
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      <ScatterField points={x.map((xi, i) => ({ x: xi, y: t[i]!, color: tokens.color.inkFaint, shape: 'ring', size: 4 }))} />
      <ScatterField
        points={x.map((xi, i) => ({ x: xi, y: t[i]!, color: tokens.color.accent, size: 2.5 + 2.5 * countOf(i) })).filter((p) => p.size > 2.5)}
      />
    </Plot>
  );
}
