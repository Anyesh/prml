import { fitDecisionStump } from '@prml/math';
import { Annotation, Axes, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { boostingToyData } from '../data.js';
import '../../widgets.css';

const { X, t } = boostingToyData(30);
const UNIFORM = new Array(X.length).fill(1 / X.length);
const { stump, weightedError } = fitDecisionStump(X, t, UNIFORM);

export default function WeakLearnerIllustration() {
  const tokens = useResolvedTokens();

  return (
    <Plot width={280} height={220} xDomain={[-2.2, 2.2]} yDomain={[-2.2, 2.2]} equalAspect label="One decision stump: a single axis-aligned threshold on overlapping data">
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
      {stump.featureIndex === 0 ? <Rule x={stump.threshold} /> : <Rule y={stump.threshold} />}
      <ScatterField points={X.map((x, i) => ({ x: x[0]!, y: x[1]!, color: tokens.series[t[i]! > 0 ? 0 : 1]!, size: 4 }))} />
      <Annotation x={-2} y={2} text={`weighted error ${weightedError.toFixed(2)}`} anchor="start" />
    </Plot>
  );
}
