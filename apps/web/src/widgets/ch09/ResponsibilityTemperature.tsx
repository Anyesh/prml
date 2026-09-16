import { useMemo, useState } from 'react';
import { gmmEStep, kmeansFit, kmeansInit, pcg32, type GmmParams } from '@prml/math';
import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import { faithfulLikeData } from './data.js';
import { blendByResponsibility } from './colors.js';
import '../widgets.css';

export const title = 'From soft responsibility to hard indicator';
export const caption =
  'Drag epsilon down. The same two means now score every point by a shared, shrinking variance; watch the blended colours snap to the K-means split.';
export const figure = '9.3.2';

const DATA = faithfulLikeData();
const K = 2;
const SEED = 20260915;
const DOMAIN: readonly [number, number] = [-4, 4];
const LOG_EPSILON_MIN = -2.5;
const LOG_EPSILON_MAX = 1;

const MEANS = (() => {
  const rng = pcg32(SEED);
  const initial = kmeansInit(rng, DATA, K);
  return kmeansFit(DATA, initial, 10).means;
})();

function paramsAt(epsilon: number): GmmParams {
  const cov = [
    [epsilon, 0],
    [0, epsilon],
  ];
  return { components: MEANS.map((mean) => ({ weight: 1 / K, mean, cov })) };
}

export default function ResponsibilityTemperature() {
  const [logEpsilon, setLogEpsilon] = useState(0);
  const tokens = useResolvedTokens();
  const epsilon = 10 ** logEpsilon;

  const responsibilities = useMemo(() => gmmEStep(DATA, paramsAt(epsilon)), [epsilon]);
  const meanMaxResponsibility =
    responsibilities.reduce((sum, row) => sum + Math.max(...row), 0) / responsibilities.length;

  return (
    <div>
      <Plot width={480} height={480} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Points coloured by responsibility under a shrinking shared variance">
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
        <ScatterField
          points={DATA.map((p, i) => ({ x: p[0]!, y: p[1]!, color: blendByResponsibility(responsibilities[i]!, tokens.series), size: 4 }))}
        />
        <ScatterField points={MEANS.map((m, k) => ({ x: m[0]!, y: m[1]!, color: tokens.series[k], shape: 'cross', size: 9 }))} />
      </Plot>
      <Slider
        label="log10(epsilon)"
        value={logEpsilon}
        onChange={setLogEpsilon}
        min={LOG_EPSILON_MIN}
        max={LOG_EPSILON_MAX}
        step={0.02}
        format={(v) => (10 ** v).toFixed(3)}
        hint="Large epsilon: soft, blended colours. Small epsilon: every point commits to one mean."
      />
      <p className="widget-readout">
        epsilon = {epsilon.toFixed(3)}. Average of each point's largest responsibility: {meanMaxResponsibility.toFixed(3)}
        {meanMaxResponsibility > 0.999 ? ', indistinguishable from a hard 0/1 indicator.' : '.'}
      </p>
    </div>
  );
}
