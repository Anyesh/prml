import { useMemo, useState } from 'react';
import { markovChainSample, pcg32 } from '@prml/math';
import { Bars, Plot, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';

import '../widgets.css';

export const title = 'Persistence in a first-order chain';
export const caption =
  'Raise the persistence and watch runs of the same state lengthen. Press "New sample" at a fixed persistence to see the run length vary while its average tracks 1/(1-p).';
export const figure = '13.3';

const LENGTH = 60;

function meanRunLength(path: readonly number[]): number {
  let runs = 1;
  for (let i = 1; i < path.length; i++) {
    if (path[i] !== path[i - 1]) runs += 1;
  }
  return path.length / runs;
}

export default function MarkovChainSampler() {
  const tokens = useResolvedTokens();
  const [persistence, setPersistence] = useState(0.85);
  const [seedTick, setSeedTick] = useState(0);

  const path = useMemo(() => {
    const A = [
      [persistence, 1 - persistence],
      [1 - persistence, persistence],
    ];
    const rng = pcg32(20260917, seedTick + 1);
    return markovChainSample(rng, [0.5, 0.5], A, LENGTH);
  }, [persistence, seedTick]);

  const bars = path.map((state, t) => ({ at: t, value: 1, color: tokens.series[state]! }));
  const empirical = meanRunLength(path);
  const theoretical = 1 / (1 - persistence);

  return (
    <div>
      <Plot width={640} height={90} xDomain={[-0.5, LENGTH - 0.5]} yDomain={[0, 1]} label="Sampled state sequence, one bar per day">
        <Bars bars={bars} thickness={0.85} />
      </Plot>
      <Slider label="Persistence p = A11 = A22" value={persistence} onChange={setPersistence} min={0.5} max={0.98} step={0.01} />
      <button type="button" className="widget-reset" onClick={() => setSeedTick((t) => t + 1)}>
        New sample
      </button>
      <p className="widget-readout">
        Mean run length in this sample: {empirical.toFixed(2)} days. Expected run length
        1/(1-p) at this p: {theoretical.toFixed(2)} days.
      </p>
    </div>
  );
}
