import { kmeansFit } from '@prml/math';
import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

/** The exact seven points and initial means worked through in prose, also pinned by tools/golden/gen_kmeans.py. */
const DATA: number[][] = [
  [0, 0],
  [1, 0],
  [0, 1],
  [5, 5],
  [6, 5],
  [5, 6],
  [2.5, 2.5],
];
const INITIAL_MEANS: number[][] = [
  [0, 0],
  [5, 5],
];

const RESULT = kmeansFit(DATA, INITIAL_MEANS, 2);

export default function WorkedExampleTrace() {
  const tokens = useResolvedTokens();
  const finalAssignments = RESULT.assignmentsHistory[RESULT.assignmentsHistory.length - 1]!;
  const finalMeans = RESULT.means;

  return (
    <Plot width={460} height={460} xDomain={[-1, 7]} yDomain={[-1, 7]} equalAspect label="The seven-point worked example, coloured by its converged assignment">
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
      <ScatterField
        points={DATA.map((p, i) => ({ x: p[0]!, y: p[1]!, color: tokens.series[finalAssignments[i]!] }))}
        size={5}
        label={(_, i) => `Point ${i + 1}`}
      />
      <ScatterField
        points={INITIAL_MEANS.map((m) => ({ x: m[0]!, y: m[1]!, color: tokens.color.inkFaint, shape: 'ring', size: 9 }))}
        label={() => 'Initial mean'}
      />
      <ScatterField
        points={finalMeans.map((m, k) => ({ x: m[0]!, y: m[1]!, color: tokens.series[k], shape: 'cross', size: 10 }))}
        label={(_, k) => `Converged mean ${k + 1}`}
      />
    </Plot>
  );
}
