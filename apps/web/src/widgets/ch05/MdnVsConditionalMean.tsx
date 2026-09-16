import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { DATASET, K, mdnParamsAt, regressionAt } from './mdnToyProblem';
import '../widgets.css';

const TEST_TS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75];

export default function MdnVsConditionalMean() {
  const tokens = useResolvedTokens();

  const regCurve = TEST_TS.map((t) => ({ x: t, y: regressionAt(t) }));
  const mdnModes = TEST_TS.flatMap((t) => {
    const params = mdnParamsAt(t);
    return Array.from({ length: K }, (_, k) => ({ x: t, y: params.means[k]![0]!, weight: params.mixing[k]! }));
  });

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={[0, 1]} yDomain={[0, 1]} equalAspect label="Training data with the conditional-mean network's single prediction curve">
        <Axes x={{ label: 't (input)' }} y={{ label: 'x (target)' }} grid />
        <ScatterField points={DATASET.inputs.map((t, i) => ({ x: t[0]!, y: DATASET.targets[i]![0]!, id: i, color: tokens.color.inkFaint, size: 2.5 }))} />
        <ScatterField points={regCurve.map((p, i) => ({ x: p.x, y: p.y, id: `reg-${i}`, color: tokens.color.danger, size: 5 }))} />
      </Plot>
      <Plot height={300} xDomain={[0, 1]} yDomain={[0, 1]} equalAspect label="Same data with the mixture density network's components, sized by mixing weight">
        <Axes x={{ label: 't (input)' }} y={{ label: 'x (target)' }} grid />
        <ScatterField points={DATASET.inputs.map((t, i) => ({ x: t[0]!, y: DATASET.targets[i]![0]!, id: i, color: tokens.color.inkFaint, size: 2.5 }))} />
        <ScatterField points={mdnModes.map((p, i) => ({ x: p.x, y: p.y, id: `mdn-${i}`, color: tokens.color.accent, size: 3 + p.weight * 7 }))} />
      </Plot>
      <p className="widget-readout">
        The conditional-mean network can only draw one curve, so through the ambiguous
        middle region it settles near the average of both branches, a value that fits
        neither. The mixture network keeps up to three candidate answers at every input,
        drawn larger the more probability its mixing coefficient gives them, and its most
        confident components sit near where the two true branches actually run.
      </p>
    </div>
  );
}
