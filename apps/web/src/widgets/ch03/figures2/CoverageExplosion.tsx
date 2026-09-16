import { useState } from 'react';
import { linspace } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../../widgets.css';

const S_MIN = 0.08;
const S_MAX = 0.5;
const DEFAULT_S = 0.2;
const MAX_D = 10;
/** "perhaps 10^80 atoms in the observable universe", from the worked example in this section. */
const ATOMS_LOG10 = 80;
const D_RANGE = Array.from({ length: MAX_D }, (_, i) => i + 1);

function countPerAxis(s: number): number {
  return Math.max(1, Math.round(1 / s));
}

export default function CoverageExplosion() {
  const [s, setS] = useState(DEFAULT_S);
  const tokens = useResolvedTokens();

  const count1D = countPerAxis(s);
  const centres = linspace(0, 1, count1D);
  const count2D = count1D * count1D;
  const log10Counts = D_RANGE.map((d) => d * Math.log10(count1D));
  const yTop = Math.max(ATOMS_LOG10 * 1.05, log10Counts[log10Counts.length - 1]! * 1.3);

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
      <div>
        <Plot height={220} xDomain={[0, 1]} yDomain={[-1, 1]} label={`Line covered by ${count1D} bumps at spacing ${s.toFixed(2)}`}>
          <Axes x={{ label: 'x', bare: true }} y={false} />
          <ScatterField points={centres.map((c, i) => ({ x: c, y: 0, id: i }))} color={tokens.color.accent} size={5} />
        </Plot>
        <p className="widget-readout">{`${count1D} bumps cover the line.`}</p>
      </div>

      <div>
        <Plot
          height={220}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          equalAspect
          label={`Square covered by ${count2D} bumps at the same spacing`}
        >
          <Axes x={{ label: 'x1', bare: true }} y={{ label: 'x2', bare: true }} />
          <ScatterField
            points={centres.flatMap((cx, i) => centres.map((cy, j) => ({ x: cx, y: cy, id: `${i}-${j}` })))}
            color={tokens.color.accent}
            size={3}
          />
        </Plot>
        <p className="widget-readout">{`${count1D}² = ${count2D} bumps cover the square.`}</p>
      </div>

      <div>
        <Plot height={220} xDomain={[1, MAX_D]} yDomain={[0, yTop]} label="Bump count for D from 1 to 10, on a log axis">
          <Axes x={{ label: 'D', ticks: D_RANGE }} y={{ label: 'log₁₀(count)' }} grid />
          <Curve points={D_RANGE.map((d, i) => [d, log10Counts[i]!] as const)} color={tokens.color.accent} width={2} />
          <Curve points={[[1, ATOMS_LOG10], [MAX_D, ATOMS_LOG10]] as const} color={tokens.color.danger} dash="dashed" width={1.5} />
        </Plot>
        <p className="widget-readout">
          {`At D=${MAX_D}, ${count1D}^${MAX_D} ≈ 10^${log10Counts[log10Counts.length - 1]!.toFixed(1)} bumps: already `}
          {'more than anyone could compute with, yet nowhere near the dashed line, the ~10^80 atoms in the observable universe that D=784 would demand.'}
        </p>
      </div>

      <Panel columns={1} dense>
        <Slider
          label="Bump spacing s"
          value={s}
          onChange={setS}
          min={S_MIN}
          max={S_MAX}
          hint="Smaller spacing needs more bumps per axis, and that count gets raised to the power of D."
        />
      </Panel>
    </div>
  );
}
