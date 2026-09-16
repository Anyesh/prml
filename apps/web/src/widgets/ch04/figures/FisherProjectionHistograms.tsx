import { useMemo, useState } from 'react';
import { fisherDirection, linspace, mvnSample, pcg32, withinClassScatter } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../../widgets.css';

const MEAN1 = [-1.5, -1.5];
const MEAN2 = [1.5, 1.5];
const COV = [
  [2.15, 1.85],
  [1.85, 2.15],
];
const N = 60;
const rng = pcg32(20260411);
const CLASS1 = Array.from({ length: N }, () => mvnSample(rng, { mean: MEAN1, cov: COV }));
const CLASS2 = Array.from({ length: N }, () => mvnSample(rng, { mean: MEAN2, cov: COV }));

const FISHER_W = fisherDirection(MEAN1, MEAN2, withinClassScatter(CLASS1, CLASS2, MEAN1, MEAN2));
const FISHER_ANGLE = (Math.atan2(FISHER_W[1]!, FISHER_W[0]!) * 180) / Math.PI;
const MEAN_DIFF_ANGLE = (Math.atan2(MEAN2[1]! - MEAN1[1]!, MEAN2[0]! - MEAN1[0]!) * 180) / Math.PI;

const AXIS_DOMAIN: readonly [number, number] = [-6, 6];
const PROJ_GRID = linspace(-6, 6, 121);

function gaussianKde(values: readonly number[], grid: readonly number[], bandwidth: number): number[] {
  return grid.map((g) => {
    let sum = 0;
    for (const v of values) sum += Math.exp(-((g - v) ** 2) / (2 * bandwidth * bandwidth));
    return sum / (values.length * bandwidth * Math.sqrt(2 * Math.PI));
  });
}

export default function FisherProjectionHistograms() {
  const [angleDeg, setAngleDeg] = useState(Math.round(MEAN_DIFF_ANGLE));
  const tokens = useResolvedTokens();

  const direction = useMemo(() => {
    const rad = (angleDeg * Math.PI) / 180;
    return [Math.cos(rad), Math.sin(rad)] as const;
  }, [angleDeg]);

  const { proj1, proj2 } = useMemo(() => {
    const dot = (p: readonly number[]) => p[0]! * direction[0] + p[1]! * direction[1];
    return { proj1: CLASS1.map(dot), proj2: CLASS2.map(dot) };
  }, [direction]);

  const density1 = useMemo(() => gaussianKde(proj1, PROJ_GRID, 0.5), [proj1]);
  const density2 = useMemo(() => gaussianKde(proj2, PROJ_GRID, 0.5), [proj2]);
  const peak = Math.max(...density1, ...density2);

  const axisLine = [-6, 6].map((t) => [t * direction[0], t * direction[1]] as const);

  return (
    <div className="widget-grid">
      <Plot height={260} xDomain={AXIS_DOMAIN} yDomain={AXIS_DOMAIN} equalAspect label="Data with the current projection axis">
        <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid zeroLine />
        <Curve points={axisLine} color={tokens.color.inkMuted} width={1.5} dash="dashed" />
        <ScatterField points={CLASS1.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `1-${i}` }))} color={tokens.series[0]} size={3} />
        <ScatterField points={CLASS2.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `2-${i}` }))} color={tokens.series[1]} size={3} />
      </Plot>

      <Plot height={260} xDomain={AXIS_DOMAIN} yDomain={[0, peak * 1.1]} label="Projected values, density estimate per class">
        <Axes x={{ label: 'projection y = wᵀx' }} y={{ label: 'density' }} grid />
        <Curve points={PROJ_GRID.map((g, i) => [g, density1[i]!] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={PROJ_GRID.map((g, i) => [g, density2[i]!] as const)} color={tokens.series[1]!} width={2} />
        <Legend
          entries={[
            { label: 'class 1', color: tokens.series[0]!, mark: 'line' },
            { label: 'class 2', color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>

      <Slider
        label="Projection angle"
        value={angleDeg}
        onChange={setAngleDeg}
        min={0}
        max={179}
        step={1}
        format={(v) => `${v.toFixed(0)}°`}
        hint={`Mean-difference direction is ${MEAN_DIFF_ANGLE.toFixed(0)}°; Fisher's direction is ${FISHER_ANGLE.toFixed(0)}°.`}
      />
    </div>
  );
}
