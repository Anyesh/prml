import { linspace, swissRollPoint } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { spiralData } from '../data.js';

import '../../widgets.css';

const { data: DATA } = spiralData();
const DOMAIN: readonly [number, number] = [-7, 7];
const TS = linspace(2, 6, 80);
const TRUE_CURVE = TS.map((t) => {
  const [x, , z] = swissRollPoint(t, 0);
  return [x!, z!] as const;
});

export default function ManifoldAmbientView() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={280} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Data recorded in two ambient coordinates, lying near a one-dimensional curve">
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
      <ScatterField points={DATA.map((p) => ({ x: p[0]!, y: p[1]! }))} color={tokens.color.inkMuted} size={3.5} />
      <Curve points={TRUE_CURVE} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
