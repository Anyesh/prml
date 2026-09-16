import { useState } from 'react';
import { matmul, ppcaMLE } from '@prml/math';
import { Axes, Plot, ScatterField, VectorField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import { ppcaDemoData } from '../data.js';

import '../../widgets.css';

const DATA = ppcaDemoData();
const PARAMS = ppcaMLE(DATA, 2);
const DOMAIN: readonly [number, number] = [-4, 4];

function rotated(angleDeg: number) {
  const theta = (angleDeg * Math.PI) / 180;
  const r = [
    [Math.cos(theta), -Math.sin(theta)],
    [Math.sin(theta), Math.cos(theta)],
  ];
  return matmul(PARAMS.w, r);
}

export default function RotationalAmbiguity() {
  const tokens = useResolvedTokens();
  const [angle, setAngle] = useState(0);
  const w = rotated(angle);
  const mean = [PARAMS.mean[0]!, PARAMS.mean[1]!] as const;

  return (
    <div>
      <Plot height={260} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Two latent bases for the same marginal distribution, related by a rotation R">
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
        <ScatterField points={DATA.map((p) => ({ x: p[0]!, y: p[1]! }))} color={tokens.color.inkMuted} size={2.5} opacity={0.6} />
        <VectorField origins={[mean]} field={() => [w[0]![0]!, w[1]![0]!]} color={tokens.series[0]!} maxLength={100} />
        <VectorField origins={[mean]} field={() => [w[0]![1]!, w[1]![1]!]} color={tokens.series[1]!} maxLength={100} />
      </Plot>
      <Slider label="Rotation R (degrees)" value={angle} onChange={setAngle} min={0} max={180} step={5} hint="Every angle gives a different W with the same W W^T, so the same marginal distribution." />
    </div>
  );
}
