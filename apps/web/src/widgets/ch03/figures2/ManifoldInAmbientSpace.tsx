import { useState } from 'react';
import { pcg32, standardNormal } from '@prml/math';
import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Toggle } from '@prml/ui';
import '../../widgets.css';

const DATA_SEED = 20260701;
const CENTRE_SEED_UNIFORM = 20260702;
const CENTRE_SEED_MANIFOLD = 20260703;
const DATA_COUNT = 90;
const CENTRE_COUNT = 60;
const DATA_JITTER = 0.02;
const CENTRE_JITTER = 0.04;
const NEAR_THRESHOLD = 0.05;

type Point = readonly [number, number];

/** Stand-in for the digit manifold: a curved one-dimensional path through the square. */
function manifold(t: number): Point {
  return [t, 0.5 + 0.3 * Math.sin(2 * Math.PI * t)];
}

function generateData(): Point[] {
  const rng = pcg32(DATA_SEED);
  return Array.from({ length: DATA_COUNT }, () => {
    const [x, y] = manifold(rng.next());
    return [x + DATA_JITTER * standardNormal(rng), y + DATA_JITTER * standardNormal(rng)] as const;
  });
}

function generateUniformCentres(): Point[] {
  const rng = pcg32(CENTRE_SEED_UNIFORM);
  return Array.from({ length: CENTRE_COUNT }, () => [rng.next(), rng.next()] as const);
}

function generateManifoldCentres(): Point[] {
  const rng = pcg32(CENTRE_SEED_MANIFOLD);
  return Array.from({ length: CENTRE_COUNT }, () => {
    const [x, y] = manifold(rng.next());
    return [x + CENTRE_JITTER * standardNormal(rng), y + CENTRE_JITTER * standardNormal(rng)] as const;
  });
}

const DATA = generateData();
const UNIFORM_CENTRES = generateUniformCentres();
const MANIFOLD_CENTRES = generateManifoldCentres();

function nearestDistance(point: Point, data: readonly Point[]): number {
  let best = Infinity;
  for (const d of data) {
    const dist = Math.hypot(point[0] - d[0], point[1] - d[1]);
    if (dist < best) best = dist;
  }
  return best;
}

function countNear(centres: readonly Point[]): number {
  return centres.filter((c) => nearestDistance(c, DATA) < NEAR_THRESHOLD).length;
}

const UNIFORM_NEAR = countNear(UNIFORM_CENTRES);
const MANIFOLD_NEAR = countNear(MANIFOLD_CENTRES);

export default function ManifoldInAmbientSpace() {
  const [onManifold, setOnManifold] = useState(false);
  const tokens = useResolvedTokens();

  const centres = onManifold ? MANIFOLD_CENTRES : UNIFORM_CENTRES;
  const near = onManifold ? MANIFOLD_NEAR : UNIFORM_NEAR;

  return (
    <div>
      <Plot
        height={260}
        xDomain={[0, 1]}
        yDomain={[0, 1]}
        equalAspect
        label="Data along a curved manifold inside a square of ambient space, with basis centres placed either uniformly or along the manifold"
      >
        <Axes x={{ label: 'dimension 1' }} y={{ label: 'dimension 2' }} grid />
        <ScatterField
          points={DATA.map((d, i) => ({ x: d[0], y: d[1], id: i }))}
          color={tokens.color.inkMuted}
          size={2.5}
          opacity={0.7}
        />
        <ScatterField
          points={centres.map((c, i) => ({ x: c[0], y: c[1], id: i }))}
          color={tokens.color.accent}
          shape="ring"
          size={5}
        />
      </Plot>
      <Toggle
        label="Place centres on the manifold"
        checked={onManifold}
        onChange={setOnManifold}
        hint="Off scatters basis centres uniformly over the square; on places them along the manifold instead."
      />
      <p className="widget-readout">
        {`${near} of ${CENTRE_COUNT} centres (${Math.round((100 * near) / CENTRE_COUNT)}%) sit within reach of the data`}
        {onManifold ? '.' : ', with most of the rest wasted on empty space the data never occupies.'}
      </p>
    </div>
  );
}
