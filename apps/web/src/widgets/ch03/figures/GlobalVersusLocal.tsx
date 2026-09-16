import { useRef, useState } from 'react';
import { designMatrix, dot, gaussianBasis, linspace, maximumLikelihoodWeights, polynomialBasis } from '@prml/math';
import { Axes, FunctionCurve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N_POINTS = 10;
const DOMAIN: readonly [number, number] = [-1, 1];
const Y_DOMAIN: readonly [number, number] = [-2, 2];
const GAUSSIAN_CENTRES = linspace(-1, 1, N_POINTS - 1);
const GAUSSIAN_SCALE = 0.28;
const POLY_BASIS = polynomialBasis(N_POINTS - 1);
const GAUSSIAN_BASIS = gaussianBasis(GAUSSIAN_CENTRES, GAUSSIAN_SCALE);

/** A new drag reads as a gap of silence longer than any two pointermove ticks within one drag. */
const NEW_DRAG_GAP_MS = 200;

interface Point {
  readonly x: number;
  readonly y: number;
}

function initialPoints(): Point[] {
  return linspace(-1, 1, N_POINTS).map((x) => ({ x, y: 0.8 * Math.sin(Math.PI * x) + 0.15 * x }));
}

const INITIAL = initialPoints();

function fitCurve(points: readonly Point[], basis: (x: number) => number[]): (x: number) => number {
  const design = designMatrix(
    points.map((p) => p.x),
    basis,
  );
  const weights = maximumLikelihoodWeights(
    design,
    points.map((p) => p.y),
  );
  return (x: number) => dot(basis(x), weights);
}

export default function GlobalVersusLocal() {
  const [points, setPoints] = useState<Point[]>(INITIAL);
  const [previous, setPrevious] = useState<Point[]>(INITIAL);
  const lastMoveAt = useRef(0);
  const tokens = useResolvedTokens();

  function handleMove(index: number, x: number, y: number) {
    const now = performance.now();
    if (now - lastMoveAt.current > NEW_DRAG_GAP_MS) setPrevious(points);
    lastMoveAt.current = now;
    setPoints((current) => current.map((p, i) => (i === index ? { x, y } : p)));
  }

  const polyCurve = fitCurve(points, POLY_BASIS);
  const gaussCurve = fitCurve(points, GAUSSIAN_BASIS);
  const prevPolyCurve = fitCurve(previous, POLY_BASIS);
  const prevGaussCurve = fitCurve(previous, GAUSSIAN_BASIS);
  const changed = points.some((p, i) => p.x !== previous[i]!.x || p.y !== previous[i]!.y);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))', gap: 'var(--prml-space-4)' }}>
      <Plot height={220} xDomain={DOMAIN} yDomain={Y_DOMAIN} label="Degree-9 polynomial fit, draggable points">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        {changed && <FunctionCurve f={prevPolyCurve} domain={DOMAIN} color={tokens.color.inkFaint} dash="dashed" width={1.25} />}
        <FunctionCurve f={polyCurve} domain={DOMAIN} color={tokens.series[0]!} width={2} />
        <ScatterField
          points={points.map((p, i) => ({ x: p.x, y: p.y, id: i }))}
          color={tokens.color.ink}
          onMove={handleMove}
          label={(_, i) => `point ${i + 1}`}
        />
      </Plot>
      <Plot height={220} xDomain={DOMAIN} yDomain={Y_DOMAIN} label="Nine-Gaussian-bump fit, the same draggable points">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        {changed && <FunctionCurve f={prevGaussCurve} domain={DOMAIN} color={tokens.color.inkFaint} dash="dashed" width={1.25} />}
        <FunctionCurve f={gaussCurve} domain={DOMAIN} color={tokens.series[1]!} width={2} />
        <ScatterField
          points={points.map((p, i) => ({ x: p.x, y: p.y, id: i }))}
          color={tokens.color.ink}
          onMove={handleMove}
          label={(_, i) => `point ${i + 1}`}
        />
      </Plot>
      <button
        type="button"
        className="widget-reset"
        onClick={() => {
          setPoints(INITIAL);
          setPrevious(INITIAL);
        }}
      >
        Reset points
      </button>
    </div>
  );
}
