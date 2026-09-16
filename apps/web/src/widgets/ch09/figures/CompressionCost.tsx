import { Annotation, Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

/** A synthetic 80x60 RGB image: 4800 pixels at 24 bits each, the baseline every K below is compared against. */
const PIXEL_COUNT = 80 * 60;
const BITS_PER_PIXEL_RAW = 24;
const RAW_BITS = PIXEL_COUNT * BITS_PER_PIXEL_RAW;
const K_VALUES = [2, 4, 8, 16, 32, 64];

function compressedBits(k: number): number {
  return BITS_PER_PIXEL_RAW * k + PIXEL_COUNT * Math.log2(k);
}

const RATIOS = K_VALUES.map((k) => (compressedBits(k) / RAW_BITS) * 100);

export default function CompressionCost() {
  const tokens = useResolvedTokens();
  const points = K_VALUES.map((k, i) => [Math.log2(k), RATIOS[i]!] as const);

  return (
    <Plot height={220} xDomain={[0, 7]} yDomain={[0, Math.max(...RATIOS) * 1.2]} label="Compressed size as a percentage of the raw image, against K on a log scale">
      <Axes
        x={{ label: 'K (log2 scale)', ticks: K_VALUES.map((k) => Math.log2(k)), format: (v) => String(2 ** v) }}
        y={{ label: '% of raw size' }}
        grid
      />
      <ScatterField points={points.map(([x, y]) => ({ x, y, color: tokens.color.accent }))} size={5} label={(_, i) => `K=${K_VALUES[i]}`} />
      {points.map(([x, y], i) => (
        <Annotation key={i} x={x} y={y} text={`${RATIOS[i]!.toFixed(1)}%`} dy={-14} />
      ))}
    </Plot>
  );
}
