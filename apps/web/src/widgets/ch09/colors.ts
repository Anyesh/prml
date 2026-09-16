import { parseCssColor } from '@prml/viz';

/**
 * Mixes `colors` by `weights` in linear sRGB, the "mixed as ink" picture PRML uses for
 * responsibility-coloured points (Figure 9.5c): a point half-owned by two components
 * should read as visibly between their two series colours, not as one or the other.
 */
export function blendByResponsibility(weights: readonly number[], colors: readonly string[]): string {
  let r = 0;
  let g = 0;
  let b = 0;
  let total = 0;
  weights.forEach((w, i) => {
    const c = parseCssColor(colors[i] ?? colors[0]!);
    r += w * c.r;
    g += w * c.g;
    b += w * c.b;
    total += w;
  });
  if (total === 0) return colors[0]!;
  const to255 = (x: number) => Math.round(Math.min(1, Math.max(0, x / total)) * 255);
  return `rgb(${to255(r)}, ${to255(g)}, ${to255(b)})`;
}

/**
 * Snaps a field value to the nearest series colour rather than interpolating between
 * them, so a `Heatmap` of nearest-cluster index reads as flat Voronoi regions instead of
 * a smeared gradient between cluster labels that carry no order.
 */
export function categoricalInterpolator(colors: readonly string[]): (value: number) => string {
  return (value: number) => colors[Math.round(Math.min(Math.max(value, 0), colors.length - 1))] ?? colors[0]!;
}
