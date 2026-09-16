import type { Mat, Vec } from '../types.js';
import { dot, norm, vecAdd, vecScale, vecSub } from '../linalg/core.js';

/**
 * The standard swiss-roll embedding (Marsland's convention): a 1D manifold rolled
 * through 3D, parametrised by arclength-like `t` plus a free `height` axis. Used only
 * to synthesise data for the manifold figures; the point of chapter 12.4 is recovering
 * `t` from `x`, not this closed form itself.
 */
export function swissRollPoint(t: number, height: number): Vec {
  return [t * Math.cos(t), height, t * Math.sin(t)];
}

function arclengths(polyline: Mat): number[] {
  const out = [0];
  let total = 0;
  for (let i = 1; i < polyline.length; i++) {
    total += norm(vecSub(polyline[i]!, polyline[i - 1]!));
    out.push(total);
  }
  return out;
}

export interface PolylineProjection {
  readonly arclength: number;
  readonly point: Vec;
  readonly distance: number;
}

/**
 * Nearest point on a piecewise-linear curve, by perpendicular projection onto each
 * segment clamped to `[0, 1]`, with the arclength of the closest such point (PRML
 * 12.4.3's projection index `g_f(x)`).
 */
export function projectToPolyline(point: Vec, polyline: Mat): PolylineProjection {
  const arcs = arclengths(polyline);
  let best: PolylineProjection = { arclength: arcs[0]!, point: polyline[0]!, distance: Infinity };
  for (let i = 0; i < polyline.length - 1; i++) {
    const a = polyline[i]!;
    const b = polyline[i + 1]!;
    const ab = vecSub(b, a);
    const denom = dot(ab, ab);
    const raw = denom === 0 ? 0 : dot(vecSub(point, a), ab) / denom;
    const t = Math.min(1, Math.max(0, raw));
    const proj = vecAdd(a, vecScale(ab, t));
    const distance = norm(vecSub(point, proj));
    if (distance < best.distance) {
      best = { arclength: arcs[i]! + t * (arcs[i + 1]! - arcs[i]!), point: proj, distance };
    }
  }
  return best;
}

export interface PrincipalCurveFitResult {
  readonly curveHistory: readonly Mat[];
}

/**
 * PRML 12.92's self-consistency condition, `E[x | g_f(x) = lambda] = f(lambda)`, applied
 * iteratively (Hastie & Stuetzle 1989): project every point onto the current curve to
 * get its arclength, then move each curve vertex to a Gaussian-kernel-weighted average
 * of the data, weighted by closeness in arclength to that vertex's own position. This is
 * a kernel-smoother in place of Hastie and Stuetzle's local regression, chosen because
 * it needs no extra machinery beyond what this module already has and stays a genuine
 * (if simplified) self-consistency iteration rather than a different algorithm.
 */
export function principalCurveFit(data: Mat, initialCurve: Mat, maxIters: number, bandwidth: number): PrincipalCurveFitResult {
  const curveHistory: Mat[] = [initialCurve];
  let curve = initialCurve;
  for (let iter = 0; iter < maxIters; iter++) {
    const curveArcs = arclengths(curve);
    const pointArcs = data.map((x) => projectToPolyline(x, curve).arclength);
    const dim = curve[0]?.length ?? 0;
    const next: number[][] = curve.map((vertex, i) => {
      const center = curveArcs[i]!;
      let weightSum = 0;
      const acc = new Array(dim).fill(0);
      pointArcs.forEach((a, n) => {
        const z = (a - center) / bandwidth;
        const w = Math.exp(-0.5 * z * z);
        weightSum += w;
        for (let d = 0; d < dim; d++) acc[d] += w * data[n]![d]!;
      });
      return weightSum < 1e-12 ? [...vertex] : acc.map((v) => v / weightSum);
    });
    curve = next;
    curveHistory.push(curve);
  }
  return { curveHistory };
}
