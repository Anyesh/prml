import { pcg32, standardNormal, swissRollPoint, type Mat } from '@prml/math';

const BASE_CLOUD_SEED = 20260916;
const RINGS_SEED = 20260917;
const SPIRAL_SEED = 20260918;
const ANISOTROPIC_SEED = 20260919;
const SPECTRUM_SEED = 20260921;
const SIGNAL_SEED = 20260922;

/** A standard 2D Gaussian point cloud, unit covariance, mean zero: the raw material `PcaAxesExplorer` reshapes and rotates live. */
export function baseCloud2D(n = 90): Mat {
  const rng = pcg32(BASE_CLOUD_SEED);
  return Array.from({ length: n }, () => [standardNormal(rng), standardNormal(rng)]);
}

/**
 * A low-rank signal plus isotropic noise: `z * direction + isotropic noise`, the exact
 * generative story PPCA assumes (PRML 12.31-12.33). `direction` is not an axis of the
 * ambient coordinate system, so the widget has to find it rather than read it off.
 */
export function ppcaDemoData(n = 60): Mat {
  const rng = pcg32(BASE_CLOUD_SEED, 2);
  const direction = [0.8, 0.6];
  const sigma = 0.4;
  return Array.from({ length: n }, () => {
    const z = standardNormal(rng) * 1.8;
    return [
      direction[0]! * z + sigma * standardNormal(rng),
      direction[1]! * z + sigma * standardNormal(rng),
    ];
  });
}

/**
 * The same low-rank signal as `ppcaDemoData`, but in 3D with per-axis noise variance
 * instead of isotropic. Only from D = 3 up is this a real restriction on PPCA: at D = 2,
 * PPCA's one loading vector plus one scalar noise variance already has as many free
 * parameters (3) as a general 2x2 covariance, so an isotropic and a diagonal-noise fit
 * cannot be told apart there. PRML 12.2.4's factor-analysis contrast needs this.
 */
export function anisotropicNoiseData(n = 200): Mat {
  const rng = pcg32(ANISOTROPIC_SEED);
  const direction = [0.7, 0.5, 0.5];
  const noiseStd: readonly [number, number, number] = [0.02, 0.8, 0.8];
  return Array.from({ length: n }, () => {
    const z = standardNormal(rng) * 1.8;
    return [
      direction[0]! * z + noiseStd[0]! * standardNormal(rng),
      direction[1]! * z + noiseStd[1]! * standardNormal(rng),
      direction[2]! * z + noiseStd[2]! * standardNormal(rng),
    ];
  });
}

/** Two concentric rings: linearly inseparable by any single projection, the standard case for kernel PCA (PRML 12.3). */
export function ringsData(perRing = 24): Mat {
  const rng = pcg32(RINGS_SEED);
  const points: number[][] = [];
  for (const radius of [1, 3]) {
    for (let i = 0; i < perRing; i++) {
      const theta = rng.next() * 2 * Math.PI;
      points.push([
        radius * Math.cos(theta) + 0.06 * standardNormal(rng),
        radius * Math.sin(theta) + 0.06 * standardNormal(rng),
      ]);
    }
  }
  return points;
}

/**
 * `swissRollPoint`'s `(x, z)` plane at fixed `height`: a 1D manifold rolled into 2D, so
 * it renders directly on a 2D `<Plot>` while still exercising the same embedding
 * `principalCurveFit`'s golden fixture uses in 3D.
 */
/** A 5D cloud with a sharply decaying, axis-aligned variance spectrum, for the eigenvalue-spectrum figure. */
export function spectrumDemoData(n = 100): Mat {
  const rng = pcg32(SPECTRUM_SEED);
  const std = [2.4, 1.6, 0.8, 0.35, 0.12];
  return Array.from({ length: n }, () => std.map((s) => s * standardNormal(rng)));
}

/**
 * `N` length-`dim` curves, each a random combination of two fixed smooth basis shapes
 * plus noise: intrinsically 2D data recorded in `dim` coordinates, so PCA's reconstruction
 * error should drop sharply once `M` reaches 2 and barely move after that.
 */
export function signalDemoData(n = 30, dim = 12): { data: Mat; basis1: number[]; basis2: number[] } {
  const rng = pcg32(SIGNAL_SEED);
  const positions = Array.from({ length: dim }, (_, i) => i / (dim - 1));
  const basis1 = positions.map((t) => Math.sin(2 * Math.PI * t));
  const basis2 = positions.map((t) => Math.cos(2 * Math.PI * t));
  const noiseStd = 0.08;
  const data = Array.from({ length: n }, () => {
    const a1 = standardNormal(rng);
    const a2 = standardNormal(rng);
    return positions.map((_, i) => a1 * basis1[i]! + a2 * basis2[i]! + noiseStd * standardNormal(rng));
  });
  return { data, basis1, basis2 };
}

export function spiralData(n = 40): { data: Mat; t: number[] } {
  const rng = pcg32(SPIRAL_SEED);
  const t: number[] = [];
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const ti = 2 + rng.next() * 4;
    const [x, , z] = swissRollPoint(ti, 0);
    t.push(ti);
    data.push([x! + 0.08 * standardNormal(rng), z! + 0.08 * standardNormal(rng)]);
  }
  return { data, t };
}
