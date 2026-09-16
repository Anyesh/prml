import type { Rng } from '../types.js';
import { categorical } from '../rng.js';

/**
 * PRML 11.1.3: a grid point's log-density and its derivative, `h(z) = ln p(z)` and
 * `h'(z)`. Adaptive rejection sampling only needs these two numbers per point, which is
 * what makes it applicable to any log-concave target without a hand-picked envelope.
 */
export interface ArsPoint {
  readonly z: number;
  readonly h: number;
  readonly hPrime: number;
}

/** One segment of the piecewise-exponential envelope (11.17): the tangent line at `z`, active on `[lo, hi]`. */
export interface ArsPiece {
  readonly lo: number;
  readonly hi: number;
  readonly z: number;
  readonly h: number;
  readonly hPrime: number;
}

export interface ArsEnvelope {
  readonly pieces: readonly ArsPiece[];
  /** Must be finite: an unbounded piece needs its tangent slope to carry the correct sign to stay integrable, which this implementation does not check for. */
  readonly domain: readonly [number, number];
}

function tangentHeight(p: { z: number; h: number; hPrime: number }, z: number): number {
  return p.h + p.hPrime * (z - p.z);
}

/** Where tangent lines at consecutive grid points cross, PRML Figure 11.6's construction. */
function intersect(a: ArsPoint, b: ArsPoint): number {
  if (a.hPrime === b.hPrime) {
    throw new Error('arsBuildEnvelope: two grid points have equal derivatives; the target is not strictly log-concave there');
  }
  return (b.h - a.h - b.z * b.hPrime + a.z * a.hPrime) / (a.hPrime - b.hPrime);
}

/**
 * Builds the upper envelope from a set of grid points: sorts them, intersects
 * consecutive tangent lines to get breakpoints, and assigns each grid point the piece
 * between its neighbouring breakpoints. Log-concavity (non-increasing `hPrime`) is
 * assumed, not checked, exactly as the book assumes it of the target.
 */
export function arsBuildEnvelope(points: readonly ArsPoint[], domain: readonly [number, number]): ArsEnvelope {
  const sorted = [...points].sort((a, b) => a.z - b.z);
  if (sorted.length < 1) throw new Error('arsBuildEnvelope: needs at least one grid point');
  const breakpoints: number[] = [domain[0]];
  for (let i = 0; i < sorted.length - 1; i++) breakpoints.push(intersect(sorted[i]!, sorted[i + 1]!));
  breakpoints.push(domain[1]);
  const pieces = sorted.map((p, i) => ({ lo: breakpoints[i]!, hi: breakpoints[i + 1]!, z: p.z, h: p.h, hPrime: p.hPrime }));
  return { pieces, domain };
}

export function arsEnvelopeLogHeight(envelope: ArsEnvelope, z: number): number {
  const piece = envelope.pieces.find((p) => z >= p.lo && z <= p.hi) ?? envelope.pieces[envelope.pieces.length - 1]!;
  return tangentHeight(piece, z);
}

const FLAT_SLOPE_EPS = 1e-10;

/**
 * Unnormalized area of one piece, `exp(height - shift)` integrated over `[lo, hi]`.
 * Exported so the golden test calls this exact function rather than a copy of its
 * arithmetic: a fixture compared against a second, hand-typed transcription in the test
 * body would only ever catch a bug in the transcription, never one here.
 */
export function arsPieceArea(piece: ArsPiece, shift: number): number {
  const loH = tangentHeight(piece, piece.lo) - shift;
  const hiH = tangentHeight(piece, piece.hi) - shift;
  if (Math.abs(piece.hPrime) < FLAT_SLOPE_EPS) {
    return Math.exp(loH) * (piece.hi - piece.lo);
  }
  return (Math.exp(hiH) - Math.exp(loH)) / piece.hPrime;
}

/**
 * Inverts the piece's cumulative area at fraction `t` in `[0, 1]` of that piece's own
 * area. Exported for the same reason as `arsPieceArea`: this is the one copy of the
 * inversion, called by both `arsEnvelopeSample` and its golden test.
 */
export function arsPieceQuantile(piece: ArsPiece, shift: number, t: number): number {
  if (Math.abs(piece.hPrime) < FLAT_SLOPE_EPS) {
    return piece.lo + t * (piece.hi - piece.lo);
  }
  const loH = tangentHeight(piece, piece.lo) - shift;
  const hiH = tangentHeight(piece, piece.hi) - shift;
  const target = loH + Math.log(1 - t + t * Math.exp(hiH - loH));
  return piece.z + (target + shift - piece.h) / piece.hPrime;
}

export interface ArsDraw {
  readonly z: number;
  readonly envelopeLogHeight: number;
}

/** Draws from the envelope directly: pick a piece by area (via the shared `categorical`), then invert within it. */
export function arsEnvelopeSample(rng: Rng, envelope: ArsEnvelope): ArsDraw {
  const shift = Math.max(...envelope.pieces.map((p) => p.h));
  const areas = envelope.pieces.map((p) => arsPieceArea(p, shift));
  const index = categorical(rng, areas);
  const piece = envelope.pieces[index]!;
  const t = rng.next();
  const z = arsPieceQuantile(piece, shift, t);
  return { z, envelopeLogHeight: tangentHeight(piece, z) };
}

export interface ArsStepResult {
  readonly accepted: boolean;
  readonly z: number;
  readonly envelope: ArsEnvelope;
}

/**
 * One draw-test-refine cycle. On rejection the candidate is folded into the grid and
 * the envelope rebuilt, exactly as PRML 11.1.3 describes: rejection makes the envelope
 * strictly better, so attempts are never wasted.
 */
export function arsStep(
  rng: Rng,
  envelope: ArsEnvelope,
  logTarget: (z: number) => number,
  dLogTarget: (z: number) => number,
): ArsStepResult {
  const { z, envelopeLogHeight } = arsEnvelopeSample(rng, envelope);
  const logU = Math.log(rng.next());
  const trueLog = logTarget(z);
  if (logU <= trueLog - envelopeLogHeight) {
    return { accepted: true, z, envelope };
  }
  const refined = arsBuildEnvelope(
    [...envelope.pieces.map((p) => ({ z: p.z, h: p.h, hPrime: p.hPrime })), { z, h: trueLog, hPrime: dLogTarget(z) }],
    envelope.domain,
  );
  return { accepted: false, z, envelope: refined };
}

export interface ArsResult {
  readonly sample: number;
  readonly envelope: ArsEnvelope;
  readonly attempts: number;
}

export interface ArsOptions {
  readonly logTarget: (z: number) => number;
  readonly dLogTarget: (z: number) => number;
  readonly initialGrid: readonly number[];
  readonly domain: readonly [number, number];
  readonly maxAttempts?: number;
}

export function arsSample(rng: Rng, opts: ArsOptions, envelope?: ArsEnvelope): ArsResult {
  let env =
    envelope ??
    arsBuildEnvelope(
      opts.initialGrid.map((z) => ({ z, h: opts.logTarget(z), hPrime: opts.dLogTarget(z) })),
      opts.domain,
    );
  const maxAttempts = opts.maxAttempts ?? 1000;
  for (let i = 0; i < maxAttempts; i++) {
    const step = arsStep(rng, env, opts.logTarget, opts.dLogTarget);
    env = step.envelope;
    if (step.accepted) return { sample: step.z, envelope: env, attempts: i + 1 };
  }
  throw new Error('arsSample: exceeded maxAttempts without an accepted sample');
}
