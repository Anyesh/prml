/**
 * The Ising-model de-noising illustration, PRML 8.3.3. A latent clean image `x` and an
 * observed noisy image `y`, both valued in {-1, +1}, coupled by a 4-neighbourhood energy
 * (8.42); iterated conditional modes minimises it one pixel at a time.
 */

export type Grid = readonly (readonly number[])[];

export interface IsingParams {
  /** Bias favouring x = -1 over x = +1 when positive; PRML sets this to 0 for an unbiased prior. */
  readonly h: number;
  /** Coupling strength between neighbouring latent pixels. Larger values smooth harder. */
  readonly beta: number;
  /** Coupling strength between a latent pixel and its own noisy observation. */
  readonly eta: number;
}

/**
 * PRML 8.42: `h * sum(x) - beta * sum over neighbouring pairs of x_i x_j - eta * sum(x * y)`.
 * Each undirected neighbour pair is counted once, by summing only the right and down edges.
 */
export function isingEnergy(x: Grid, y: Grid, params: IsingParams): number {
  const { h, beta, eta } = params;
  const rows = x.length;
  const cols = x[0]?.length ?? 0;
  let bias = 0;
  let data = 0;
  let coupling = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const xij = x[i]![j]!;
      bias += xij;
      data += xij * y[i]![j]!;
      if (j + 1 < cols) coupling += xij * x[i]![j + 1]!;
      if (i + 1 < rows) coupling += xij * x[i + 1]![j]!;
    }
  }
  return h * bias - beta * coupling - eta * data;
}

/**
 * One raster-order ICM sweep: each pixel is set to whichever of {-1, +1} minimises its
 * local energy given every other pixel's *current* value, including neighbours already
 * revisited earlier in this same sweep (Gauss-Seidel, not Jacobi, updates). A field of
 * exactly 0 favours +1, matching the golden fixture's tie-break.
 */
export function icmSweep(x: Grid, y: Grid, params: IsingParams): Grid {
  const { h, beta, eta } = params;
  const rows = x.length;
  const cols = x[0]?.length ?? 0;
  const next: number[][] = x.map((row) => [...row]);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let neighbourSum = 0;
      if (i > 0) neighbourSum += next[i - 1]![j]!;
      if (i + 1 < rows) neighbourSum += next[i + 1]![j]!;
      if (j > 0) neighbourSum += next[i]![j - 1]!;
      if (j + 1 < cols) neighbourSum += next[i]![j + 1]!;
      const field = beta * neighbourSum + eta * y[i]![j]! - h;
      next[i]![j] = field >= 0 ? 1 : -1;
    }
  }
  return next;
}

export interface IcmResult {
  /** `history[0]` is `initialX`; `history[s]` is the grid after sweep `s`. */
  readonly history: readonly Grid[];
  readonly energyHistory: readonly number[];
}

/**
 * Repeated ICM sweeps from `initialX`. Energy is non-increasing sweep to sweep, because
 * each pixel update can only lower or hold the total energy that pixel's terms contribute
 * and every other term is unaffected by a single-pixel flip; that monotonicity is what
 * lets the de-noising widget stop on "converged" rather than a fixed iteration count.
 */
export function icmDenoise(initialX: Grid, y: Grid, params: IsingParams, sweeps: number): IcmResult {
  const history: Grid[] = [initialX];
  const energyHistory: number[] = [isingEnergy(initialX, y, params)];
  let current = initialX;
  for (let s = 0; s < sweeps; s++) {
    current = icmSweep(current, y, params);
    history.push(current);
    energyHistory.push(isingEnergy(current, y, params));
  }
  return { history, energyHistory };
}
