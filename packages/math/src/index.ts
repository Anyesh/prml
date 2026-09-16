export type { Vec, Mat, Rng, Family } from './types.js';
export { NotImplemented } from './types.js';

export * from './rng.js';
export * from './numeric.js';
export * from './special.js';
export * from './linalg/index.js';
export * from './distributions/index.js';
export * from './regression/index.js';

// One directory per chapter under construction, each owned end to end by the author of
// that chapter, so no two parallel authors ever write the same barrel.
export * from './information/index.js';
export * from './gaussian/index.js';
export * from './classification/index.js';
export * from './mixtures/index.js';
export * from './neural/index.js';
export * from './kernels/index.js';
export * from './sparse/index.js';
export * from './latent/index.js';
export * from './graphical/index.js';   // ch8
export * from './variational/index.js'; // ch10
export * from './sampling/index.js';    // ch11
export * from './sequential/index.js';  // ch13
export * from './ensemble/index.js';    // ch14
