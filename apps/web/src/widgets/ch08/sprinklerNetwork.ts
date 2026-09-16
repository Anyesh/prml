/** The cloudy/sprinkler/rain/wet-grass network shared by AncestralSampling and ExplainingAwayNumbers. */

export function pCloudy(): number {
  return 0.5;
}
export function pSprinkler(cloudy: 0 | 1): number {
  return cloudy === 1 ? 0.1 : 0.5;
}
export function pRain(cloudy: 0 | 1): number {
  return cloudy === 1 ? 0.8 : 0.2;
}
export function pWetGrass(sprinkler: 0 | 1, rain: 0 | 1): number {
  if (sprinkler === 0 && rain === 0) return 0.01;
  if (sprinkler === 1 && rain === 0) return 0.9;
  if (sprinkler === 0 && rain === 1) return 0.9;
  return 0.99;
}

function bernoulli(x: 0 | 1, p: number): number {
  return x === 1 ? p : 1 - p;
}

/** Every joint assignment's probability, for exact brute-force conditioning (16 states, cheap enough to enumerate directly). */
export function enumerateJoint(): readonly { c: 0 | 1; s: 0 | 1; r: 0 | 1; w: 0 | 1; probability: number }[] {
  const states: (0 | 1)[] = [0, 1];
  const rows: { c: 0 | 1; s: 0 | 1; r: 0 | 1; w: 0 | 1; probability: number }[] = [];
  for (const c of states) {
    for (const s of states) {
      for (const r of states) {
        for (const w of states) {
          const probability =
            bernoulli(c, pCloudy()) * bernoulli(s, pSprinkler(c)) * bernoulli(r, pRain(c)) * bernoulli(w, pWetGrass(s, r));
          rows.push({ c, s, r, w, probability });
        }
      }
    }
  }
  return rows;
}

/** P(rain = 1) among rows matching every key in `given`, i.e. a plain conditional from the joint above. */
export function conditionalRainProbability(given: Partial<{ s: 0 | 1; w: 0 | 1 }>): number {
  const rows = enumerateJoint().filter(
    (row) => (given.s === undefined || row.s === given.s) && (given.w === undefined || row.w === given.w),
  );
  const total = rows.reduce((sum, row) => sum + row.probability, 0);
  const rainy = rows.filter((row) => row.r === 1).reduce((sum, row) => sum + row.probability, 0);
  return rainy / total;
}
