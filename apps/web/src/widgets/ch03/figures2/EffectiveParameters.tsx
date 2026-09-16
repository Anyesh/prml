import { useState } from 'react';
import { designMatrix, eigSym, gaussianBasis, linspace, matmul, pcg32, standardNormal, transpose } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../../widgets.css';

const DATASET_SEED = 20260601;
const DATASET_SIZE = 25;
const NOISE_STD = 0.2;
const CENTRE_COUNT = 9;
const BASIS_SCALE = 0.12;
const BETA = 1 / (NOISE_STD * NOISE_STD);
const BAR_WIDTH_PX = 16;

interface Observation {
  readonly x: number;
  readonly t: number;
}

function generateDataset(): Observation[] {
  const rng = pcg32(DATASET_SEED);
  const xs = Array.from({ length: DATASET_SIZE }, () => rng.next());
  return xs.map((x) => ({ x, t: Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng) }));
}

const DATASET = generateDataset();
const XS = DATASET.map((p) => p.x);
const PHI = gaussianBasis(linspace(0, 1, CENTRE_COUNT), BASIS_SCALE, { bias: true });
const DESIGN = designMatrix(XS, PHI);
const DIMS = DESIGN[0]!.length;
// Eigenvalues of βΦᵀΦ, PRML 3.87: eigSym already returns them in descending order, so the
// bars come out sorted for free.
const GRAM_EIGENVALUES = eigSym(matmul(transpose(DESIGN), DESIGN)).values.map((e) => BETA * e);

export default function EffectiveParameters() {
  const [alpha, setAlpha] = useState(1);
  const tokens = useResolvedTokens();

  const ratios = GRAM_EIGENVALUES.map((lambda) => lambda / (alpha + lambda));
  const gamma = ratios.reduce((sum, r) => sum + r, 0);

  return (
    <div>
      <Plot
        height={220}
        xDomain={[-0.5, DIMS - 0.5]}
        yDomain={[0, 1]}
        label="Each ratio lambda over alpha plus lambda, sorted descending, PRML 3.91"
      >
        <Axes x={{ label: 'basis function, by λ rank', ticks: [] }} y={{ label: 'λ/(α+λ)' }} grid />
        {ratios.map((r, i) => (
          <Curve key={i} points={[[i, 0], [i, r]] as const} color={tokens.color.accent} width={BAR_WIDTH_PX} />
        ))}
      </Plot>
      <Panel columns={2} dense>
        <Slider
          label="Prior precision α"
          value={alpha}
          onChange={setAlpha}
          min={1e-2}
          max={1e3}
          scale="log"
          hint="A tighter prior pulls every ratio, and γ with them, toward zero."
        />
        <p className="widget-readout">{`γ = ${gamma.toFixed(2)} effective parameters out of M = ${DIMS} nominal ones.`}</p>
      </Panel>
    </div>
  );
}
