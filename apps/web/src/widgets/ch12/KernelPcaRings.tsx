import { useMemo, useState } from 'react';
import { kernelPcaFit, kernelPcaRbfKernel, kernelPcaTrainingProjections, pcaFitCov, pcaProject } from '@prml/math';
import { Axes, divergingScale, Plot, ScatterField } from '@prml/viz';
import { Slider } from '@prml/ui';
import { ringsData } from './data.js';

import '../widgets.css';

export const title = 'Kernel PCA on two rings';
export const caption =
  'Turn the kernel width. A single linear projection always mixes the rings; the first kernel component separates them once the width is small enough to see local structure.';
export const figure = '12.3';

const DATA = ringsData();
const DOMAIN: readonly [number, number] = [-4, 4];
const LINEAR_PCA = pcaFitCov(DATA);
const LINEAR_PROJECTION = pcaProject(DATA, LINEAR_PCA.mean, LINEAR_PCA.components, 1).map((row) => row[0]!);

export default function KernelPcaRings() {
  const [gamma, setGamma] = useState(0.25);

  const model = useMemo(() => kernelPcaFit(DATA, kernelPcaRbfKernel(gamma), 1), [gamma]);
  const projections = useMemo(() => kernelPcaTrainingProjections(model).map((row) => row[0]!), [model]);

  const maxAbs = Math.max(...projections.map((v) => Math.abs(v)), 1e-9);
  const colorAt = divergingScale([-maxAbs, maxAbs]);
  const linearMaxAbs = Math.max(...LINEAR_PROJECTION.map((v) => Math.abs(v)), 1e-9);
  const linearColorAt = divergingScale([-linearMaxAbs, linearMaxAbs]);

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Rings coloured by the first linear principal component">
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
        <ScatterField points={DATA.map((p, i) => ({ x: p[0]!, y: p[1]!, color: linearColorAt(LINEAR_PROJECTION[i]!), size: 4 }))} />
      </Plot>
      <div>
        <Plot height={300} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Rings coloured by the first kernel principal component">
          <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
          <ScatterField points={DATA.map((p, i) => ({ x: p[0]!, y: p[1]!, color: colorAt(projections[i]!), size: 4 }))} />
        </Plot>
        <Slider label="Kernel width (gamma)" value={gamma} onChange={setGamma} min={0.02} max={1} step={0.01} hint="Small gamma: each point only feels its near neighbours, which is what separates the rings." />
        <p className="widget-readout">
          Left: one straight line through the data cannot tell inner from outer. Right: the kernel component's colour follows the ring, not the direction from the centre.
        </p>
      </div>
    </div>
  );
}
