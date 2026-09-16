import { useMemo, useState } from 'react';
import { pcaFitCov, type Mat } from '@prml/math';
import { Axes, Bars, Plot, ScatterField, VectorField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import { baseCloud2D } from './data.js';

import '../widgets.css';

export const title = 'Rotating and reshaping the cloud';
export const caption =
  'Drag the sliders to stretch and turn the point cloud. The two arrows always find the axes of greatest and least spread, however the cloud sits.';
export const figure = '12.1';

const BASE = baseCloud2D();
const DOMAIN: readonly [number, number] = [-5, 5];
const AXIS_PIXEL_LENGTH = 90;

function transform(data: Mat, angleDeg: number, stretchX: number, stretchY: number): Mat {
  const theta = (angleDeg * Math.PI) / 180;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return data.map(([x, y]) => {
    const sx = x! * stretchX;
    const sy = y! * stretchY;
    return [c * sx - s * sy, s * sx + c * sy];
  });
}

export default function PcaAxesExplorer() {
  const [angle, setAngle] = useState(20);
  const [stretchX, setStretchX] = useState(2.2);
  const [stretchY, setStretchY] = useState(0.7);
  const tokens = useResolvedTokens();

  const data = useMemo(() => transform(BASE, angle, stretchX, stretchY), [angle, stretchX, stretchY]);
  const pca = useMemo(() => pcaFitCov(data), [data]);
  const totalVariance = pca.eigenvalues[0]! + pca.eigenvalues[1]!;
  const mean: readonly [number, number] = [pca.mean[0]!, pca.mean[1]!];

  return (
    <div className="widget-grid">
      <Plot height={320} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Point cloud with its two principal axes, scaled by the square root of their eigenvalues">
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
        <ScatterField points={data.map((p, i) => ({ x: p[0]!, y: p[1]!, id: i }))} color={tokens.color.inkMuted} size={3} opacity={0.75} />
        <VectorField
          origins={[mean]}
          field={() => [pca.components[0]![0]! * Math.sqrt(pca.eigenvalues[0]!), pca.components[0]![1]! * Math.sqrt(pca.eigenvalues[0]!)]}
          color={tokens.series[0]!}
          maxLength={AXIS_PIXEL_LENGTH}
        />
        <VectorField
          origins={[mean]}
          field={() => [pca.components[1]![0]! * Math.sqrt(pca.eigenvalues[1]!), pca.components[1]![1]! * Math.sqrt(pca.eigenvalues[1]!)]}
          color={tokens.series[1]!}
          maxLength={AXIS_PIXEL_LENGTH * Math.sqrt(pca.eigenvalues[1]! / pca.eigenvalues[0]!)}
        />
      </Plot>

      <div>
        <Plot height={140} xDomain={[-0.7, 1.7]} yDomain={[0, totalVariance * 1.15]} label="Projected variance along each principal axis">
          <Axes x={{ label: '', ticks: [0, 1] }} y={{ label: 'variance' }} grid />
          <Bars
            bars={[
              { at: 0, value: pca.eigenvalues[0]!, color: tokens.series[0]! },
              { at: 1, value: pca.eigenvalues[1]!, color: tokens.series[1]! },
            ]}
            thickness={0.6}
          />
        </Plot>
        <Panel columns={1} dense>
          <Slider label="Rotation (degrees)" value={angle} onChange={setAngle} min={-90} max={90} step={1} />
          <Slider label="Spread along axis 1" value={stretchX} onChange={setStretchX} min={0.3} max={2.5} step={0.05} />
          <Slider label="Spread along axis 2" value={stretchY} onChange={setStretchY} min={0.3} max={2.5} step={0.05} />
          <p className="widget-readout">
            Eigenvalues {pca.eigenvalues[0]!.toFixed(2)} and {pca.eigenvalues[1]!.toFixed(2)}: the first axis alone carries{' '}
            {((100 * pca.eigenvalues[0]!) / totalVariance).toFixed(0)}% of the variance.
          </p>
        </Panel>
      </div>
    </div>
  );
}
