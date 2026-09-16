import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

function curve(): [number, number][] {
  const points: [number, number][] = [];
  for (let i = 0; i <= 400; i++) {
    const rho = -0.995 + (1.99 * i) / 400;
    points.push([rho, 1 / (1 - rho * rho)]);
  }
  return points;
}

export default function StepsToIndependence() {
  const tokens = useResolvedTokens();
  return (
    <div className="widget-grid">
      <Plot width={420} height={240} xDomain={[-1, 1]} yDomain={[1, 1000]} yScaleKind="log" label="Sweeps to an independent sample against correlation, on a log y axis">
        <Axes x={{ label: 'correlation rho' }} y={{ label: '(L/l)^2, log scale' }} grid zeroLine />
        <Curve points={curve()} color={tokens.series[0]!} width={2.5} />
      </Plot>
      <p className="widget-readout">
        1/(1 - rho^2) stays near 1 for most of the range and only diverges as rho approaches
        plus or minus one: mild correlation costs Gibbs sampling almost nothing, extreme correlation costs it
        everything.
      </p>
    </div>
  );
}
