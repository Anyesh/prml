import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const BINS_PER_AXIS = 3;
const MAX_D = 10;
const POINTS = Array.from({ length: MAX_D }, (_, i) => {
  const d = i + 1;
  return [d, d * Math.log10(BINS_PER_AXIS)] as const;
});
const BINS_AT_MAX_D = Math.pow(BINS_PER_AXIS, MAX_D);

export default function CurseOfDimensionalityBins() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={200} xDomain={[1, MAX_D]} yDomain={[0, 5]} label="Number of histogram bins against dimension, on a log scale">
        <Axes x={{ label: 'dimension D' }} y={{ label: 'log₁₀(bin count)' }} grid />
        <Curve points={POINTS} color={tokens.color.danger} width={2} />
      </Plot>
      <p className="widget-readout">
        {`${BINS_PER_AXIS} bins per axis: a straight line here is Mᴰ growth. At D=${MAX_D} that is `}
        {`${BINS_AT_MAX_D.toLocaleString()} bins, almost all of them empty for any dataset you could collect.`}
      </p>
    </div>
  );
}
