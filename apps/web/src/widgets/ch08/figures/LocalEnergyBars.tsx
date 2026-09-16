import { isingEnergy, type Grid } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const PARAMS = { h: 0, beta: 1.5, eta: 2.1 };

function energyWithCenter(center: number): number {
  const base: Grid = [
    [-1, 1, -1],
    [1, center, 1],
    [-1, 1, -1],
  ];
  const y: Grid = [
    [-1, 1, -1],
    [1, 1, 1],
    [-1, 1, -1],
  ];
  return isingEnergy(base, y, PARAMS);
}

export default function LocalEnergyBars() {
  const tokens = useResolvedTokens();
  const energyMinus = energyWithCenter(-1);
  const energyPlus = energyWithCenter(1);
  const bars = [
    { at: 0, value: energyMinus, color: tokens.color.danger },
    { at: 1, value: energyPlus, color: tokens.color.accent },
  ];

  return (
    <Plot width={260} height={200} xDomain={[-0.6, 1.6]} yDomain={[Math.min(energyMinus, energyPlus) - 1, Math.max(energyMinus, energyPlus) + 1]} label="Total energy with the centre pixel at -1 versus +1, everything else fixed">
      <Axes y={{ label: 'energy' }} grid />
      <Bars bars={bars} thickness={0.5} />
    </Plot>
  );
}
