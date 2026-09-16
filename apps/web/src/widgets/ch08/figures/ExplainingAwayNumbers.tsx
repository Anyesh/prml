import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import { conditionalRainProbability } from '../sprinklerNetwork.js';
import '../../widgets.css';

export default function ExplainingAwayNumbers() {
  const tokens = useResolvedTokens();
  const prior = conditionalRainProbability({});
  const givenWet = conditionalRainProbability({ w: 1 });
  const givenWetAndSprinkler = conditionalRainProbability({ w: 1, s: 1 });

  const bars = [
    { at: 0, value: prior, color: tokens.color.inkFaint },
    { at: 1, value: givenWet, color: tokens.color.accent },
    { at: 2, value: givenWetAndSprinkler, color: tokens.color.danger },
  ];

  return (
    <Plot width={320} height={220} xDomain={[-0.6, 2.6]} yDomain={[0, 1]} label="P(rain) alone, given wet grass, and given wet grass and a running sprinkler">
      <Axes y={{ label: 'P(rain = 1)' }} grid />
      <Bars bars={bars} thickness={0.6} />
    </Plot>
  );
}
