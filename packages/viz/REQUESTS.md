# Primitive requests

A widget that needs a rendering capability `@prml/viz` does not have **stops and files a
request here**. It does not mount its own `<canvas>`, reach for raw `<svg>`, or inline a
one-off drawing helper in the widget file. Those three are what turn fourteen chapters
into fourteen different-looking sites, and the lint rules reject them.

Requests are absorbed into the primitives centrally between waves.

## Format

```
### <PrimitiveName>
- **Wanted by**: ch07/SvmMarginExplorer
- **Shape**: what it draws, in one sentence
- **Props**: the signature you would want to call
- **Why no existing primitive fits**: which one you tried first and where it fell short
```

## Open

### NetworkDiagram
- **Wanted by**: ch05/BackpropStepper, ch05/NetworkFunctionExplorer
- **Shape**: a node-link diagram laying out units in layers with weighted edges between them, so a forward and backward pass can be highlighted edge by edge the way the book's figures 5.1, 5.2 and 5.7 do
- **Props**: `{ layers: readonly number[]; edgeValue?: (from, to) => number; highlight?: ... }`
- **Why no existing primitive fits**: nothing lays out nodes and edges at all. Both widgets stand in with `ScatterField` for units and `Annotation` for the numbers on them, which reads as a scatter plot of dots rather than as a network, and cannot draw the edges that carry the deltas.

### Simplex
- **Wanted by**: ch02/DirichletSimplex, ch02/figures/DirichletCorners
- **Shape**: the 2-simplex as an equilateral triangle in barycentric coordinates, so a Dirichlet density can be drawn over it the way the book's figures 2.4 and 2.5 do
- **Props**: `{ vertices?: [string, string, string]; z?: number }` plus a projection helper turning a 3-vector that sums to one into plot coordinates
- **Why no existing primitive fits**: `Plot` publishes two linear scales, so the widget currently renders the simplex as the right triangle in the (mu1, mu2) plane and lets the third component stay implicit. That is a correct region but the wrong picture: it hides the symmetry between the three components, which is the whole point of the Dirichlet's concentration parameter.

## Absorbed

### LogScale
- **Wanted by**: ch05/figures/EigenvalueSpectrum, with prior workarounds in ch03, ch06 and ch09
- **Shape**: a logarithmic axis on `Plot`, for quantities spanning decades
- **Props**: `xScaleKind?: 'linear' | 'log'` and the same for y
- **Why no existing primitive fits**: `Plot` always built `scaleLinear`, so every figure spanning decades transformed its own data with `Math.log10` and labelled the axis `log10(x)`. That makes the reader translate exponents back into values, and it silently breaks `toData`, so pointer handling on such a plot returns the wrong coordinate.
- **Resolved**: added as `xScaleKind` / `yScaleKind`. `Frame.xScale` is now typed to the narrow
  `PlotScale` interface the primitives actually use rather than to `ScaleLinear`, so `scaleLog`
  satisfies it and nothing downstream of `Plot` changed. A log domain reaching zero or below
  throws, because the alternative is a plot full of silent NaNs.

### Annotation
- **Wanted by**: ch03/figures/ThreePointFit, ch03/figures/BiasVarianceAtAPoint
- **Shape**: a short text string drawn at one (x, y) in data coordinates, for a numeric callout next to a residual segment, a bias distance, or a variance spread
- **Props**: `{ x: number; y: number; text: string; color: string; anchor?: 'start' | 'middle' | 'end'; dy?: number }`
- **Why no existing primitive fits**: `Legend` only places a fixed-corner list of (colour, label) pairs, not a label anchored to an arbitrary drawn point, and `Axes` only labels its own ticks. Both figures worked around this by pushing the numbers into the widget's caption paragraph instead of drawing them on the plot itself.
- **Resolved**: added as `Annotation` in `primitives/Annotation.tsx`, with `plate` for
  legibility over a field, plus a `Rule` for reference lines at a fixed data value, which
  the same two figures needed and `Axes`'s zero-line could not provide.

### Bars
- **Wanted by**: ch01/figures/TwoBoxMarbles, ch01/figures/EntropyAcrossDistributions, ch01/figures/MutualInformationDiagram, ch02/figures/MLThreeHeads, ch02/figures/MultinomialPosteriorUpdate
- **Shape**: rectangles from a baseline to a value, one per category, for discrete probability distributions and comparison bars
- **Props**: `{ bars: { at, value, color, opacity? }[]; thickness?: number; baseline?: number; orientation?: 'vertical' | 'horizontal'; z?: number }`
- **Why no existing primitive fits**: `Curve` joins samples of a continuous function, which asserts a distance between neighbours that a category axis does not have, and `Heatmap` fills a grid rather than drawing to a baseline. Five figures across two chapters fell back to grids of styled `div`s, which pass lint but sit outside the shared scale, token and z-ordering system.
- **Resolved**: added as `Bars` in `primitives/Bars.tsx`. Categorical tick labels needed no new
  primitive: `Axes` already accepts explicit `ticks` and a `format` mapping a slot index to
  its label.

### Histogram
- **Wanted by**: ch04/figures/FisherProjectionHistograms
- **Shape**: per-class density bars over a shared one-dimensional axis, for comparing two projected distributions
- **Props**: `{ bins: number[]; counts: number[]; color: string }[]`
- **Why no existing primitive fits**: the figure substituted a hand-rolled Gaussian kernel density estimate drawn with `Curve`, which works but reimplements binning in every figure that wants it.
- **Resolved**: `Bars` covers this once the caller bins its own samples, which is the part
  that genuinely varies between figures. The chapter 4 figure kept its kernel density
  estimate because a smooth projected density is the better picture there, not because the
  primitive was missing.
