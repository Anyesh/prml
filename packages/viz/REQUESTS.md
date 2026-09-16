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

_(none)_

## Absorbed

### Annotation
- **Wanted by**: ch03/figures/ThreePointFit, ch03/figures/BiasVarianceAtAPoint
- **Shape**: a short text string drawn at one (x, y) in data coordinates, for a numeric callout next to a residual segment, a bias distance, or a variance spread
- **Props**: `{ x: number; y: number; text: string; color: string; anchor?: 'start' | 'middle' | 'end'; dy?: number }`
- **Why no existing primitive fits**: `Legend` only places a fixed-corner list of (colour, label) pairs, not a label anchored to an arbitrary drawn point, and `Axes` only labels its own ticks. Both figures worked around this by pushing the numbers into the widget's caption paragraph instead of drawing them on the plot itself.
- **Resolved**: added as `Annotation` in `primitives/Annotation.tsx`, with `plate` for
  legibility over a field, plus a `Rule` for reference lines at a fixed data value, which
  the same two figures needed and `Axes`'s zero-line could not provide.
