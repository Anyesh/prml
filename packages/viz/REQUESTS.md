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

_(none yet)_

## Absorbed

_(none yet)_
