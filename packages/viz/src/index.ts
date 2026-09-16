export type { Frame, Margin, Domain, CanvasDraw, CanvasRegistry } from './frame.js';
export { DEFAULT_MARGIN, NotImplemented } from './frame.js';

export { Plot, useFrame, useCanvasLayer, type PlotProps } from './Plot.js';
export { useResolvedTokens, useDrag, usePlotClick, type ResolvedTokens, type DragBindings } from './hooks.js';

export * from './primitives/Axes.js';
export * from './primitives/Curve.js';
export * from './primitives/Bars.js';
export * from './primitives/ColorScale.js';
export * from './primitives/Legend.js';
export * from './primitives/ContourField.js';
export * from './primitives/Heatmap.js';
export * from './primitives/ScatterField.js';
export * from './primitives/VectorField.js';
export * from './primitives/ClickSurface.js';
export * from './primitives/Annotation.js';
