import { useMemo, useState } from 'react';
import {
  eigSym,
  exponentialKernel,
  gramMatrix,
  linearKernel,
  linspace,
  polynomialKernel,
  productKernel,
  rbfKernel,
  scaleKernel,
  sigmoidKernel,
  sumKernel,
  type KernelFunction,
} from '@prml/math';
import { Axes, Curve, Heatmap, Plot, divergingScale, useResolvedTokens } from '@prml/viz';
import { Panel, Select, Slider } from '@prml/ui';
import '../widgets.css';

type KernelType = 'linear' | 'polynomial' | 'rbf' | 'exponential' | 'sigmoid';
type CombineMode = 'a-only' | 'sum' | 'product' | 'difference';

const KERNEL_OPTIONS = [
  { value: 'linear', label: 'Linear' },
  { value: 'polynomial', label: 'Polynomial' },
  { value: 'rbf', label: 'Gaussian (RBF)' },
  { value: 'exponential', label: 'Exponential' },
  { value: 'sigmoid', label: 'Sigmoid (tanh)' },
] as const;

const COMBINE_OPTIONS = [
  { value: 'a-only', label: 'A alone (scaled)' },
  { value: 'sum', label: 'A + B  (6.17)' },
  { value: 'product', label: 'A x B  (6.18)' },
  { value: 'difference', label: 'A - B  (not a rule from the book)' },
] as const;

const PARAM_RANGE: Record<KernelType, { min: number; max: number; step: number; default: number; label: string }> = {
  linear: { min: 0, max: 1, step: 1, default: 0, label: 'no parameter' },
  polynomial: { min: 1, max: 5, step: 1, default: 2, label: 'degree M' },
  rbf: { min: 0.05, max: 1.5, step: 0.01, default: 0.4, label: 'length scale' },
  exponential: { min: 0.2, max: 5, step: 0.1, default: 1.5, label: 'theta' },
  sigmoid: { min: -3, max: 3, step: 0.05, default: 1.5, label: 'a' },
};

function makeKernel(type: KernelType, param: number): KernelFunction {
  switch (type) {
    case 'linear':
      return linearKernel();
    case 'polynomial':
      return polynomialKernel(Math.round(param), 1);
    case 'rbf':
      return rbfKernel(param);
    case 'exponential':
      return exponentialKernel(param);
    case 'sigmoid':
      return sigmoidKernel(param, 0.2);
  }
}

const POINTS = linspace(-1, 1, 8).map((x) => [x]);
const CURVE_GRID = linspace(-1, 1, 161);

export default function KernelConstructionPlayground() {
  const [typeA, setTypeA] = useState<KernelType>('rbf');
  const [paramA, setParamA] = useState(PARAM_RANGE.rbf.default);
  const [typeB, setTypeB] = useState<KernelType>('sigmoid');
  const [paramB, setParamB] = useState(PARAM_RANGE.sigmoid.default);
  const [mode, setMode] = useState<CombineMode>('a-only');
  const tokens = useResolvedTokens();

  const { combined, gram, minEigenvalue } = useMemo(() => {
    const a = makeKernel(typeA, paramA);
    const b = makeKernel(typeB, paramB);
    const k: KernelFunction =
      mode === 'a-only'
        ? scaleKernel(a, 1)
        : mode === 'sum'
          ? sumKernel(a, b)
          : mode === 'product'
            ? productKernel(a, b)
            : (x, xp) => a(x, xp) - b(x, xp);
    const g = gramMatrix(k, POINTS);
    const eigenvalues = eigSym(g).values;
    return { combined: k, gram: g, minEigenvalue: Math.min(...eigenvalues) };
  }, [typeA, paramA, typeB, paramB, mode]);

  const curve = CURVE_GRID.map((x) => [x, combined([x], [0])] as const);
  const peak = Math.max(...curve.map(([, y]) => Math.abs(y)), 1e-6);
  const fill = divergingScale([-peak, peak], 0);
  const isInvalid = minEigenvalue < -1e-8;

  return (
    <div className="widget-grid">
      <Plot height={260} xDomain={[-1, 1]} yDomain={[-peak * 1.1, peak * 1.1]} label="k(x, 0) as a function of x, for the combined kernel">
        <Axes x={{ label: 'x' }} y={{ label: "k(x, 0)" }} grid zeroLine />
        <Curve points={curve} color={isInvalid ? tokens.color.danger : tokens.color.accent} width={2} />
      </Plot>

      <Plot height={260} xDomain={[0, POINTS.length - 1]} yDomain={[0, POINTS.length - 1]} equalAspect label="Gram matrix over 8 fixed points">
        <Heatmap data={{ xs: gram.map((_, i) => i), ys: gram.map((_, i) => i), values: gram }} interpolator={fill} />
      </Plot>

      <Panel columns={2} dense>
        <Select label="Kernel A" value={typeA} onChange={(v) => { setTypeA(v as KernelType); setParamA(PARAM_RANGE[v as KernelType].default); }} options={KERNEL_OPTIONS} />
        <Slider label={`A: ${PARAM_RANGE[typeA].label}`} value={paramA} onChange={setParamA} min={PARAM_RANGE[typeA].min} max={PARAM_RANGE[typeA].max} step={PARAM_RANGE[typeA].step} />
        <Select label="Kernel B" value={typeB} onChange={(v) => { setTypeB(v as KernelType); setParamB(PARAM_RANGE[v as KernelType].default); }} options={KERNEL_OPTIONS} />
        <Slider label={`B: ${PARAM_RANGE[typeB].label}`} value={paramB} onChange={setParamB} min={PARAM_RANGE[typeB].min} max={PARAM_RANGE[typeB].max} step={PARAM_RANGE[typeB].step} />
        <Select label="Combine as" value={mode} onChange={(v) => setMode(v as CombineMode)} options={COMBINE_OPTIONS} />
        <p className="widget-readout">
          {isInvalid
            ? `Smallest eigenvalue of the Gram matrix is ${minEigenvalue.toExponential(2)}, negative: this combination is not a valid kernel.`
            : `Smallest eigenvalue of the Gram matrix is ${minEigenvalue.toExponential(2)}, non-negative: positive semi-definite on these 8 points.`}
        </p>
      </Panel>
    </div>
  );
}
