import {
  useCallback,
  useId,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import './Slider.css';

export type SliderScale = 'linear' | 'log';

export interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  format?: (value: number) => string;
  scale?: SliderScale;
  hint?: string;
  disabled?: boolean;
}

export const DEFAULT_STEP_DIVISIONS = 100;
export const COARSE_STEP_MULTIPLIER = 10;

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function defaultFormat(value: number): string {
  if (Number.isInteger(value)) return String(value);
  return String(Math.round(value * 1000) / 1000);
}

export function quantize(value: number, min: number, step: number): number {
  if (!(step > 0)) return value;
  return min + Math.round((value - min) / step) * step;
}

function assertPositiveMinForLog(min: number): void {
  if (min <= 0) {
    throw new Error('Slider: scale="log" requires min > 0 because log(0) is undefined.');
  }
}

export function valueToFraction(value: number, min: number, max: number, scale: SliderScale): number {
  if (max <= min) return 0;
  const v = clamp(value, min, max);
  if (scale === 'log') {
    assertPositiveMinForLog(min);
    return (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min));
  }
  return (v - min) / (max - min);
}

export function fractionToValue(fraction: number, min: number, max: number, scale: SliderScale): number {
  const f = clamp(fraction, 0, 1);
  if (scale === 'log') {
    assertPositiveMinForLog(min);
    return Math.exp(Math.log(min) + f * (Math.log(max) - Math.log(min)));
  }
  return min + f * (max - min);
}

/**
 * The keyboard/step contract: an explicit `step` always moves in additive real units and snaps
 * to that grid (matches the literal meaning of the prop). Without one, the default divides the
 * *travel* into DEFAULT_STEP_DIVISIONS parts, so a log slider nudges by a constant ratio instead
 * of a constant amount, avoiding a step that is invisible at one end of a decade-spanning range
 * and huge at the other.
 */
export function nextValue(
  current: number,
  direction: 1 | -1,
  min: number,
  max: number,
  scale: SliderScale,
  step: number | undefined,
  coarse: boolean,
): number {
  const multiplier = coarse ? COARSE_STEP_MULTIPLIER : 1;
  if (step !== undefined && step > 0) {
    const raw = current + direction * step * multiplier;
    return clamp(quantize(raw, min, step), min, max);
  }
  if (scale === 'log') {
    assertPositiveMinForLog(min);
    const ratioPerDivision = Math.exp(Math.log(max / min) / DEFAULT_STEP_DIVISIONS);
    const ratio = ratioPerDivision ** multiplier;
    return clamp(direction === 1 ? current * ratio : current / ratio, min, max);
  }
  const linearStep = ((max - min) / DEFAULT_STEP_DIVISIONS) * multiplier;
  return clamp(current + direction * linearStep, min, max);
}

export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  format,
  scale = 'linear',
  hint,
  disabled = false,
}: SliderProps) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const labelId = useId();
  const sliderId = useId();
  const hintId = useId();
  const [dragging, setDragging] = useState(false);

  const formatValue = format ?? defaultFormat;
  const fraction = valueToFraction(value, min, max, scale);
  const display = formatValue(value);

  const commit = useCallback(
    (raw: number) => {
      const quantized = step !== undefined && step > 0 ? quantize(raw, min, step) : raw;
      const next = clamp(quantized, min, max);
      if (next !== value) onChange(next);
    },
    [min, max, step, value, onChange],
  );

  const fractionFromClientX = useCallback(
    (clientX: number): number => {
      const el = trackRef.current;
      if (!el) return fraction;
      const rect = el.getBoundingClientRect();
      if (rect.width === 0) return fraction;
      return clamp((clientX - rect.left) / rect.width, 0, 1);
    },
    [fraction],
  );

  const handlePointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
    commit(fractionToValue(fractionFromClientX(e.clientX), min, max, scale));
  };

  const handlePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (disabled || !dragging) return;
    commit(fractionToValue(fractionFromClientX(e.clientX), min, max, scale));
  };

  const endDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    setDragging(false);
  };

  const handleKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    const coarse = e.shiftKey;
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        e.preventDefault();
        commit(nextValue(value, 1, min, max, scale, step, coarse));
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        e.preventDefault();
        commit(nextValue(value, -1, min, max, scale, step, coarse));
        break;
      case 'Home':
        e.preventDefault();
        commit(min);
        break;
      case 'End':
        e.preventDefault();
        commit(max);
        break;
      default:
        break;
    }
  };

  return (
    <div className={`prml-slider${disabled ? ' prml-slider--disabled' : ''}`}>
      <div className="prml-slider-header">
        <span id={labelId} className="prml-slider-label">
          {label}
        </span>
        <output className="prml-slider-value" htmlFor={sliderId}>
          {display}
        </output>
      </div>
      <div
        ref={trackRef}
        className={`prml-slider-track${dragging ? ' prml-slider-track--dragging' : ''}`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <div className="prml-slider-fill" style={{ width: `${fraction * 100}%` }} />
        <div
          id={sliderId}
          role="slider"
          tabIndex={disabled ? -1 : 0}
          aria-orientation="horizontal"
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={value}
          aria-valuetext={display}
          aria-labelledby={labelId}
          aria-describedby={hint ? hintId : undefined}
          aria-disabled={disabled ? true : undefined}
          className="prml-slider-thumb"
          style={{ left: `${fraction * 100}%` }}
          onKeyDown={handleKeyDown}
        />
      </div>
      {hint ? (
        <p id={hintId} className="prml-slider-hint">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
