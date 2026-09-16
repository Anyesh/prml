import { useCallback, useEffect, useId, useRef, useState } from 'react';
import './StepThrough.css';

export interface StepThroughProps {
  step: number;
  stepCount: number;
  onStep: (step: number) => void;
  interval?: number;
  labels?: readonly string[];
  onReset?: () => void;
}

export function clampStep(step: number, stepCount: number): number {
  return Math.min(Math.max(step, 0), Math.max(stepCount - 1, 0));
}

export function isAtStart(step: number): boolean {
  return step <= 0;
}

export function isAtEnd(step: number, stepCount: number): boolean {
  return step >= stepCount - 1;
}

function IconStepBack() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="prml-stepthrough-icon">
      <path fill="currentColor" d="M11 3.5 6 8l5 4.5v-9Z" />
      <rect x="3.5" y="3" width="1.5" height="10" fill="currentColor" />
    </svg>
  );
}

function IconStepForward() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="prml-stepthrough-icon">
      <path fill="currentColor" d="M5 3.5 10 8l-5 4.5v-9Z" />
      <rect x="11" y="3" width="1.5" height="10" fill="currentColor" />
    </svg>
  );
}

function IconPlay() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="prml-stepthrough-icon">
      <path fill="currentColor" d="M4.5 2.8v10.4a.8.8 0 0 0 1.22.68l8.2-5.2a.8.8 0 0 0 0-1.36l-8.2-5.2A.8.8 0 0 0 4.5 2.8Z" />
    </svg>
  );
}

function IconPause() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="prml-stepthrough-icon">
      <rect x="4" y="2.5" width="3" height="11" rx="0.75" fill="currentColor" />
      <rect x="9" y="2.5" width="3" height="11" rx="0.75" fill="currentColor" />
    </svg>
  );
}

function IconReset() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="prml-stepthrough-icon">
      <path
        fill="currentColor"
        d="M8 2.5a5.5 5.5 0 1 1-5.2 3.7.75.75 0 0 1 1.42.48A4 4 0 1 0 8 4v1.8a.5.5 0 0 1-.82.38L4.5 3.9a.5.5 0 0 1 0-.76L7.18.9A.5.5 0 0 1 8 1.3v1.2Z"
      />
    </svg>
  );
}

export function StepThrough({ step, stepCount, onStep, interval, labels, onReset }: StepThroughProps) {
  const [playing, setPlaying] = useState(false);
  const statusId = useId();

  const stepRef = useRef(step);
  const stepCountRef = useRef(stepCount);
  const onStepRef = useRef(onStep);
  useEffect(() => {
    stepRef.current = step;
  }, [step]);
  useEffect(() => {
    stepCountRef.current = stepCount;
  }, [stepCount]);
  useEffect(() => {
    onStepRef.current = onStep;
  }, [onStep]);

  useEffect(() => {
    if (!playing || interval === undefined) return undefined;
    const id = window.setInterval(() => {
      const next = stepRef.current + 1;
      if (next >= stepCountRef.current) {
        setPlaying(false);
        return;
      }
      onStepRef.current(next);
    }, interval);
    return () => window.clearInterval(id);
  }, [playing, interval]);

  useEffect(() => {
    function handleVisibilityChange() {
      if (document.hidden) setPlaying(false);
    }
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  const handleBack = useCallback(() => {
    onStep(clampStep(step - 1, stepCount));
  }, [onStep, step, stepCount]);

  const handleForward = useCallback(() => {
    onStep(clampStep(step + 1, stepCount));
  }, [onStep, step, stepCount]);

  const handleReset = useCallback(() => {
    setPlaying(false);
    if (onReset) onReset();
    else onStep(0);
  }, [onReset, onStep]);

  const togglePlay = useCallback(() => {
    if (!playing && isAtEnd(step, stepCount)) onStep(0);
    setPlaying((p) => !p);
  }, [playing, step, stepCount, onStep]);

  const currentLabel = labels?.[step];

  return (
    <div className="prml-stepthrough">
      <div className="prml-stepthrough-controls">
        <button
          type="button"
          className="prml-stepthrough-btn"
          onClick={handleBack}
          disabled={isAtStart(step)}
          aria-label="Previous step"
        >
          <IconStepBack />
        </button>
        {interval !== undefined ? (
          <button
            type="button"
            className="prml-stepthrough-btn"
            onClick={togglePlay}
            aria-label={playing ? 'Pause' : 'Play'}
            aria-pressed={playing}
          >
            {playing ? <IconPause /> : <IconPlay />}
          </button>
        ) : null}
        <button
          type="button"
          className="prml-stepthrough-btn"
          onClick={handleForward}
          disabled={isAtEnd(step, stepCount)}
          aria-label="Next step"
        >
          <IconStepForward />
        </button>
        <button
          type="button"
          className="prml-stepthrough-btn prml-stepthrough-btn--reset"
          onClick={handleReset}
          aria-label="Reset"
        >
          <IconReset />
        </button>
      </div>
      <p id={statusId} className="prml-stepthrough-status" aria-live="polite">
        Step {step + 1} of {stepCount}
        {currentLabel ? `: ${currentLabel}` : ''}
      </p>
    </div>
  );
}
