import { useId } from 'react';
import './Toggle.css';

export interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  hint?: string;
  disabled?: boolean;
}

export function Toggle({ label, checked, onChange, hint, disabled = false }: ToggleProps) {
  const hintId = useId();

  return (
    <label className={`prml-toggle${disabled ? ' prml-toggle--disabled' : ''}`}>
      <input
        type="checkbox"
        role="switch"
        className="prml-toggle-input"
        checked={checked}
        disabled={disabled}
        aria-describedby={hint ? hintId : undefined}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="prml-toggle-track" aria-hidden="true">
        <span className="prml-toggle-thumb" />
      </span>
      <span className="prml-toggle-body">
        <span className="prml-toggle-label">{label}</span>
        {hint ? (
          <span id={hintId} className="prml-toggle-hint">
            {hint}
          </span>
        ) : null}
      </span>
    </label>
  );
}
