import { useId, type ChangeEvent } from 'react';
import './Select.css';

export interface SelectOption {
  value: string;
  label: string;
  hint?: string;
}

export interface SelectProps {
  label: string;
  value: string;
  options: readonly SelectOption[];
  onChange: (value: string) => void;
  hint?: string;
  disabled?: boolean;
}

export function Select({ label, value, options, onChange, hint, disabled = false }: SelectProps) {
  const selectId = useId();
  const hintId = useId();

  const handleChange = (e: ChangeEvent<HTMLSelectElement>) => {
    onChange(e.target.value);
  };

  return (
    <div className={`prml-select${disabled ? ' prml-select--disabled' : ''}`}>
      <label htmlFor={selectId} className="prml-select-label">
        {label}
      </label>
      <div className="prml-select-control">
        <select
          id={selectId}
          className="prml-select-input"
          value={value}
          disabled={disabled}
          aria-describedby={hint ? hintId : undefined}
          onChange={handleChange}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value} title={option.hint}>
              {option.label}
            </option>
          ))}
        </select>
        <svg className="prml-select-caret" aria-hidden="true" viewBox="0 0 16 16">
          <path d="M4 6.2 8 10l4-3.8" stroke="currentColor" strokeWidth="1.3" fill="none" />
        </svg>
      </div>
      {hint ? (
        <p id={hintId} className="prml-select-hint">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
