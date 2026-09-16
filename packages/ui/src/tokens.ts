/**
 * The single source of truth for colour, spacing, type, and motion.
 *
 * `tokens.css` is generated from this file by `pnpm tokens:build` and the freshness gate
 * fails the build if the two drift. Nothing else in the repo may contain a colour
 * literal: components read CSS custom properties, and canvas code reads the resolved
 * values through `useResolvedTokens` in `@prml/viz`, which is the only way a canvas can
 * follow a theme switch.
 */

export interface ColorScheme {
  /** Page ground. */
  readonly bg: string;
  /** Raised surfaces: widget frames, cards, the command palette. */
  readonly surface: string;
  /** Recessed surfaces: code blocks, well backgrounds. */
  readonly sunken: string;
  readonly border: string;
  readonly borderStrong: string;
  /** Body text. */
  readonly ink: string;
  /** Secondary text: captions, axis labels, metadata. */
  readonly inkMuted: string;
  /** Tertiary text: tick labels, disabled controls. */
  readonly inkFaint: string;
  /** Interactive accent: links, focus rings, active controls. */
  readonly accent: string;
  readonly accentHover: string;
  /** Accent at low opacity, for selected rows and slider tracks. */
  readonly accentWash: string;
  readonly danger: string;
  readonly success: string;
  /** Plot ground, distinct from `surface` so a figure reads as a figure. */
  readonly plotBg: string;
  readonly gridLine: string;
  readonly axisLine: string;
}

/**
 * Okabe-Ito, the standard eight-colour set that stays distinguishable under all three
 * common dichromacies. Series colours must be taken from here in order rather than
 * chosen per widget, so that "the blue one" means the same thing across fourteen chapters.
 */
export const SERIES = [
  '#0072B2',
  '#D55E00',
  '#009E73',
  '#CC79A7',
  '#E69F00',
  '#56B4E9',
  '#8C6D1F',
  '#525252',
] as const;

/** Semantic roles layered on top of `SERIES`, so a widget names the idea rather than the index. */
export const ROLE = {
  data: SERIES[0],
  model: SERIES[1],
  truth: SERIES[2],
  posterior: SERIES[3],
  prior: SERIES[4],
  predictive: SERIES[5],
  classA: SERIES[0],
  classB: SERIES[1],
  classC: SERIES[2],
} as const;

/** Viridis, sampled at 9 stops. Perceptually uniform, so contour spacing reads as density. */
export const SEQUENTIAL = [
  '#440154',
  '#472D7B',
  '#3B518B',
  '#2C718E',
  '#21908C',
  '#27AD81',
  '#5DC863',
  '#AADC32',
  '#FDE725',
] as const;

/** Blue-white-red, for signed fields (weight values, residuals, message differences). */
export const DIVERGING = [
  '#053061',
  '#2166AC',
  '#4393C3',
  '#92C5DE',
  '#F7F7F7',
  '#F4A582',
  '#D6604D',
  '#B2182B',
  '#67001F',
] as const;

export const LIGHT: ColorScheme = {
  bg: '#FDFCFA',
  surface: '#FFFFFF',
  sunken: '#F4F2EE',
  border: '#E4E0D8',
  borderStrong: '#CBC5B8',
  ink: '#1A1B1F',
  inkMuted: '#5A5C64',
  inkFaint: '#8E9099',
  accent: '#2B5FD9',
  accentHover: '#1E47AB',
  accentWash: '#E8EEFC',
  danger: '#C0392B',
  success: '#1E7A4C',
  plotBg: '#FFFFFF',
  gridLine: '#EDEAE3',
  axisLine: '#9A958A',
};

export const DARK: ColorScheme = {
  bg: '#131417',
  surface: '#1B1D22',
  sunken: '#101114',
  border: '#2B2E35',
  borderStrong: '#41454E',
  ink: '#E9E7E2',
  inkMuted: '#A3A6AF',
  inkFaint: '#70747E',
  accent: '#7EA6FF',
  accentHover: '#A3C0FF',
  accentWash: '#1C2540',
  danger: '#F08276',
  success: '#5FD39B',
  plotBg: '#181A1F',
  gridLine: '#25282F',
  axisLine: '#5E626B',
};

/** 4px base. Index is not pixels: `space[3]` is 12px. */
export const SPACE = [0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128] as const;

export const TYPE = {
  /** Prose. A transitional serif keeps 60 lines of exposition readable in a way a UI sans does not. */
  serif: "'Source Serif 4', 'Source Serif Pro', Charter, Georgia, 'Times New Roman', serif",
  /** Chrome: nav, controls, labels, captions. */
  sans: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, system-ui, sans-serif",
  mono: "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace",
  scale: {
    xs: '0.75rem',
    sm: '0.8125rem',
    base: '0.9375rem',
    md: '1.0625rem',
    lg: '1.25rem',
    xl: '1.5rem',
    xxl: '2rem',
    display: '2.75rem',
  },
  leading: {
    tight: '1.2',
    snug: '1.4',
    normal: '1.65',
    prose: '1.75',
  },
  weight: {
    regular: '400',
    medium: '500',
    semibold: '600',
  },
} as const;

export const RADIUS = {
  sm: '3px',
  md: '6px',
  lg: '10px',
  pill: '999px',
} as const;

export const SHADOW = {
  /** Deliberately soft. A teaching figure should sit on the page, not hover above it. */
  sm: '0 1px 2px rgb(0 0 0 / 0.06)',
  md: '0 2px 8px rgb(0 0 0 / 0.08)',
  lg: '0 8px 28px rgb(0 0 0 / 0.14)',
} as const;

export const MOTION = {
  instant: '80ms',
  fast: '140ms',
  base: '220ms',
  slow: '400ms',
  /** Springy enough to feel responsive under a slider drag without overshoot on a plot. */
  ease: 'cubic-bezier(0.32, 0.72, 0, 1)',
  easeOut: 'cubic-bezier(0.16, 1, 0.3, 1)',
} as const;

export const LAYOUT = {
  /** Prose column. Wider than this and the eye loses the line return at this type size. */
  proseWidth: '68ch',
  railWidth: '15rem',
  asideWidth: '13rem',
  /** Widgets break out of the prose column to this width. */
  widgetWidth: '52rem',
} as const;

export const Z = {
  base: 0,
  sticky: 10,
  overlay: 100,
  palette: 200,
} as const;

/** CSS custom property name for a colour role, the bridge between this file and `tokens.css`. */
export function colorVar(role: keyof ColorScheme): string {
  return `--prml-color-${kebab(role)}`;
}

export function kebab(s: string): string {
  return s.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);
}
