import { DIVERGING, SEQUENTIAL } from '@prml/ui';

export type Interpolator = (t: number) => string;

interface Rgba {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;
}

function srgbChannelToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function linearChannelToSrgb(c: number): number {
  const clamped = Math.min(1, Math.max(0, c));
  return clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055;
}

/** sRGB channels in `[0, 1]` to Oklab `[L, a, b]`. */
export function srgbToOklab(r: number, g: number, b: number): readonly [number, number, number] {
  const lr = srgbChannelToLinear(r);
  const lg = srgbChannelToLinear(g);
  const lb = srgbChannelToLinear(b);

  const l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
  const m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
  const s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;

  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);

  return [
    0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  ];
}

/** Oklab `[L, a, b]` to sRGB channels in `[0, 1]`, clamped to gamut. */
export function oklabToSrgb(L: number, a: number, b: number): readonly [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;

  const l = l_ ** 3;
  const m = m_ ** 3;
  const s = s_ ** 3;

  const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const b2 = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s;

  return [linearChannelToSrgb(r), linearChannelToSrgb(g), linearChannelToSrgb(b2)];
}

const HEX = /^#([0-9a-f]{3,8})$/i;
const COLOR_FN = /^(rgb|rgba|oklch|oklab)\(([^)]*)\)$/i;

function fail(color: string): never {
  throw new Error(`@prml/viz: cannot parse colour "${color}"`);
}

function splitSlashAlpha(body: string): readonly [string, string | undefined] {
  const slash = body.indexOf('/');
  if (slash === -1) return [body, undefined];
  return [body.slice(0, slash), body.slice(slash + 1).trim()];
}

function unitValue(token: string, whole: string): number {
  const t = token.trim();
  const n = t.endsWith('%') ? parseFloat(t) / 100 : parseFloat(t);
  if (Number.isNaN(n)) fail(whole);
  return n;
}

function rgbChannel(token: string, whole: string): number {
  const t = token.trim();
  const n = t.endsWith('%') ? parseFloat(t) / 100 : parseFloat(t) / 255;
  if (Number.isNaN(n)) fail(whole);
  return n;
}

function angleDegrees(token: string, whole: string): number {
  const t = token.trim();
  const n = parseFloat(t);
  if (Number.isNaN(n)) fail(whole);
  if (t.endsWith('rad')) return (n * 180) / Math.PI;
  if (t.endsWith('turn')) return n * 360;
  return n;
}

function parseHexDigits(digits: string, whole: string): Rgba {
  const nibble = (s: string) => parseInt(s.length === 1 ? s + s : s, 16) / 255;
  if (digits.length === 3 || digits.length === 4) {
    const [r, g, b, a] = digits;
    if (r === undefined || g === undefined || b === undefined) fail(whole);
    return { r: nibble(r), g: nibble(g), b: nibble(b), a: a === undefined ? 1 : nibble(a) };
  }
  if (digits.length !== 6 && digits.length !== 8) fail(whole);
  const alpha = digits.slice(6, 8);
  return {
    r: nibble(digits.slice(0, 2)),
    g: nibble(digits.slice(2, 4)),
    b: nibble(digits.slice(4, 6)),
    a: alpha === '' ? 1 : nibble(alpha),
  };
}

/**
 * Decodes any colour string the design tokens or the scales here can produce into
 * normalised sRGB channels, without touching the DOM so that callers stay unit-testable
 * under a plain Node runtime.
 *
 * Gamut mapping on the Oklab forms is a per-channel clamp rather than the full CSS Color 4
 * algorithm, because that needs iterative chroma reduction and this repo's palette stays
 * inside sRGB in practice.
 */
export function parseCssColor(color: string): Rgba {
  const value = color.trim();

  const hex = HEX.exec(value);
  if (hex?.[1]) return parseHexDigits(hex[1], color);

  const fn = COLOR_FN.exec(value);
  if (!fn?.[1] || fn[2] === undefined) fail(color);

  const kind = fn[1].toLowerCase();
  const [main, slashAlpha] = splitSlashAlpha(fn[2]);
  const parts = main.trim().split(/[\s,]+/).filter(Boolean);
  const [p0, p1, p2, p3] = parts;
  if (p0 === undefined || p1 === undefined || p2 === undefined) fail(color);

  if (kind === 'rgb' || kind === 'rgba') {
    // Legacy syntax carries alpha as a fourth comma-separated component rather than
    // after a slash, and both forms reach here from real token values.
    const alphaToken = slashAlpha ?? p3;
    return {
      r: rgbChannel(p0, color),
      g: rgbChannel(p1, color),
      b: rgbChannel(p2, color),
      a: alphaToken === undefined ? 1 : unitValue(alphaToken, color),
    };
  }

  const alpha = slashAlpha === undefined ? 1 : unitValue(slashAlpha, color);
  const lightness = unitValue(p0, color);
  const second = parseFloat(p1);
  if (Number.isNaN(second)) fail(color);

  const [a, b] =
    kind === 'oklch'
      ? (() => {
          const h = (angleDegrees(p2, color) * Math.PI) / 180;
          return [second * Math.cos(h), second * Math.sin(h)] as const;
        })()
      : (() => {
          const third = parseFloat(p2);
          if (Number.isNaN(third)) fail(color);
          return [second, third] as const;
        })();

  const [r, g, bb] = oklabToSrgb(lightness, a, b);
  return { r, g, b: bb, a: alpha };
}

const toByte = (c: number) => Math.round(Math.min(1, Math.max(0, c)) * 255);

/**
 * The same decode as `parseCssColor` but as 0-255 bytes, which is what `ImageData` wants.
 * Callers building a pixel buffer must go through this rather than rolling their own
 * parser, because a second parser is a second place to teach every new colour format.
 */
export function cssColorToRgbaBytes(color: string): readonly [number, number, number, number] {
  const { r, g, b, a } = parseCssColor(color);
  return [toByte(r), toByte(g), toByte(b), toByte(a)];
}

function formatRgba(color: Rgba, alpha: number): string {
  const to255 = (c: number) => Math.round(Math.min(1, Math.max(0, c)) * 255);
  return `rgba(${to255(color.r)}, ${to255(color.g)}, ${to255(color.b)}, ${alpha})`;
}

/**
 * Piecewise-linear interpolation through `stops` in Oklab rather than sRGB, because
 * interpolating viridis in sRGB reintroduces the luminance banding it was designed to
 * remove, and banding in a density field reads as structure that is not there.
 */
export function interpolateStops(stops: readonly string[]): Interpolator {
  if (stops.length === 0) throw new Error('@prml/viz: interpolateStops requires at least one stop');

  const parsed = stops.map(parseCssColor);
  const oklab = parsed.map(({ r, g, b }) => srgbToOklab(r, g, b));

  if (parsed.length === 1) {
    // invariant: length check above guarantees index 0 exists.
    const only = parsed[0]!;
    return () => formatRgba(only, only.a);
  }

  return (t: number) => {
    const clamped = Math.min(1, Math.max(0, t));
    const scaled = clamped * (parsed.length - 1);
    const i0 = Math.min(parsed.length - 2, Math.floor(scaled));
    const i1 = i0 + 1;
    const localT = scaled - i0;

    // invariant: i0 is clamped into [0, parsed.length - 2] above, so both indices exist.
    const [L0, a0, b0] = oklab[i0]!;
    const [L1, a1, b1] = oklab[i1]!;
    const c0 = parsed[i0]!;
    const c1 = parsed[i1]!;

    const L = L0 + (L1 - L0) * localT;
    const a = a0 + (a1 - a0) * localT;
    const b = b0 + (b1 - b0) * localT;
    const alpha = c0.a + (c1.a - c0.a) * localT;

    const [r, g, b2] = oklabToSrgb(L, a, b);
    return formatRgba({ r, g, b: b2, a: alpha }, alpha);
  };
}

function normalize(x: number, domain: readonly [number, number]): number {
  const [d0, d1] = domain;
  if (d0 === d1) return 0.5;
  return (x - d0) / (d1 - d0);
}

export function sequentialScale(domain: readonly [number, number], stops: readonly string[] = SEQUENTIAL): Interpolator {
  const interpolator = interpolateStops(stops);
  return (x: number) => interpolator(normalize(x, domain));
}

/**
 * Symmetric about `center`, so that equal positive and negative values get equally
 * saturated colours no matter how lopsided the data range is.
 */
export function divergingScale(
  domain: readonly [number, number],
  center = 0,
  stops: readonly string[] = DIVERGING,
): Interpolator {
  const interpolator = interpolateStops(stops);
  const halfRange = Math.max(domain[1] - center, center - domain[0], Number.EPSILON);
  return (x: number) => {
    const signed = Math.min(halfRange, Math.max(-halfRange, x - center)) / halfRange;
    return interpolator(0.5 + signed * 0.5);
  };
}

/** `n` evenly spaced samples, for discrete legends and contour band fills. */
export function quantize(interpolator: Interpolator, n: number): string[] {
  if (n <= 0) return [];
  if (n === 1) return [interpolator(0.5)];
  return Array.from({ length: n }, (_, i) => interpolator(i / (n - 1)));
}

/**
 * Must accept every colour form the tokens can resolve to, including hex and `oklch()`,
 * because a caller cannot know which one a given theme produced.
 */
export function withAlpha(color: string, alpha: number): string {
  const parsed = parseCssColor(color);
  return formatRgba(parsed, alpha);
}
