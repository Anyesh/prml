import { defineConfig, globalIgnores } from 'eslint/config';
import reactHooks from 'eslint-plugin-react-hooks';

let tseslint;
try {
  tseslint = (await import('typescript-eslint')).default;
} catch (err) {
  console.error(
    `[eslint.config.js] typescript-eslint failed to load (${err.message.split('\n')[0]}). ` +
      'TypeScript-aware linting is disabled until the root package.json pins a typescript-eslint ' +
      'release that supports the installed TypeScript major version - see ' +
      'https://github.com/typescript-eslint/typescript-eslint/issues/10940. ' +
      'JS/MJS files still lint normally.',
  );
  tseslint = null;
}

// SVG and CSS keywords that occupy a colour slot without naming a colour. Banning these
// would push every icon towards a literal, which is the opposite of what the rule is for.
const COLOUR_KEYWORDS_ALLOWED = ['none', 'transparent', 'currentcolor', 'inherit', 'initial', 'unset', 'revert'];

const CSS_NAMED_COLORS_RAW = [
  'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black',
  'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
  'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
  'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta',
  'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
  'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink',
  'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen',
  'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
  'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender',
  'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
  'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon',
  'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue',
  'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
  'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue',
  'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream',
  'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange',
  'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred',
  'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple',
  'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
  'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen',
  'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
  'whitesmoke', 'yellow', 'yellowgreen', 'transparent', 'currentcolor',
];

const CSS_NAMED_COLORS = CSS_NAMED_COLORS_RAW.filter((c) => !COLOUR_KEYWORDS_ALLOWED.includes(c));

const colorLiteralSelectors = [
  {
    selector: `Literal[value=/#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\\b/]`,
    message: 'No hex colour literals outside packages/ui/src/tokens.ts - use a design token.',
  },
  {
    selector: `Literal[value=/\\b(?:rgb|rgba|hsl|hsla)\\(/]`,
    message: 'No rgb()/rgba()/hsl()/hsla() colour literals outside packages/ui/src/tokens.ts - use a design token.',
  },
  {
    selector: `Literal[value=/^(?:${CSS_NAMED_COLORS.join('|')})$/i]`,
    message: 'No CSS named-colour literals outside packages/ui/src/tokens.ts - use a design token.',
  },
];

const svgCanvasSelectors = [
  {
    selector: `JSXOpeningElement[name.name='svg']`,
    message: 'No raw <svg> in widgets - import primitives from @prml/viz instead.',
  },
  {
    selector: `JSXOpeningElement[name.name='canvas']`,
    message: 'No raw <canvas> in widgets - import primitives from @prml/viz instead.',
  },
];

const mathRandomSelector = {
  selector: `MemberExpression[object.name='Math'][property.name='random']`,
  message: 'No Math.random - take a seeded Rng so figures and tests stay reproducible.',
};

const widgetImportSelector = {
  selector:
    `ImportDeclaration[source.value!=/^(@prml\\/(math|viz|ui))(\\/.*)?$/]` +
    `[source.value!=/^react$/][source.value!=/^\\.{1,2}\\//]`,
  message: 'Widget code may only import @prml/math, @prml/viz, @prml/ui, react, or relative paths.',
};

const CONSOLE_METHODS_EXCEPT_LOG = [
  'warn', 'error', 'info', 'debug', 'trace', 'table', 'group', 'groupEnd', 'groupCollapsed',
  'time', 'timeEnd', 'timeLog', 'count', 'countReset', 'assert', 'dir', 'dirxml', 'clear',
];

// @eslint/js isn't in the approved dependency list, so when typescript-eslint
// fails to load (see the TS7 note above) there is no "recommended" preset
// available at all. This hand-picked set of core rules - all built into
// eslint itself, no extra package - is the fallback floor for that case.
const coreFallbackRules = {
  'no-unused-vars': 'error',
  'no-undef': 'error',
  eqeqeq: 'error',
  'no-var': 'error',
  'prefer-const': 'error',
};

// The `globals` package isn't in the approved dependency list either, so
// Node's ambient identifiers are declared by hand rather than pulled in.
const nodeGlobals = {
  process: 'readonly',
  console: 'readonly',
  URL: 'readonly',
  URLSearchParams: 'readonly',
  Buffer: 'readonly',
  TextEncoder: 'readonly',
  TextDecoder: 'readonly',
  fetch: 'readonly',
  structuredClone: 'readonly',
  setTimeout: 'readonly',
  clearTimeout: 'readonly',
  setInterval: 'readonly',
  clearInterval: 'readonly',
  queueMicrotask: 'readonly',
};

export default defineConfig([
  globalIgnores(['**/dist/**', '**/node_modules/**', '**/.astro/**', '**/__fixtures__/**']),

  { languageOptions: { globals: nodeGlobals } },

  ...(tseslint
    ? [...tseslint.configs.recommended]
    : [{ files: ['**/*.{js,mjs,cjs,jsx}'], languageOptions: { sourceType: 'module' }, rules: coreFallbackRules }]),

  {
    files: ['**/*.{js,mjs,cjs,jsx,ts,tsx}'],
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrorsIgnorePattern: '^_' },
      ],
    },
  },
  {
    files: ['**/*.{js,mjs,cjs,jsx,ts,tsx}'],
    // This file is exempt for the same reason tokens.ts is: it enumerates every CSS colour
    // name in order to ban them, so the rule would otherwise flag its own definition.
    ignores: ['packages/ui/src/tokens.ts', 'eslint.config.js', '**/*.test.ts', '**/*.test.tsx'],
    rules: {
      'no-restricted-syntax': ['error', ...colorLiteralSelectors],
    },
  },
  {
    files: ['**/*.{js,mjs,cjs,jsx,ts,tsx}'],
    ignores: ['scripts/**', 'tools/**'],
    rules: {
      'no-console': ['error', { allow: CONSOLE_METHODS_EXCEPT_LOG }],
    },
  },
  {
    files: ['packages/math/**/*.{js,ts,tsx}'],
    rules: {
      'no-restricted-syntax': ['error', ...colorLiteralSelectors, mathRandomSelector],
    },
  },
  {
    files: ['**/*.{jsx,tsx}'],
    plugins: { 'react-hooks': reactHooks },
    rules: {
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
    },
  },
  {
    files: ['apps/web/src/widgets/**/*.{js,jsx,ts,tsx}'],
    rules: {
      'no-restricted-syntax': [
        'error',
        ...colorLiteralSelectors,
        ...svgCanvasSelectors,
        mathRandomSelector,
        widgetImportSelector,
      ],
    },
  },
]);
