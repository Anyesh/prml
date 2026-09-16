import { parseArgs } from 'node:util';

export function parseGateArgs(argv, extraOptions = {}) {
  const { values } = parseArgs({
    args: argv,
    options: {
      json: { type: 'boolean', default: false },
      ...extraOptions,
    },
    allowPositionals: false,
    strict: false,
  });
  return values;
}

/**
 * Prints a gate result and returns the process exit code.
 * `result` shape: { gate, ok, checked, failures, warnings, info, vacuous }
 */
export function reportGate(result, { json = false } = {}) {
  if (json) {
    console.log(JSON.stringify(result, null, 2));
    return result.ok ? 0 : 1;
  }

  const { gate, checked, failures = [], warnings = [], info = [], vacuous = false } = result;
  console.log(`\n=== gate: ${gate} ===`);

  if (vacuous) {
    console.log(`0 files checked - nothing to validate yet (this still passes).`);
  } else {
    console.log(`checked ${checked} file${checked === 1 ? '' : 's'}`);
  }

  for (const line of info) {
    console.log(`  info: ${line}`);
  }

  for (const warning of warnings) {
    console.log(`  warn: ${formatLocated(warning)}`);
  }

  if (failures.length === 0) {
    console.log(`PASS: ${gate}`);
    return 0;
  }

  for (const failure of failures) {
    console.log(`  FAIL: ${formatLocated(failure)}`);
  }
  console.log(`FAIL: ${gate} (${failures.length} problem${failures.length === 1 ? '' : 's'})`);
  return 1;
}

function formatLocated(entry) {
  if (typeof entry === 'string') return entry;
  const { file, line, message } = entry;
  const loc = file ? (line ? `${file}:${line}` : file) : null;
  return loc ? `${loc} - ${message}` : message;
}
