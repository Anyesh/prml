import fs from 'node:fs';
import path from 'node:path';

/** Recursively lists files under `dir` matching `extension`. Returns [] if `dir` does not exist. */
export function findFiles(dir, extension) {
  if (!fs.existsSync(dir)) return [];
  const results = [];
  const stack = [dir];
  while (stack.length > 0) {
    const current = stack.pop();
    const entries = fs.readdirSync(current, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
      } else if (entry.isFile() && full.endsWith(extension)) {
        results.push(full);
      }
    }
  }
  results.sort();
  return results;
}
