#!/usr/bin/env node
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { DIST_DIR } from './lib/paths.mjs';

const PORT = Number(process.env.PORT ?? 4173);
const ROOT = process.argv[2] ?? DIST_DIR;

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.woff2': 'font/woff2',
  '.wasm': 'application/wasm',
};

if (!fs.existsSync(ROOT)) {
  console.error(`${ROOT} does not exist. Run \`pnpm build\` first.`);
  process.exit(1);
}

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent((req.url ?? '/').split('?')[0]);
  const candidates = urlPath.endsWith('/')
    ? [path.join(ROOT, urlPath, 'index.html')]
    : [path.join(ROOT, urlPath), path.join(ROOT, urlPath, 'index.html')];

  const resolved = candidates.find((p) => fs.existsSync(p) && fs.statSync(p).isFile());
  if (!resolved) {
    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('not found');
    return;
  }

  const type = CONTENT_TYPES[path.extname(resolved)] ?? 'application/octet-stream';
  res.writeHead(200, { 'content-type': type });
  fs.createReadStream(resolved).pipe(res);
});

server.listen(PORT, () => {
  console.log(`serving ${ROOT} at http://localhost:${PORT}`);
});
