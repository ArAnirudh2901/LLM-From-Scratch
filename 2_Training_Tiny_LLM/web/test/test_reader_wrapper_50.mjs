// Exercise the JS wrapper's prefill, cache growth, and one-token decode on 50 PyTorch fixtures.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const arg = (name) => {
  const value = process.argv.find((x) => x.startsWith(`--${name}=`));
  if (!value) throw new Error(`pass --${name}=...`);
  return value.slice(name.length + 3);
};
const exported = path.resolve(arg('export'));
const goldenDir = path.resolve(arg('golden'));
const meta = JSON.parse(await readFile(path.join(exported, 'reader.json'), 'utf8'));
const g = JSON.parse(await readFile(path.join(goldenDir, 'golden.json'), 'utf8'));
const bytes = await readFile(path.join(goldenDir, 'golden.bin'));
const want = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
const shuffled = gunzipSync(await readFile(path.join(exported, 'reader.bin.gz')));
const modulePath = pathToFileURL(path.resolve('2_Training_Tiny_LLM/web/reader.js')).href;
const { ReaderWasm } = await import(modulePath);
const maxDiff = (a, b) => {
  assert.equal(a.length, b.length);
  let d = 0;
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
};

for (const engine of ['simd', 'scalar']) {
  const reader = await ReaderWasm.load(meta, shuffled, engine);
  let maxLogp = 0, maxProb = 0, offset = 0;
  for (const row of g.cases) {
    const P = row.P;
    offset += P * g.C + g.C + g.vocab + (P - 2) * g.DC;
    const expectedLogp = want.subarray(offset, offset += g.vocab);
    const state = reader.prefill(row.ids, 16);
    assert.equal(state.prefixLength, P);
    assert.equal(state.promptIds.length, row.copy);
    const actualLogp = reader.decode(state, row.extra, []);
    const d = maxDiff(actualLogp, expectedLogp);
    assert.ok(d < 1e-3, `${engine} ${row.kind} log-prob max diff ${d}`);
    maxLogp = Math.max(maxLogp, d);
    maxProb = Math.max(maxProb, maxDiff(Float32Array.from(actualLogp, Math.exp), Float32Array.from(expectedLogp, Math.exp)));
  }
  console.log(`ReaderWasm wrapper ${engine}: ${g.cases.length} prompts; max logp ${maxLogp}; max probability ${maxProb}`);
}
