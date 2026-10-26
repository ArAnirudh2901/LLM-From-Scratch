// Fifty deterministic generated prefixes compared stage-by-stage with model_reader.py.
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
const engine = await import(pathToFileURL(path.resolve(arg('engine'))).href).then((m) => m.default());
const meta = JSON.parse(await readFile(path.join(exported, 'reader.json'), 'utf8'));
const goldenDir = path.resolve(arg('golden'));
const g = JSON.parse(await readFile(path.join(goldenDir, 'golden.json'), 'utf8'));
const wantBytes = await readFile(path.join(goldenDir, 'golden.bin'));
const want = new Float32Array(wantBytes.buffer, wantBytes.byteOffset, wantBytes.byteLength / 4);
const shuffled = gunzipSync(await readFile(path.join(exported, 'reader.bin.gz')));
const n = meta.floats, weights = engine._malloc(n * 4);
for (let b = 0; b < 4; b++) for (let i = 0; i < n; i++) engine.HEAPU8[weights + i * 4 + b] = shuffled[b * n + i];
assert.equal(engine._crc32(weights, n * 4, 0) >>> 0, meta.crc32 >>> 0, 'fp32 weight CRC');
const c = meta.config;
const model = engine._reader_new(weights, c.n_layer, c.n_head, c.n_embd, c.vocab_size, c.block_size,
  meta.hidden_size, meta.d_copy, 128, c.block_size, meta.rms_eps, meta.rope_base);
assert.ok(model, 'reader allocation');
const alloc = (a) => {
  const ptr = engine._malloc(a.byteLength);
  assert.ok(ptr, `allocation for ${a.byteLength} bytes`);
  new a.constructor(engine.HEAPU8.buffer, ptr, a.length).set(a);
  return ptr;
};
const maxDiff = (a, b) => {
  assert.equal(a.length, b.length);
  let d = 0;
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
};
const maxDiffAt = (a, b) => {
  let iMax = 0;
  for (let i = 1; i < a.length; i++) if (Math.abs(a[i] - b[i]) > Math.abs(a[iMax] - b[iMax])) iMax = i;
  return { i: iMax, actual: a[iMax], expected: b[iMax] };
};
const totals = { prefix: 0, decode: 0, logits: 0, copyKeys: 0, copyMixLogp: 0, copyMixProb: 0 };
let offset = 0, cacheCap = 128;
for (let ci = 0; ci < g.cases.length; ci++) {
  const row = g.cases[ci], P = row.P, ids = Int32Array.from(row.ids), C = g.C, DC = g.DC;
  const positions = Int32Array.from({ length: P }, (_, i) => i);
  const slots = Int32Array.from({ length: P }, (_, i) => i);
  const offsets = Int32Array.from({ length: P + 1 }, (_, i) => i * P);
  const contexts = Int32Array.from({ length: P * P }, (_, i) => i % P);
  const idsPtr = alloc(ids), positionsPtr = alloc(positions), slotsPtr = alloc(slots);
  const offsetsPtr = alloc(offsets), contextsPtr = alloc(contexts);
  const hp = engine._malloc(P * C * 4);
  assert.equal(engine._reader_forward(model, P, idsPtr, positionsPtr, slotsPtr, offsetsPtr, contextsPtr, hp), 0);
  const hPrefix = Float32Array.from(new Float32Array(engine.HEAPU8.buffer, hp, P * C));
  if (P + 1 > cacheCap) {
    cacheCap = Math.min(c.block_size, Math.max(P + 1, cacheCap * 2));
    assert.equal(engine._reader_grow(model, cacheCap, 0), 0, `grow cache for prompt ${ci}`);
  }
  const decodeSlot = alloc(Int32Array.of(P)), decodePos = alloc(Int32Array.of(P));
  const decodeTok = alloc(Int32Array.of(row.extra)), decodeOff = alloc(Int32Array.of(0, P + 1));
  const decodeCtx = alloc(Int32Array.from({ length: P + 1 }, (_, i) => i));
  const hd = engine._malloc(C * 4);
  assert.equal(engine._reader_forward(model, 1, decodeTok, decodePos, decodeSlot, decodeOff, decodeCtx, hd), 0);
  const hDecode = Float32Array.from(new Float32Array(engine.HEAPU8.buffer, hd, C));
  const lp = engine._malloc(g.vocab * 4);
  engine._reader_logits(model, hd, 1, lp);
  const logits = Float32Array.from(new Float32Array(engine.HEAPU8.buffer, lp, g.vocab));
  const kp = engine._malloc(P * DC * 4);
  engine._reader_copy_keys(model, hp, P, kp);
  const allKeys = new Float32Array(engine.HEAPU8.buffer, kp, P * DC);
  const keys = Float32Array.from(allKeys.subarray(DC, (P - 1) * DC));
  const copyIds = ids.slice(1, -1), keyPtr = alloc(keys), idPtr = alloc(copyIds), outPtr = engine._malloc(g.vocab * 4);
  engine._reader_mix(model, hd, lp, keyPtr, idPtr, copyIds.length, outPtr);
  const mixed = Float32Array.from(new Float32Array(engine.HEAPU8.buffer, outPtr, g.vocab));
  const expectedPrefix = want.subarray(offset, offset += P * C);
  const expectedDecode = want.subarray(offset, offset += C);
  const expectedLogits = want.subarray(offset, offset += g.vocab);
  const expectedKeys = want.subarray(offset, offset += (P - 2) * DC);
  const expectedMixed = want.subarray(offset, offset += g.vocab);
  const probActual = Float32Array.from(mixed, Math.exp), probExpected = Float32Array.from(expectedMixed, Math.exp);
  const probDiff = maxDiff(probActual, probExpected);
  const diffs = {
    prefix: maxDiff(hPrefix, expectedPrefix), decode: maxDiff(hDecode, expectedDecode),
    logits: maxDiff(logits, expectedLogits), copyKeys: maxDiff(keys, expectedKeys),
    copyMixLogp: maxDiff(mixed, expectedMixed), copyMixProb: probDiff,
  };
  for (const [name, diff] of Object.entries(diffs)) {
    const tolerance = name === 'copyMixLogp' ? 1e-3 : 1e-4;
    const detail = name === 'copyMixProb' ? maxDiffAt(probActual, probExpected) : maxDiffAt(mixed, expectedMixed);
    assert.ok(diff < tolerance, `case ${ci} (${row.kind}) ${name}: max diff ${diff} >= ${tolerance}; ${JSON.stringify(detail)}`);
    totals[name] = Math.max(totals[name], diff);
  }
  for (const ptr of [idsPtr, positionsPtr, slotsPtr, offsetsPtr, contextsPtr, hp,
    decodeSlot, decodePos, decodeTok, decodeOff, decodeCtx, hd, lp, kp, keyPtr, idPtr, outPtr]) engine._free(ptr);
}
assert.equal(offset, want.length, 'all PyTorch reference values consumed');
console.log(`reader WASM parity: ${g.cases.length} generated prompts`, totals);
