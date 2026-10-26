// Compare the v10 WASM forward/cache/copy path with model_reader.py; all diffs must stay below 1e-4.
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
const engine = await import(pathToFileURL(path.resolve(arg('engine'))).href).then((m) => m.default());
const meta = JSON.parse(await readFile(path.join(exported, 'reader.json'), 'utf8'));
const g = JSON.parse(await readFile(path.join(goldenDir, 'golden.json'), 'utf8'));
const wantBytes = await readFile(path.join(goldenDir, 'golden.bin'));
const want = new Float32Array(wantBytes.buffer, wantBytes.byteOffset, wantBytes.byteLength / 4);
const shuffled = gunzipSync(await readFile(path.join(exported, 'reader.bin.gz')));
const n = meta.floats, weights = engine._malloc(n * 4);
for (let b = 0; b < 4; b++) for (let i = 0; i < n; i++) engine.HEAPU8[weights + i * 4 + b] = shuffled[b * n + i];
assert.equal(engine._crc32(weights, n * 4, 0) >>> 0, meta.crc32 >>> 0, 'fp32 weight CRC');
const c = meta.config;
const model = engine._reader_new(weights, c.n_layer, c.n_head, c.n_embd, c.vocab_size, c.block_size,
  meta.hidden_size, meta.d_copy, 128, 16, meta.rms_eps, meta.rope_base);
assert.ok(model, 'reader allocation');
const alloc = (a) => {
  const ptr = engine._malloc(a.byteLength);
  new a.constructor(engine.HEAPU8.buffer, ptr, a.length).set(a);
  return ptr;
};
const p = g.P, ids = Int32Array.from(g.ids);
const positions = Int32Array.from({ length: p }, (_, i) => i);
const slots = Int32Array.from({ length: p }, (_, i) => i);
const offsets = Int32Array.from({ length: p + 1 }, (_, i) => i * p);
const contexts = Int32Array.from({ length: p * p }, (_, i) => i % p);
const hp = engine._malloc(p * g.C * 4);
assert.equal(engine._reader_forward(model, p, alloc(ids), alloc(positions), alloc(slots), alloc(offsets), alloc(contexts), hp), 0);
const hPrefix = Float32Array.from(new Float32Array(engine.HEAPU8.buffer, hp, p * g.C));
assert.equal(engine._reader_grow(model, 256, p), 0, 'grow cache and keep prompt KV slots');
const decodeSlot = alloc(Int32Array.of(p)), decodePos = alloc(Int32Array.of(p));
const decodeTok = alloc(Int32Array.of(g.extra)), decodeOff = alloc(Int32Array.of(0, p + 1));
const decodeCtx = alloc(Int32Array.from({ length: p + 1 }, (_, i) => i));
const hd = engine._malloc(g.C * 4);
assert.equal(engine._reader_forward(model, 1, decodeTok, decodePos, decodeSlot, decodeOff, decodeCtx, hd), 0);
const hDecode = Float32Array.from(new Float32Array(engine.HEAPU8.buffer, hd, g.C));
const lp = engine._malloc(g.vocab * 4);
engine._reader_logits(model, hd, 1, lp);
const logits = Float32Array.from(new Float32Array(engine.HEAPU8.buffer, lp, g.vocab));
const kp = engine._malloc(p * g.DC * 4);
engine._reader_copy_keys(model, hp, p, kp);
const allKeys = new Float32Array(engine.HEAPU8.buffer, kp, p * g.DC);
const keys = Float32Array.from(allKeys.subarray(g.DC, (p - 1) * g.DC));
const copyIds = ids.slice(1, -1), keyPtr = alloc(keys), idPtr = alloc(copyIds), outPtr = engine._malloc(g.vocab * 4);
engine._reader_mix(model, hd, lp, keyPtr, idPtr, copyIds.length, outPtr);
const mixed = Float32Array.from(new Float32Array(engine.HEAPU8.buffer, outPtr, g.vocab));

const maxDiff = (a, b) => {
  assert.equal(a.length, b.length);
  let d = 0;
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
};
let at = 0;
const expectedPrefix = want.subarray(at, at += p * g.C);
const expectedDecode = want.subarray(at, at += g.C);
const expectedLogits = want.subarray(at, at += g.vocab);
const expectedKeys = want.subarray(at, at += (p - 2) * g.DC);
const expectedMixed = want.subarray(at);
const diffs = {
  prefix: maxDiff(hPrefix, expectedPrefix), decode: maxDiff(hDecode, expectedDecode),
  logits: maxDiff(logits, expectedLogits), copyKeys: maxDiff(keys, expectedKeys), copyMix: maxDiff(mixed, expectedMixed),
};
for (const [name, diff] of Object.entries(diffs)) assert.ok(diff < 1e-4, `${name}: max diff ${diff}`);
console.log('reader WASM parity:', diffs);
