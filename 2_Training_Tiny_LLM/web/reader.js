// Exact-fp32 v10 reader bindings. The inference worker owns one instance and one active question at a time.
import { hasSimd } from './kinship.js';

export class ReaderWasm {
  static async load(meta, shuffledWeights, engine = 'auto') {
    const enginePath = engine === 'mt' ? (hasSimd() ? './reader-simd-mt.mjs' : './reader-scalar-mt.mjs')
      : engine === 'simd' ? './reader-simd.mjs' : engine === 'scalar' ? './reader-scalar.mjs'
      : hasSimd() ? './reader-simd.mjs' : './reader-scalar.mjs';
    const mod = await import(enginePath).then((m) => m.default());
    const n = meta.floats;
    if (shuffledWeights.byteLength !== n * 4) throw new Error(`reader weight bytes ${shuffledWeights.byteLength} != ${n * 4}`);
    const weights = mod._malloc(n * 4);
    if (!weights) throw new Error('not enough memory for the exact fp32 reader weights');
    const bytes = new Uint8Array(shuffledWeights.buffer, shuffledWeights.byteOffset, shuffledWeights.byteLength);
    for (let plane = 0; plane < 4; plane++) {
      for (let i = 0; i < n; i++) mod.HEAPU8[weights + i * 4 + plane] = bytes[plane * n + i];
    }
    const crc = mod._crc32(weights, n * 4, 0) >>> 0;
    if (crc !== (meta.crc32 >>> 0)) throw new Error(`reader weights CRC mismatch (${crc} != ${meta.crc32})`);
    const c = meta.config;
    const ptr = mod._reader_new(weights, c.n_layer, c.n_head, c.n_embd, c.vocab_size, c.block_size,
      meta.hidden_size, meta.d_copy, Math.min(128, c.block_size), c.block_size, meta.rms_eps, meta.rope_base);
    if (!ptr) throw new Error('not enough memory to initialize the exact fp32 reader');
    return new ReaderWasm(mod, meta, ptr, weights, enginePath.includes('reader-simd') ? 'simd' : 'scalar');
  }

  static async loadStream(meta, shuffledStream, engine = 'auto') {
    const enginePath = engine === 'mt' ? (hasSimd() ? './reader-simd-mt.mjs' : './reader-scalar-mt.mjs')
      : engine === 'simd' ? './reader-simd.mjs' : engine === 'scalar' ? './reader-scalar.mjs'
      : hasSimd() ? './reader-simd.mjs' : './reader-scalar.mjs';
    const mod = await import(enginePath).then((m) => m.default());
    const n = meta.floats;
    const weights = mod._malloc(n * 4);
    if (!weights) throw new Error('not enough memory for the exact fp32 reader weights');
    const reader = shuffledStream.getReader();
    let read = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (read + value.length > n * 4) throw new Error('reader weights are larger than expected');
      const heap = mod.HEAPU8;
      for (let j = 0; j < value.length;) {
        const plane = Math.floor(read / n), index = read - plane * n;
        const run = Math.min(value.length - j, n - index);
        for (let k = 0; k < run; k++) heap[weights + 4 * (index + k) + plane] = value[j + k];
        read += run;
        j += run;
      }
    }
    if (read !== n * 4) throw new Error(`reader weights are truncated (${read} of ${n * 4} bytes)`);
    const crc = mod._crc32(weights, n * 4, 0) >>> 0;
    if (crc !== (meta.crc32 >>> 0)) throw new Error(`reader weights CRC mismatch (${crc} != ${meta.crc32})`);
    const c = meta.config;
    const ptr = mod._reader_new(weights, c.n_layer, c.n_head, c.n_embd, c.vocab_size, c.block_size,
      meta.hidden_size, meta.d_copy, Math.min(128, c.block_size), c.block_size, meta.rms_eps, meta.rope_base);
    if (!ptr) throw new Error('not enough memory to initialize the exact fp32 reader');
    return new ReaderWasm(mod, meta, ptr, weights, enginePath.includes('reader-simd') ? 'simd' : 'scalar');
  }

  constructor(mod, meta, ptr, weights, engine) {
    this.mod = mod; this.meta = meta; this.ptr = ptr; this.weights = weights;
    this.engine = engine;
    this.C = meta.config.n_embd; this.V = meta.config.vocab_size; this.DC = meta.d_copy;
    this.cap = Math.min(128, meta.config.block_size); this.usedSlots = 0;
  }

  setThreads(maxThreads) {
    if (typeof this.mod._reader_set_threads !== 'function') return 1;
    // never more threads than build.sh pre-starts (PTHREAD_POOL_SIZE + 1): set_threads would wait forever
    const pool = Math.max(1, Math.min(globalThis.navigator?.hardwareConcurrency || 4, 4) - 1) + 1;
    const threads = this.mod._reader_set_threads(this.ptr, Math.max(1, Math.min(pool, maxThreads | 0)));
    if (threads > 1 && !this.engine.endsWith('-mt')) this.engine += '-mt';
    return threads;
  }

  ensureCapacity(need) {
    if (need <= this.cap) return;
    let cap = this.cap;
    while (cap < need) cap = Math.min(this.meta.config.block_size * 4, cap * 2);
    if (cap < need || this.mod._reader_grow(this.ptr, cap, this.usedSlots) !== 0) {
      throw new Error('not enough memory for the reader KV cache on this device');
    }
    this.cap = cap;
  }

  alloc(data) {
    const ptr = this.mod._malloc(data.byteLength);
    if (!ptr) throw new Error('not enough memory for reader scratch buffers');
    new data.constructor(this.mod.HEAPU8.buffer, ptr, data.length).set(data);
    return ptr;
  }

  forward(rows, { lastOnly = false, copyKeys = true } = {}) {
    if (!rows.length || rows.length > this.meta.config.block_size) throw new Error('reader batch exceeds the context window');
    const need = Math.max(...rows.map((r) => r.slot)) + 1;
    this.ensureCapacity(need);
    const tokens = Int32Array.from(rows, (r) => r.token);
    const positions = Int32Array.from(rows, (r) => r.position);
    const slots = Int32Array.from(rows, (r) => r.slot);
    const offsets = new Int32Array(rows.length + 1), context = [];
    rows.forEach((row, i) => { offsets[i] = context.length; context.push(...row.context); });
    offsets[rows.length] = context.length;
    const ti = this.alloc(tokens), po = this.alloc(positions), sl = this.alloc(slots);
    const off = this.alloc(offsets), ctx = this.alloc(Int32Array.from(context));
    const h = this.mod._malloc(rows.length * this.C * 4);
    if (!h) throw new Error('not enough memory for reader outputs');
    const status = this.mod._reader_forward(this.ptr, rows.length, ti, po, sl, off, ctx, h);
    if (status !== 0) throw new Error(`reader forward failed (${status})`);
    this.usedSlots = Math.max(this.usedSlots, need);
    const hidden = Float32Array.from(new Float32Array(this.mod.HEAPU8.buffer, h, rows.length * this.C));
    const logitsCount = lastOnly ? 1 : rows.length;
    const logitsPtr = this.mod._malloc(logitsCount * this.V * 4);
    const keysPtr = copyKeys ? this.mod._malloc(rows.length * this.DC * 4) : 0;
    if (!logitsPtr || (copyKeys && !keysPtr)) throw new Error('not enough memory for reader output heads');
    const logitsHidden = lastOnly ? h + (rows.length - 1) * this.C * 4 : h;
    this.mod._reader_logits(this.ptr, logitsHidden, logitsCount, logitsPtr);
    if (copyKeys) this.mod._reader_copy_keys(this.ptr, h, rows.length, keysPtr);
    const raw = Float32Array.from(new Float32Array(this.mod.HEAPU8.buffer, logitsPtr, logitsCount * this.V));
    const keys = copyKeys ? Float32Array.from(new Float32Array(this.mod.HEAPU8.buffer, keysPtr, rows.length * this.DC)) : null;
    const states = rows.map((row, i) => ({ hidden: hidden.subarray(i * this.C, (i + 1) * this.C),
      logits: lastOnly ? (i === rows.length - 1 ? raw : null) : raw.subarray(i * this.V, (i + 1) * this.V),
      key: keys?.subarray(i * this.DC, (i + 1) * this.DC), row }));
    for (const [name, ptr] of Object.entries({ ti, po, sl, off, ctx, h, logitsPtr, keysPtr })) this.mod._free(ptr);
    return states;
  }

  logProbabilities(state, promptKeys, promptIds) {
    const h = this.alloc(state.hidden), logits = this.alloc(state.logits);
    const keys = this.alloc(promptKeys), ids = this.alloc(Int32Array.from(promptIds));
    const out = this.mod._malloc(this.V * 4);
    if (!out) throw new Error('not enough memory for reader probabilities');
    this.mod._reader_mix(this.ptr, h, logits, keys, ids, promptIds.length, out);
    const result = Float32Array.from(new Float32Array(this.mod.HEAPU8.buffer, out, this.V));
    for (const p of [h, logits, keys, ids, out]) this.mod._free(p);
    return result;
  }

  prefill(inputIds, maxNew = 96) {
    const maxPrefix = Math.max(2, this.meta.config.block_size - maxNew);
    const ids = inputIds.length > maxPrefix ? inputIds.slice(-maxPrefix) : inputIds;
    const P = ids.length;
    if (P < 2 || P > this.meta.config.block_size) throw new Error('reader prompt exceeds its context window');
    const context = Array.from({ length: P }, (_, i) => i);
    const rows = ids.map((token, i) => ({ token, position: i, slot: i, context }));
    const states = this.forward(rows, { lastOnly: true });
    const promptIds = ids.slice(1, -1);
    const promptKeys = Float32Array.from(states.slice(1, -1).flatMap((s) => [...s.key]));
    const logprobs = this.logProbabilities(states.at(-1), promptKeys, promptIds);
    return { prefixLength: P, promptIds, promptKeys, logprobs };
  }

  decode(state, token, generated) {
    const slot = state.prefixLength + generated.length;
    const context = Array.from({ length: slot + 1 }, (_, i) => i);
    const decoded = this.forward([{ token, position: slot, slot, context }], { copyKeys: false })[0];
    return this.logProbabilities(decoded, state.promptKeys, state.promptIds);
  }
}
