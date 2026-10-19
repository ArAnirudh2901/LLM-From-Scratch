// Kinship QA in the browser: the name tagger and the QA model (exact fp32 weights) run in a 20 KB WebAssembly
// engine. Port of ask_qa.answer + fast_qa (Python reference); load once with KinshipQA.load(), then ask() many times.
// Runs on old hardware too: SIMD build when the browser supports WebAssembly SIMD, plain build otherwise; no regex
// lookbehind or BigInt (Safari < 16.4); weights stream straight into engine memory (low peak RAM).

const W = '\\p{L}\\p{N}_';   // Python's Unicode \w
const QA_TOKEN = new RegExp(`@\\d+|'s|[${W}]+(?:-[${W}]+)*|[^${W}\\s]`, 'gu');
const TAG_WORD = new RegExp(`'s|[${W}]+(?:-[${W}]+)*|[^${W}\\s]`, 'gu');
const BRACKET = /\[([^\]]+)\]/g;
const PRONOUN_START = /^(he|she|his|her|him|they|their|them)\b/i;

// smallest module using a SIMD instruction (as in wasm-feature-detect)
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
const RELAXED_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11]);
const valid = (bytes) => { try { return WebAssembly.validate(bytes); } catch { return false; } };
export const hasSimd = () => valid(SIMD_PROBE);
// Threads need SharedArrayBuffer, which browsers give only to cross-origin isolated pages (COOP/COEP headers).
const canThread = () => typeof SharedArrayBuffer !== 'undefined' && globalThis.crossOriginIsolated !== false &&
  (globalThis.navigator?.hardwareConcurrency || 1) > 1;
// best engine build this browser can run: threads, then fused multiply-add > SIMD > plain
export const bestEngine = () => {
  const simd = valid(SIMD_PROBE), relaxed = simd && valid(RELAXED_PROBE), mt = canThread() ? '-mt' : '';
  return relaxed ? `relaxed${mt}` : simd ? `simd${mt}` : canThread() ? 'scalar-mt' : 'scalar';
};
const ENGINES = { 'relaxed-mt': './engine-relaxed-mt.mjs', 'simd-mt': './engine-mt.mjs', 'scalar-mt': './engine-scalar-mt.mjs',
                  relaxed: './engine-relaxed.mjs', simd: './engine.mjs', scalar: './engine-scalar.mjs' };

function splitSentences(text) {   // Python: re.split(r'(?<=[.!?])\s+', text)
  const parts = text.split(/([.!?])\s+/), out = [];
  for (let i = 0; i < parts.length; i += 2) out.push(parts[i] + (parts[i + 1] || ''));
  return out;
}

function wrapName(text, n) {   // [brackets] around whole-word mentions not already bracketed
  return text.replace(new RegExp(`(^|[^\\[${W}])(${escapeRe(n)})(?![\\]${W}])`, 'gu'), (_, pre) => `${pre}[${n}]`);
}

// Byte-plane weights (export_web.py) streamed into WASM memory: byte b of float i lands at w + 4i + b.
async function streamWeights(mod, res, w, n, name) {
  const gz = res.headers.get('content-type') !== 'application/octet-stream-raw';
  const body = gz && typeof DecompressionStream !== 'undefined' ? res.body.pipeThrough(new DecompressionStream('gzip')) : res.body;
  const reader = body.getReader();
  let g = 0;
  for (;;) {
    let chunk;
    try { chunk = await reader.read(); }
    catch (e) { throw new Error(`${name} weights file is corrupted or incomplete (${e.message || 'decompression failed'}). Re-copy the model files.`); }
    const { done, value } = chunk;
    if (done) break;
    const heap = mod.HEAPU8;
    for (let j = 0; j < value.length;) {
      const b = Math.floor(g / n), i = g - b * n, run = Math.min(value.length - j, n - i);
      for (let k = 0; k < run; k++) heap[w + 4 * (i + k) + b] = value[j + k];
      j += run; g += run;
    }
  }
  if (g !== 4 * n) throw new Error(`${name} weights file has ${g} bytes, expected ${4 * n}: truncated, corrupted or not matching ${name}.json. Re-copy the model files.`);
}

// One model inside the WASM engine: weights copied in once, a KV slot pool, reusable input buffers.
class Net {
  constructor(mod, meta, cap, rmax) {
    this.mod = mod;
    this.cfg = meta.config;
    this.vocab = meta.vocab;
    this.stoi = new Map(meta.vocab.map((t, i) => [t, i]));
    this.n = meta.floats;
    this.wptr = mod._malloc(4 * this.n);
    if (!this.wptr) throw new Error('out of memory');
    this.C = this.cfg.n_embd;
    this.cap = cap;
    this.rmax = rmax;
    this.bufs = {};
  }
  init() {   // after the weights are in place
    const c = this.cfg;
    this.ptr = this.mod._model_new(this.wptr, c.n_layer, c.n_head, c.n_embd, c.vocab_size, c.block_size, this.cap, this.rmax);
    if (!this.ptr) throw new Error('out of memory');
  }
  grow(need, used) {   // double the KV slot pool until `need` slots fit
    let cap = this.cap;
    while (cap < need) cap *= 2;
    if (this.mod._model_grow(this.ptr, cap, used) !== 0) throw new Error('out of memory');
    this.cap = cap;
  }
  buf(name, n) {   // int32/float32 scratch in WASM memory, grown on demand
    const b = this.bufs[name];
    if (b && b.n >= n) return b.ptr;
    if (b) this.mod._free(b.ptr);
    const ptr = this.mod._malloc(4 * Math.max(n, 16));
    this.bufs[name] = { ptr, n: Math.max(n, 16) };
    return ptr;
  }
  i32(ptr, n) { return new Int32Array(this.mod.HEAPU8.buffer, ptr, n); }
  f32(ptr, n) { return new Float32Array(this.mod.HEAPU8.buffer, ptr, n); }
  // rows: {tok, pos, slot, ctx: arrays of slot lists}; returns ln_f(hidden) [R*C] (a copy)
  forward(tok, pos, slot, ctx) {
    const R = tok.length;
    if (R > this.rmax) {   // split big batches; later rows only see earlier slots, which are already written
      const out = new Float32Array(R * this.C);
      for (let i = 0; i < R; i += this.rmax) {
        const j = Math.min(R, i + this.rmax);
        out.set(this.forward(tok.slice(i, j), pos.slice(i, j), slot.slice(i, j), ctx.slice(i, j)), i * this.C);
      }
      return out;
    }
    let nnz = 0;
    for (const c of ctx) nnz += c.length;
    const pTok = this.buf('tok', R), pPos = this.buf('pos', R), pSlot = this.buf('slot', R);
    const pOff = this.buf('off', R + 1), pCtx = this.buf('ctx', nnz), pOut = this.buf('out', R * this.C);
    this.i32(pTok, R).set(tok); this.i32(pPos, R).set(pos); this.i32(pSlot, R).set(slot);
    const off = this.i32(pOff, R + 1), cs = this.i32(pCtx, nnz);
    let k = 0;
    for (let r = 0; r < R; r++) { off[r] = k; cs.set(ctx[r], k); k += ctx[r].length; }
    off[R] = k;
    if (this.mod._forward(this.ptr, R, pTok, pPos, pSlot, pOff, pCtx, pOut) !== 0) throw new Error('batch too large');
    return this.f32(pOut, R * this.C).slice();
  }
  logits(h, n, withLse = false) {   // full vocabulary logits [n * V] (+ log-sum-exp per row)
    const V = this.cfg.vocab_size, pH = this.buf('h', n * this.C), pL = this.buf('lg', n * V), pS = this.buf('lse', n);
    this.f32(pH, n * this.C).set(h.subarray(0, n * this.C));
    this.mod._logits(this.ptr, pH, n, pL, withLse ? pS : 0);
    const lg = this.f32(pL, n * V).slice();
    return withLse ? [lg, this.f32(pS, n).slice()] : lg;
  }
  logitsSome(h, n, ids) {   // logits for chosen vocabulary ids only [n * ids.length]
    const k = ids.length, pH = this.buf('h', n * this.C), pI = this.buf('ids', k), pL = this.buf('lg', n * k);
    this.f32(pH, n * this.C).set(h.subarray(0, n * this.C));
    this.i32(pI, k).set(ids);
    this.mod._logits_some(this.ptr, pH, n, pI, k, pL);
    return this.f32(pL, n * k).slice();
  }
}

const unique = (xs) => [...new Set(xs)];
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export function tokenize(text) { return text.toLowerCase().match(QA_TOKEN) || []; }

export function linkedInText(story, a, b) {   // qa_data.linked_in_text
  const groups = [];
  let cur = [];
  for (const sent of splitSentences(story.trim())) {
    if (cur.length && !PRONOUN_START.test(sent.trim())) { groups.push(cur); cur = []; }
    for (const m of sent.matchAll(BRACKET)) cur.push(m[1]);
  }
  groups.push(cur);
  const comp = new Map();
  for (const g of groups) {
    const merged = new Set();
    for (const n of g) for (const x of (comp.get(n) || [n])) merged.add(x);
    for (const n of merged) comp.set(n, merged);
  }
  return comp.has(a) && comp.has(b) && comp.get(a) === comp.get(b);
}

export function bracketNames(text, names) {   // name_tagger.bracket_names
  for (const n of [...names].sort((x, y) => y.length - x.length)) text = wrapName(text, n);
  return text;
}

export class KinshipQA {
  // engine: 'auto' (best this browser supports), 'relaxed', 'simd' or 'scalar'. fetchFn may add caching.
  static async load({ base = './model/', fetchFn = fetch, engine = 'auto', engineModule, maxThreads = 4, onProgress = () => {} } = {}) {
    let kind = engine === 'auto' ? bestEngine() : engine;
    let mod;
    try { mod = await import(engineModule || ENGINES[kind]).then((m) => m.default()); }
    catch (e) {   // threaded build refused (e.g. memory): single-threaded instead
      if (!kind.endsWith('-mt')) throw e;
      kind = kind.slice(0, -3);
      mod = await import(ENGINES[kind]).then((m) => m.default());
    }
    const simd = kind !== 'scalar';
    const maxT = kind.endsWith('-mt') ? Math.min(globalThis.navigator?.hardwareConcurrency || 4, maxThreads) : 1;
    const get = async (name) => {
      const res = await fetchFn(base + name);
      if (!res.ok) throw new Error(`${name}: ${res.status}`);
      return res;
    };
    const [qaMeta, tagMeta] = await Promise.all([get('qa.json').then((r) => r.json()), get('tagger.json').then((r) => r.json())]);
    const tc = tagMeta.config;
    const qa = new Net(mod, qaMeta, 512, 256), tagger = new Net(mod, tagMeta, tc.block_size, tc.block_size);
    onProgress('downloading');
    const file = (name) => typeof DecompressionStream !== 'undefined' ? get(`${name}.bin.gz`)
      : get(`${name}.bin`).then((r) => new Response(r.body, { headers: { 'content-type': 'application/octet-stream-raw' } }));
    await Promise.all([file('qa').then((r) => streamWeights(mod, r, qa.wptr, qa.n, 'qa')),
                       file('tagger').then((r) => streamWeights(mod, r, tagger.wptr, tagger.n, 'tagger'))]);
    for (const [net, meta, name] of [[qa, qaMeta, 'qa'], [tagger, tagMeta, 'tagger']]) {   // refuse damaged weights
      if (meta.vocab.length !== meta.config.vocab_size) throw new Error(`${name}.json is damaged (vocab size)`);
      const crc = mod._crc32(net.wptr, 4 * net.n, 0) >>> 0;
      if (crc !== meta.crc32) throw new Error(`${name} weights are corrupted or do not match ${name}.json (CRC ${crc} != ${meta.crc32}). Re-copy the model files.`);
    }
    qa.init(); tagger.init();
    const threads = maxT > 1 ? mod._model_set_threads(qa.ptr, maxT) : 1;   // thread pool is shared by both models
    onProgress('ready');
    const out = new KinshipQA(qa, tagger, qaMeta, simd);
    out.engine = kind;
    out.tile = mod._tile_choice();   // GEMM tile picked for this CPU (rows*10 + outputs)
    out.threads = threads;
    return out;
  }

  constructor(qa, tagger, meta, simd) {
    this.qa = qa; this.tagger = tagger; this.meta = meta; this.simd = simd;
    const s = qa.stoi;
    this.ids = {
      sep: s.get('<sep>'), a: s.get('<a>'), c: s.get('<c>'), nofact: s.has(meta.nofact) ? s.get(meta.nofact) : -1,
      unknown: s.has(meta.unknown) ? s.get(meta.unknown) : -1,
      rel: meta.relations.map((r) => s.get(r)), edge: meta.edge_relations.map((r) => s.get(r)),
      ent: meta.entities.map((e) => s.get(e)),
    };
    this.entBase = this.ids.ent[0];
  }

  // ---- name tagger (name_tagger.find_names) ----
  findNames(text) {
    const t = this.tagger, words = text.match(TAG_WORD) || [];
    const limit = Math.floor((t.cfg.block_size - 2) / 2);
    const chunks = [];
    let cur = [];
    for (const w of words) {
      cur.push(w);
      if ('.!?'.includes(w) && w.length === 1 && cur.length > Math.floor(limit / 2)) { chunks.push(cur); cur = []; }
    }
    chunks.push(cur);
    const unk = t.stoi.get('<Unk>'), unkLow = t.stoi.get('<unk>'), bos = t.stoi.get('<bos>'), sep = t.stoi.get('<sep>');
    const N = t.stoi.get('<N>'), O = t.stoi.get('<O>');
    const names = [];
    for (const c of chunks) {
      for (let i = 0; i < Math.max(1, c.length); i += limit) {
        const ws = c.slice(i, i + limit);
        const ids = ws.map((w) => t.stoi.has(w) ? t.stoi.get(w)
          : (w[0] !== w[0].toLowerCase() && w[0] === w[0].toUpperCase() ? unk : unkLow));
        const seq = [bos, ...ids, sep, ...ids], n = seq.length;
        const slot = seq.map((_, j) => j), pos = slot;
        const ctx = seq.map((_, j) => slot.slice(0, j + 1));
        const h = t.forward(seq, pos, slot, ctx);
        const lg = t.logitsSome(h.subarray((ids.length + 2) * t.C), ids.length, [N, O]);
        ws.forEach((w, j) => { if (lg[2 * j] >= lg[2 * j + 1]) names.push(w); });   // argmax picks <N> on ties
        void n;
      }
    }
    return unique(names);
  }

  // ---- QA model ----
  viewTokens(ex, view) {   // fast_qa.view_tokens
    const names = unique([...[...ex.story.matchAll(BRACKET)].map((m) => m[1]), ex.a, ex.b]);
    const m = new Map(names.map((n, i) => [n, `@${i}`]));
    const story = ex.story.replace(BRACKET, (_, n) => ` ${m.get(n)} `);
    const p = this.meta.perms[view];
    const ren = (t) => /^@\d+$/.test(t) ? `@${p[+t.slice(1)]}` : t;
    const toks = ['<bos>', '<q>', ren(m.get(ex.a)), ren(m.get(ex.b)), '<s>', ...tokenize(story).map(ren), '<c>'];
    const unk = this.qa.stoi.get('<unk>');
    return { ids: toks.map((x) => this.qa.stoi.has(x) ? this.qa.stoi.get(x) : unk), names: m };
  }

  search(seqs, { beam = 8, branch = 3, prune = 10 } = {}) {   // fast_qa.search
    const q = this.qa, I = this.ids, V = q.cfg.vocab_size, block = q.cfg.block_size, n = seqs.length;
    let next = 0;
    const alloc = (k) => {
      if (next + k > q.cap) { if (next + k > 16384) throw new Error('slot pool full'); q.grow(next + k, next); }
      const s = [];
      for (let i = 0; i < k; i++) s.push(next++);
      return s;
    };
    // story prefixes
    const preSlots = seqs.map((p) => alloc(p.length));
    const tok = [], pos = [], slot = [], ctx = [];
    seqs.forEach((p, s) => p.forEach((t, j) => { tok.push(t); pos.push(j); slot.push(preSlots[s][j]); ctx.push(preSlots[s].slice(0, j + 1)); }));
    let h = q.forward(tok, pos, slot, ctx);
    let rows = tok.length;
    const pStart = new Float64Array(n);
    if (I.unknown >= 0) {
      let r = 0;
      const last = new Float32Array(n * q.C);
      seqs.forEach((p, s) => { r += p.length; last.set(h.subarray((r - 1) * q.C, r * q.C), s * q.C); });
      const lg = q.logitsSome(last, n, [...seqs.map((p) => p[2]), I.a]);
      // logitsSome returned [n][n+1]: column s is sequence s's own A token, column n is <a>
      for (let s = 0; s < n; s++) {
        const x = lg[s * (n + 1) + s], y = lg[s * (n + 1) + n], mx = Math.max(x, y);
        pStart[s] = Math.exp(y - mx) / (Math.exp(x - mx) + Math.exp(y - mx));
      }
    }
    const target = seqs.map((p) => p[3]);
    const people = seqs.map((p) => { let b = 0; for (const t of new Set(p.slice(5))) if (t >= this.entBase && t < this.entBase + 32) b |= 1 << (t - this.entBase); return b; });
    const votes = seqs.map(() => new Float64Array(I.rel.length));
    const mass = new Float64Array(n), validMass = new Float64Array(n), bestDone = new Float64Array(n).fill(-Infinity);
    const bestChain = seqs.map(() => '');
    let hyps = seqs.map((p, s) => ({ s, ch: [p[2]], sl: [], vis: 1 << (p[2] - this.entBase), score: 0, ph: 0, lv: 0 }));
    while (hyps.length) {
      const tk = [], ps = [], st = [], cx = [], where = [];
      for (const hy of hyps) {
        const first = tk.length, fresh = alloc(hy.ch.length - hy.sl.length);
        for (let j = hy.sl.length; j < hy.ch.length; j++) {
          const k = j - hy.sl.length;
          tk.push(hy.ch[j]); ps.push(seqs[hy.s].length + j); st.push(fresh[k]);
          cx.push([...preSlots[hy.s], ...hy.sl, ...fresh.slice(0, k + 1)]);
        }
        where.push([first, tk.length, fresh]);
      }
      h = q.forward(tk, ps, st, cx);
      rows += tk.length;
      const want = new Float32Array(hyps.length * q.C), check = [];
      where.forEach(([first, end], j) => {
        want.set(h.subarray((end - 1) * q.C, end * q.C), j * q.C);
        if (hyps[j].ph !== 1 && hyps[j].ch.length > 2) check.push([j, end - 3]);
      });
      const [lg, lse] = q.logits(want, hyps.length, true);
      const ok = new Float64Array(hyps.length);
      if (check.length && I.nofact >= 0) {
        const hc = new Float32Array(check.length * q.C);
        check.forEach(([, r], i) => hc.set(h.subarray(r * q.C, (r + 1) * q.C), i * q.C));
        const two = q.logitsSome(hc, check.length, [I.sep, I.nofact]);
        check.forEach(([j], i) => {
          const a = two[2 * i], b = two[2 * i + 1], m2 = Math.max(a, b);
          ok[j] = a - m2 - Math.log(Math.exp(a - m2) + Math.exp(b - m2));
        });
      }
      const next_ = [];
      hyps.forEach((hy, j) => {
        const row = lg.subarray(j * V, (j + 1) * V), lp = (i) => row[i] - lse[j];
        const sl = [...hy.sl, ...where[j][2]];
        const lv = hy.lv + ok[j];
        if (hy.ph === 2) {
          let amx = -Infinity;
          for (const r of I.rel) if (row[r] > amx) amx = row[r];
          let az = 0;
          for (const r of I.rel) az += Math.exp(row[r] - amx);
          const w = Math.exp(hy.score + lv);
          I.rel.forEach((r, k) => { votes[hy.s][k] += w * Math.exp(row[r] - amx) / az; });
          mass[hy.s] += Math.exp(hy.score); validMass[hy.s] += w;
          if (hy.score + lv > bestDone[hy.s]) {
            bestDone[hy.s] = hy.score + lv;
            bestChain[hy.s] = hy.ch.slice(0, -1).map((t) => q.vocab[t]).join(' ');
          }
        } else if (hy.ph === 0) {
          const top = I.edge.map((t) => [t, lp(t)]).sort((x, y) => y[1] - x[1]).slice(0, branch);
          for (const [t, x] of top) next_.push({ s: hy.s, ch: [...hy.ch, t], sl, vis: hy.vis, score: hy.score + x, ph: 1, lv });
        } else {
          const free = people[hy.s] & ~hy.vis;
          const top = I.ent.map((t, i) => [t, (free >>> i) & 1 ? lp(t) : -Infinity])
            .sort((x, y) => y[1] - x[1]).slice(0, branch);
          for (const [t, x] of top) {
            if (x === -Infinity) continue;
            if (t === target[hy.s]) next_.push({ s: hy.s, ch: [...hy.ch, t, I.sep, I.a], sl, vis: hy.vis, score: hy.score + x, ph: 2, lv });
            else if (seqs[hy.s].length + hy.ch.length < block - 8)
              next_.push({ s: hy.s, ch: [...hy.ch, t, I.sep, t], sl, vis: hy.vis | (1 << (t - this.entBase)), score: hy.score + x, ph: 0, lv });
          }
        }
      });
      next_.sort((x, y) => x.s - y.s || (y.score + y.lv) - (x.score + x.lv));
      const count = new Map();
      hyps = next_.filter((hy) => {
        const c = count.get(hy.s) || 0;
        if (c < beam && hy.score + hy.lv > bestDone[hy.s] - prune) { count.set(hy.s, c + 1); return true; }
        return false;
      });
    }
    const pLink = Array.from(mass, (m, s) => m > 0 ? validMass[s] / m : 0);
    return { votes, bestChain, pStart: Array.from(pStart), pLink, rows };
  }

  // Defaults from test/calib.json (950 questions): 1 view + beam 4 matches 4 views + beam 8 on average at ~1/6 the work.
  predict(ex, { views = 1, adaptive = null, unkThreshold = 0.5, beam = 4, ...kw } = {}) {   // fast_qa.predict, one question
    const R = this.meta.relations, votes = new Float64Array(R.length);
    let pStart = 0, pLink = 0, used = 0, rows = 0, chain = '';
    const perView = [];
    const first = this.viewTokens(ex, 0);
    const add = (res, j) => {
      const tot = Math.max(res.votes[j].reduce((x, y) => x + y, 0), 1e-30);
      res.votes[j].forEach((x, k) => { votes[k] += x / tot; });
      pStart += res.pStart[j]; pLink += res.pLink[j]; used++;
      perView.push({ votes: Array.from(res.votes[j], (x) => x / tot), pStart: res.pStart[j], pLink: res.pLink[j] });
    };
    kw = { beam, ...kw };
    const r0 = this.search([first.ids], kw);
    rows += r0.rows; chain = r0.bestChain[0]; add(r0, 0);
    const clear = adaptive && Math.max(...votes) >= adaptive[0] && Math.abs(Math.max(pStart, 1 - pLink) - unkThreshold) >= adaptive[1];
    if (views > 1 && !clear) {   // remaining views in one batched search: weights are read once per step for all
      const seqs = [];
      for (let v = 1; v < views; v++) seqs.push(this.viewTokens(ex, v).ids);
      let parts;
      try { parts = [[this.search(seqs, kw), seqs.length]]; }
      catch (e) { if (!/slot pool full/.test(e.message)) throw e; parts = seqs.map((s) => [this.search([s], kw), 1]); }
      for (const [res, n] of parts) { rows += res.rows; for (let j = 0; j < n; j++) add(res, j); }
    }
    pStart /= used; pLink /= used;
    let best = 0;
    for (let k = 1; k < R.length; k++) if (votes[k] > votes[best]) best = k;
    const share = votes[best] / used, pUnknown = Math.max(pStart, 1 - pLink);
    const total = votes.reduce((a, b) => a + b, 0);
    const relation = pUnknown >= unkThreshold || total === 0 ? this.meta.unknown : R[best];
    const back = new Map([...first.names].map(([n, t]) => [t, n]));
    const steps = chain.split('<sep>').map((s) => s.trim()).filter(Boolean).map((s) => s.split(/\s+/).map((w) => back.get(w) || w).join(' '));
    return { relation, steps, share, pUnknown, pStart, pLink, views: used, rows, perView };
  }

  prefixFits(ex, room = 16) { return this.viewTokens(ex, 0).ids.length <= this.qa.cfg.block_size - room; }

  // ---- the whole question (ask_qa.answer) ----
  ask(story, a, b, { names = [], linkCheck = true, minConf = 0.7, ...opts } = {}) {
    const t0 = performance.now();
    let people;
    if (story.includes('[')) people = [...story.matchAll(BRACKET)].map((m) => m[1]);
    else { people = this.findNames(story); story = bracketNames(story, people); }
    for (const n of names) story = wrapName(story, n);
    people = unique([...people, ...names]);
    const done = (o) => ({ ...o, people, ms: performance.now() - t0 });
    const lookup = new Map(people.map((p) => [p.toLowerCase(), p]));
    const missing = [a, b].filter((x) => !lookup.has(x.toLowerCase()));
    if (missing.length)
      return done({ kind: 'refused', text: `Can't answer: ${missing.join(' and ')} ${missing.length === 1 ? 'is' : 'are'} not a person in the story. I only answer how two people in the story are related.` });
    a = lookup.get(a.toLowerCase()); b = lookup.get(b.toLowerCase());
    if (a === b) return done({ kind: 'same', text: `${a} and ${b} are the same person.` });
    if (linkCheck && !linkedInText(story, a, b))
      return done({ kind: 'unknown', text: `The story doesn't say how ${b} is related to ${a} (they are never connected in the text).` });
    const ex = { story, a, b };
    if (!this.prefixFits(ex))
      return done({ kind: 'too long', text: `Can't answer: the story is too long for this model (about ${this.qa.cfg.block_size - 40} words at most).` });
    const r = this.predict(ex, opts);
    const f = (x) => x.toFixed(2);
    if (r.relation === this.meta.unknown)
      return done({ kind: 'unknown', ...r, steps: [], text: `The story doesn't say how ${b} is related to ${a} (p_unknown ${f(r.pUnknown)}).` });
    if (r.share < minConf)
      return done({ kind: 'not sure', ...r, text: `Not sure. Best guess: ${b} is ${a}'s ${r.relation} (agreement ${f(r.share)}, p_unknown ${f(r.pUnknown)}).` });
    return done({ kind: 'answer', ...r, text: `${b} is ${a}'s ${r.relation} (agreement ${f(r.share)}, p_unknown ${f(r.pUnknown)}).` });
  }
}
