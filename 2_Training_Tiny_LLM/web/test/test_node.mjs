// Compares the browser engine (kinship.js + engine.wasm) with the Python reference outputs in golden.json,
// then measures accuracy and speed for search settings.  Run: node web/test/test_node.mjs [--sweep]
import { readFile } from 'node:fs/promises';
import { KinshipQA } from '../kinship.js';

const here = new URL('.', import.meta.url);
const fetchFn = async (url) => new Response(await readFile(new URL(url, here)));
const arg = (k, d) => (process.argv.find((a) => a.startsWith(`--${k}=`)) || `=${d}`).split('=')[1];
if (process.argv.includes('--nogzip')) delete globalThis.DecompressionStream;   // old-browser download path
const t0 = performance.now();
const qa = await KinshipQA.load({ base: '../model/', fetchFn, engine: arg('engine', 'auto') });
const mb = (x) => (x / 1e6).toFixed(0);
console.log(`load: ${(performance.now() - t0).toFixed(0)} ms | engine ${qa.engine}, tile ${qa.tile}, threads ${qa.threads} | ` +
            `${process.argv.includes('--nogzip') ? 'raw' : 'gzip'} download | WASM memory ${mb(qa.qa.mod.HEAPU8.length)} MB | RSS ${mb(process.memoryUsage().rss)} MB`);
const golden = JSON.parse(await readFile(new URL('golden.json', here)));

// 1. raw forward: logits at the last story token
for (const g of golden.forward) {
  const n = g.ids.length, slot = g.ids.map((_, j) => j);
  const h = qa.qa.forward(g.ids, slot, slot, slot.map((j) => slot.slice(0, j + 1)));
  const lg = qa.qa.logits(h.subarray((n - 1) * qa.qa.C), 1);
  let d = 0;
  g.logits.forEach((x, i) => { d = Math.max(d, Math.abs(x - lg[i])); });
  console.log(`forward (${n} tokens): max |logit diff| ${d.toExponential(2)}`);
}

// 2. single-view search
let same = 0, sameChain = 0, dS = 0, dL = 0;
for (const g of golden.search) {
  const r = qa.predict(g, { views: 1, adaptive: null, beam: 8 });
  same += r.relation === g.pred;
  const { bestChain } = qa.search([qa.viewTokens(g, 0).ids]);
  sameChain += bestChain[0] === g.chain;
  dS = Math.max(dS, Math.abs(r.pStart - g.p_start)); dL = Math.max(dL, Math.abs(r.pLink - g.p_link));
}
console.log(`search: same answer ${same}/${golden.search.length}, same chain ${sameChain}/${golden.search.length}, ` +
            `max |p_start diff| ${dS.toExponential(2)}, max |p_link diff| ${dL.toExponential(2)}`);

// 3. name tagger
let tagSame = 0;
for (const g of golden.tagger) {
  const names = qa.findNames(g.text);
  if (JSON.stringify(names) === JSON.stringify(g.names)) tagSame++;
  else console.log('  tagger differs:', names, 'vs', g.names);
}
console.log(`tagger: same names ${tagSame}/${golden.tagger.length}`);

// 4. accuracy and speed per search setting
const LIMIT = +(process.argv.find((a) => a.startsWith('--n='))?.slice(4) || 1e9);
if (process.argv.includes('--sweep')) {
  const configs = [
    ['beam 8, views 4 (reference)', { views: 4, adaptive: null }],
    ['beam 8, adaptive views', { views: 4, adaptive: [0.9, 0.3] }],
    ['beam 6, adaptive views', { views: 4, adaptive: [0.9, 0.3], beam: 6 }],
    ['beam 4, adaptive views', { views: 4, adaptive: [0.9, 0.3], beam: 4 }],
    ['beam 4, adaptive, prune 6', { views: 4, adaptive: [0.9, 0.3], beam: 4, prune: 6 }],
    ['beam 8, views 1', { views: 1, adaptive: null }],
  ];
  for (const [name, opt] of configs) {
    const parts = [];
    let ms = 0, n = 0, rows = 0, views = 0;
    for (const [set, all] of Object.entries(golden.sweep)) {
      const qs = all.slice(0, LIMIT);
      let hit = 0;
      for (const g of qs) {
        const t = performance.now();
        const r = qa.predict(g, opt);
        ms += performance.now() - t; n++; rows += r.rows; views += r.views;
        hit += r.relation === g.rel;
      }
      parts.push(`${set} ${(hit / qs.length).toFixed(3)}`);
    }
    console.log(`${name.padEnd(28)} | ${parts.join(' | ')} || ${(ms / n).toFixed(1)} ms/question, ${(rows / n).toFixed(0)} rows, ${(views / n).toFixed(2)} views`);
  }
}
