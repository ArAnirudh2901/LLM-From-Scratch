// Per-view results (beam 4, all 4 views) on the labelled sets, to choose the adaptive-view rule offline.
import { readFile, writeFile } from 'node:fs/promises';
import { KinshipQA } from '../kinship.js';
const here = new URL('.', import.meta.url);
const qa = await KinshipQA.load({ base: '../model/', fetchFn: async (u) => new Response(await readFile(new URL(u, here))) });
const golden = JSON.parse(await readFile(new URL('golden.json', here)));
const out = {};
for (const [set, qs] of Object.entries(golden.sweep)) {
  out[set] = qs.map((g) => {
    const t = performance.now(), r = qa.predict(g, { views: 4, adaptive: null, beam: 4 });
    return { gold: g.rel, k: g.k, ms: performance.now() - t, rows: r.rows, perView: r.perView };
  });
  console.log(set, out[set].length);
}
await writeFile(new URL('calib.json', here), JSON.stringify({ relations: qa.meta.relations, unknown: qa.meta.unknown, sets: out }));
