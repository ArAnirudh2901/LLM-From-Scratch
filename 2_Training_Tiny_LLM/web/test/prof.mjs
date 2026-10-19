// Time per question for an engine build (argv[2] = module path), 100 sweep questions, default settings.
import { readFile } from 'node:fs/promises';
import { KinshipQA } from '../kinship.js';
const here = new URL('.', import.meta.url);
const qa = await KinshipQA.load({ base: '../model/', fetchFn: async (u) => new Response(await readFile(new URL(u, here))), engineModule: process.argv[2], maxThreads: +(process.argv[3] || 8) });
const qs = Object.values(JSON.parse(await readFile(new URL('golden.json', here))).sweep).flatMap((x) => x.slice(0, 20));
qa.predict(qs[0]);
const t = performance.now();
for (const g of qs) qa.predict(g);
console.log(`${process.argv[2] || 'engine.mjs'} threads ${qa.threads}: ${((performance.now() - t) / qs.length).toFixed(1)} ms/question`);
