"""Rewording at scale with an open model on a free Colab GPU (no API quotas): the same prompts and the same checks as
kinlogic/reword.py (names kept, plain text, an independent reading of the new text, the engine's answer and two-way
meaning), batched. Default model: Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0), fp16 on a T4 via vLLM; falls back to
transformers when vLLM is missing.

    python -m kinlogic.reword_local --src data/dataset_v2 --out /content/drive/MyDrive/kinthread/reworded_local.jsonl.gz --n 20000
Works in chunks; every kept row is its own gzip block, so a session end loses at most one chunk; resumable.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor

from .kin import KIN
from .prepare import prepare
from .reword import EXTRACT, REWRITE, SKIP, accept, clean, needed, same_meaning, to_facts

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
BAD = re.compile(r'\\[a-z]+|\$[A-Za-z]|\*\*|^\s*[-*] ', re.M)


class Backend:
    """chat(prompts, temperature, max_tokens) -> texts, batched."""

    def __init__(self, model: str):
        try:
            from vllm import LLM, SamplingParams
            self.sp = SamplingParams
            self.llm = LLM(model=model, dtype='float16', max_model_len=4096, gpu_memory_utilization=0.90,
                           enable_prefix_caching=True)   # every prompt shares a long fixed preamble
            self.kind = 'vllm'
        except Exception as e:   # not installed, or this vLLM does not run on the GPU (T4 is compute 7.5)
            print('vLLM unavailable:', type(e).__name__, str(e)[:120], '-> transformers', flush=True)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map='auto')
            self.kind = 'transformers'
        print('backend', self.kind, model, flush=True)

    def chat(self, prompts, temperature, max_tokens):
        msgs = [[{'role': 'user', 'content': p}] for p in prompts]
        if self.kind == 'vllm':
            outs = self.llm.chat(msgs, self.sp(temperature=temperature, top_p=0.95 if temperature else 1.0, max_tokens=max_tokens),
                                 use_tqdm=False)
            return [o.outputs[0].text.strip() for o in outs]
        import torch
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        out = []
        for i in range(0, len(texts), 16):
            enc = self.tok(texts[i:i + 16], return_tensors='pt', padding=True).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                          temperature=temperature or None, top_p=0.95 if temperature else None)
            out += [self.tok.decode(g[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip() for g in gen]
        return out


def sample_sources(src, n, seed=0, max_depth=8):
    """Rows from every file (all kinds and depths) except the locked test, parsed lazily."""
    files = [f for f in sorted(glob.glob(f'{src}/**/*.jsonl.gz', recursive=True)) if '/test/' not in f]
    rng = random.Random(seed)
    per = max(5, 3 * n // max(1, len(files)))
    rows = []
    for f in files:
        lines = gzip.open(f, 'rt').readlines()
        rows += [json.loads(l) for l in rng.sample(lines, min(per, len(lines)))]
    rows = [r for r in rows if isinstance(r.get('truth'), str) and "'s " not in r['truth'] and r.get('names')
            and (r.get('depth') or 0) <= max_depth and r['kind'] not in SKIP]   # a 4B reader misreads very long chains
    rng.shuffle(rows)
    return rows


def _verify(job):
    """CPU side of one candidate: (training row or None, reason)."""
    ex, new, raw = job
    try:
        p = prepare(new)
        m = re.search(r'\{.*\}', raw, re.S)
        if not m:
            return None, 'no json'
        try:
            lf = to_facts(json.loads(m.group(0)), p['ents'])
        except (ValueError, TypeError, AttributeError):
            return None, 'bad json'
        if lf is None:
            return None, 'unmapped fact'
        if not accept(lf, ex['truth'], p['tokens']):
            return None, 'wrong answer'
        if not same_meaning(ex, lf, p['ents']):
            return None, 'meaning differs'
        return {'kind': 'reworded-' + ex['kind'], 'text': new, 'tokens': p['tokens'], 'lf': lf, 'truth': ex['truth'],
                'depth': ex.get('depth'), 'source': hashlib.sha1(ex['text'].encode()).hexdigest()}, 'kept'
    except Exception as e:
        return None, 'error ' + type(e).__name__


def done_sources(out):
    done = set()
    if os.path.exists(out):
        try:
            for l in gzip.open(out, 'rt'):
                done.add(json.loads(l)['source'])
        except (EOFError, OSError, ValueError):
            pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/dataset_v2')
    ap.add_argument('--out', default='data/reworded_local.jsonl.gz')
    ap.add_argument('--n', type=int, default=20000, help='kept rewordings wanted (counting earlier runs)')
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args()
    done = done_sources(a.out)
    rows = [r for r in sample_sources(a.src, a.n, seed=int(time.time())) if hashlib.sha1(r['text'].encode()).hexdigest() not in done]   # new sample per run
    be = Backend(a.model)
    kept, tried, t0 = len(done), 0, time.time()
    words = ', '.join(sorted(KIN))
    pending = None   # verification of the previous chunk runs on the CPUs while the GPU writes the next one

    def finish(job):
        nonlocal kept, tried
        batch, ok, raws, why, eg, futs = job
        got = []
        for ((ex, new), raw), fut in zip(zip(ok, raws), futs):
            g, r = fut.result()
            why[r] = why.get(r, 0) + 1
            if g:
                got.append(g)
            elif r not in eg:
                eg[r] = f'{new[:100]} => {raw[:140]}'.replace('\n', ' ')
        with gzip.open(a.out, 'at') as f:   # one complete block per chunk
            for g in got:
                f.write(json.dumps(g, ensure_ascii=False) + '\n')
        tried += len(batch)
        kept += len(got)
        rate = tried / max(1e-9, time.time() - t0) * 3600
        print(f'{tried} tried, {kept} kept in total ({len(got)}/{len(batch)} this chunk), {rate:.0f} tried/hour | '
              + ', '.join(f'{k} {v}' for k, v in sorted(why.items(), key=lambda kv: -kv[1])), flush=True)
        for k, v in eg.items():
            print(f'   e.g. {k}: {v}', flush=True)

    with ProcessPoolExecutor(a.workers) as pool:
        for i in range(0, len(rows), a.chunk):
            if kept >= a.n:
                break
            batch = rows[i:i + a.chunk]
            news = [clean(t) for t in be.chat([REWRITE.format(q=ex['text']) for ex in batch], 0.7, 300)]
            ok, why, eg = [], {}, {}
            for ex, new in zip(batch, news):
                names = needed(ex)
                r = 'latex/markdown' if not new or BAD.search(new) else 'names lost' if not all(nm.lower() in new.lower() for nm in names) else None
                if r:
                    why[r] = why.get(r, 0) + 1
                    eg.setdefault(r, new[:120])
                else:
                    ok.append((ex, new))
            raws = be.chat([EXTRACT.format(q=new, words=words) for _, new in ok], 0.0, 400) if ok else []
            futs = [pool.submit(_verify, (ex, new, raw)) for (ex, new), raw in zip(ok, raws)]
            if pending:
                finish(pending)
            pending = (batch, ok, raws, why, eg, futs)
        if pending:
            finish(pending)
    print(f'done: {kept} kept in {a.out}')


if __name__ == '__main__':
    main()
