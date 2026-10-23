"""Training rows from answer-keyed questions that have no facts (kinlogic/symgen.py): our reader proposes readings
(beam plus samples), the engine answers each, and a reading that reaches the key becomes the label (STaR, Zelikman
et al. 2022; rejection sampling as in Llama 3; RAFT matches GRPO, 2504.11343). Guards against a reading that is right
by luck (4-5 options): it must name every person the question names, the engine must give one answer, and at least
two different readings must reach the key. Questions an exact reader (codes, rules, speech: never wrong where they apply)
reads to the key keep that reading (the exact readers are distilled into the model). Each row records `hit`, the share of
the model's readings that reached the key: low but non-zero is the learnable frontier (GLM-5 difficulty filtering).

    python star.py --ckpt runs/v10-r3-reworded/final.pt --questions data/symgen/*.jsonl.gz --out data/star_r1.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import re

import torch


def main():
    from eval.evaluate import load
    from kinlogic.codes import _fits, read as read_code
    from kinlogic.rules import read as read_rules
    from kinlogic.speech import read as read_speech
    from kinlogic.exam import _answer_one
    from kinlogic.prepare import prepare
    from kinlogic.symgen import as_text
    from reader import generate
    from train import get_device
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--questions', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--bs', type=int, default=16)
    a = ap.parse_args()
    dev = get_device()
    model, vocab = load(a.ckpt, dev)
    qs = [json.loads(l) for p in a.questions for f in glob.glob(p) for l in gzip.open(f, 'rt')]
    stats = collections.Counter()
    out = []
    for i in range(0, len(qs), a.bs):
        chunk = qs[i:i + a.bs]
        preps = [prepare(as_text(q)) for q in chunk]
        toks = [p['tokens'] for p in preps]
        with torch.no_grad():
            beams = generate(model, vocab, toks, dev, k=a.beam)
            samples = generate(model, vocab, toks, dev, k=a.samples, sample=True, temperature=1.0)
        for q, p, b, s in zip(chunk, preps, beams, samples):
            key = 'abcde'[q['answer']]
            exact = read_code(p['tokens'], p['options'], p['ents']) or _fits(read_rules(p['tokens']) or read_speech(p['tokens']))
            named = {t for t in p['tokens'] if t[0] == '@'}
            good = {}
            for lf, sc in list(b) + list(s):
                if lf in good:
                    continue
                try:
                    res = _answer_one(p, lf, lambda batch: [''] * len(batch), None)
                except Exception:
                    continue
                if res.get('option') != key or res.get('status') in ('error', 'contradiction', 'none'):
                    continue
                if named - set(re.findall(r'@\d+', lf)):
                    continue
                good[lf] = sc
            hit = len(good) / max(1, len({lf for lf, _ in list(b) + list(s)}))
            if exact:
                try:
                    ok = _answer_one(p, exact, lambda batch: [''] * len(batch), None).get('option') == key
                except Exception:
                    ok = False
                if ok:
                    out.append({'kind': 'exact-' + q.get('style', 'q'), 'text': as_text(q), 'tokens': p['tokens'], 'lf': exact,
                                'truth': q['options'][q['answer']], 'readings': len(good), 'hit': round(hit, 3)})
                    stats['kept (exact reader)'] += 1
                    continue
            if len(good) < 2:
                stats['no confirmed reading'] += 1
                continue
            lf = max(good, key=good.get)
            out.append({'kind': 'star-' + q.get('style', 'q'), 'text': as_text(q), 'tokens': p['tokens'], 'lf': lf,
                        'truth': q['options'][q['answer']], 'readings': len(good), 'hit': round(hit, 3)})
            stats['kept'] += 1
        print(f"{min(i + a.bs, len(qs))}/{len(qs)} | {dict(stats)}", flush=True)
    with gzip.open(a.out, 'wt') as f:
        f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    print('done', dict(stats))


if __name__ == '__main__':
    main()
