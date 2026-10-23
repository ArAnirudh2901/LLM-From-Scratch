"""Where real questions are lost: reading or selection. For each fair dev question: is the pipeline right, and would any
of the reader's readings (beam, optional samples) give the keyed option? A large gap between the two means better
selection (verifier reranking, LEVER-style) pays; a small one means the reader never writes the right facts.

    python -m eval.oracle_real --ckpt runs/v10-r3-reworded/final.pt --k 8 --samples 16
"""
from __future__ import annotations

import argparse
import collections
import json


def main():
    import torch
    from eval.evaluate import load
    from eval.real_clean import broken, category
    from eval.run_real import as_text, correct, load_real, split_of
    from kinlogic.codes import _fits, read as read_code
    from kinlogic.exam import _answer_one, answer
    from kinlogic.prepare import prepare
    from kinlogic.rules import read as read_rules
    from kinlogic.speech import read as read_speech
    from reader import generate
    from train import get_device
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--samples', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = get_device()
    model, vocab = load(a.ckpt, dev)
    rows = [r for r in load_real() if split_of(r) == a.split and not broken(r)][:a.limit]
    read = lambda b: generate(model, vocab, b, dev, k=a.k)
    by = collections.defaultdict(collections.Counter)
    detail = []
    for i, r in enumerate(rows):
        text, cat = as_text(r), category(r)
        p = prepare(text)
        got = correct(r, answer(text, read))
        exact = read_code(p['tokens'], p['options'], p['ents']) or _fits(read_rules(p['tokens']) or read_speech(p['tokens']))
        if exact:
            kind, hit, n = 'exact', got, 1
        else:
            with torch.no_grad():
                cands = [lf for lf, _ in generate(model, vocab, [p['tokens']], dev, k=a.k)[0]]
                if a.samples:
                    cands += [lf for lf, _ in generate(model, vocab, [p['tokens']], dev, k=a.samples, sample=True, temperature=1.0)[0]]
            cands = list(dict.fromkeys(cands))
            hits = []
            for lf in cands:
                try:
                    hits.append(correct(r, _answer_one(p, lf, lambda b: [''] * len(b), None)))
                except Exception:
                    hits.append(False)
            kind, hit, n = 'model', any(hits), sum(hits)
        c = by[f'{cat} ({kind})']
        c['n'] += 1
        c['right'] += got
        c['any reading right'] += hit or got
        detail.append({'cat': cat, 'kind': kind, 'right': got, 'oracle': bool(hit or got), 'readings_right': n})
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(rows)}', flush=True)
    tot = collections.Counter()
    for c in by.values():
        tot.update(c)
    out = {k: f"right {c['right']}/{c['n']} | some reading right {c['any reading right']}/{c['n']}" for k, c in sorted(by.items())}
    out['ALL'] = f"right {tot['right']}/{tot['n']} = {tot['right'] / tot['n']:.1%} | some reading right {tot['any reading right']}/{tot['n']} = {tot['any reading right'] / tot['n']:.1%}"
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump({'summary': out, 'detail': detail}, open(a.out, 'w'))


if __name__ == '__main__':
    main()
