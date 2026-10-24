"""Every candidate reading of each question the model must read (beam + samples), with the engine's answer and
whether it reaches the key: lets selection rules (votes, coverage checks, verifiers) be compared offline without
running the reader again. Tune selection on keyed TRAINING questions (symgen); check on real dev once.

    python -m eval.dump_readings --ckpt runs/v10-r5-plain/best.pt --src symgen --n 800 --out runs/readings_symgen.jsonl
    python -m eval.dump_readings --ckpt runs/v10-r5-plain/best.pt --src dev --out runs/readings_dev.jsonl
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import random


def main():
    import torch
    from eval.evaluate import load
    from eval.real_clean import broken, category
    from eval.run_real import as_text, correct, load_real, split_of
    from kinlogic.codes import _fits, read as read_code
    from kinlogic.exam import _answer_one
    from kinlogic.prepare import prepare
    from kinlogic.rules import read as read_rules
    from kinlogic.speech import read as read_speech
    from reader import generate
    from train import get_device
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--src', default='symgen', help="'symgen' (keyed training questions) or 'dev' / 'test' (real)")
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = get_device()
    model, vocab = load(a.ckpt, dev)
    if a.src == 'symgen':
        rows = []
        for f in sorted(glob.glob('data/symgen/*.jsonl.gz')):
            try:
                for l in gzip.open(f, 'rt'):
                    q = json.loads(l)
                    rows.append({'question': q['question'], 'options': [[x, t] for x, t in zip('abcde', q['options'])], 'answer': 'abcde'[int(q['answer'])]})
            except EOFError:
                pass
        random.Random(0).shuffle(rows)
    else:
        rows = [r for r in load_real() if split_of(r) == a.src and not broken(r)]
    out, n = open(a.out, 'w'), 0
    for r in rows:
        if n >= a.n:
            break
        p = prepare(as_text(r))
        try:
            if read_code(p['tokens'], p['options'], p['ents']) or _fits(read_rules(p['tokens']) or read_speech(p['tokens'])):
                continue   # exact readers answer these; selection matters only where the model reads
        except Exception:
            pass
        with torch.no_grad():
            cands = [(lf, sc, 'beam') for lf, sc in generate(model, vocab, [p['tokens']], dev, k=a.k)[0]]
            cands += [(lf, sc, 'sample') for lf, sc in generate(model, vocab, [p['tokens']], dev, k=a.samples, sample=True, temperature=1.0)[0]]
        got = []
        for lf, sc, how in cands:
            try:
                res = _answer_one(p, lf, lambda b: [''] * len(b), None)
                opt, st = res.get('option'), res.get('status')
            except Exception:
                res, opt, st = {}, None, 'error'
            got.append({'lf': lf, 'score': sc, 'how': how, 'option': opt, 'status': st, 'right': bool(correct(r, res)) if opt else False})
        out.write(json.dumps({'cat': category(r), 'tokens': p['tokens'], 'key': r['answer'], 'readings': got}, ensure_ascii=False) + '\n')
        out.flush()
        n += 1
        if n % 50 == 0:
            print(f'{n}/{a.n}', flush=True)
    print('done', n)


if __name__ == '__main__':
    main()
