"""Trains the learned verifier (kinlogic/verifier.py) on keyed exam-style questions that are not evaluation questions
(data/symgen): each reading's features, labelled by whether its option is the key. Reports held-out question accuracy
of the plain vote vs the verifier vote.

    python train_verifier.py --ckpt best.pt --n 1500 --out runs/verifier.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import random


def keyed(n, seed=0):
    from eval.leak import Guard
    from kinlogic.prepare import prepare
    rows = []
    for f in sorted(glob.glob('data/symgen/part-k*.jsonl.gz')):
        try:
            for line in gzip.open(f, 'rt'):
                r = json.loads(line)
                if isinstance(r.get('options'), list) and isinstance(r.get('answer'), int) and 0 <= r['answer'] < len(r['options']) <= 5:
                    rows.append(r)
        except (EOFError, OSError):
            pass   # a part still being written
    random.Random(seed).shuffle(rows)
    guard, out = Guard(), []
    for r in rows:
        text = r['question'] + ' ' + ' '.join(f'({l}) {t}' for l, t in zip('abcde', r['options']))
        if not guard.leaks(prepare(text)['tokens']):
            out.append((text, 'abcde'[r['answer']]))
        if len(out) >= n:
            break
    return out


def main():
    import torch
    from eval.evaluate import load
    from kinlogic.exam import candidates, with_exact_clauses
    from kinlogic.verifier import Verifier, features, fit
    from reader import generate
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--split_search', action='store_true', help='train on split-search joins too, as run_real uses them')
    ap.add_argument('--focus_search', action='store_true', help='also join sentence reads made in the whole question (exam.with_split_search focus=True), as run_real --focus_search')
    ap.add_argument('--out', default='runs/verifier.json')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab = load(a.ckpt, dev)
    from eval.batched import Rounds
    R = Rounds(lambda b, **kw: generate(model, vocab, b, dev, **kw), rows=256 if dev.type == 'cuda' else 64)   # batched across questions
    inner = lambda b: [x + y for x, y in zip(R(b, k=2 * a.k), R(b, k=2 * a.k, sample=True, temperature=1.0))]
    if a.split_search or a.focus_search:   # joined readings get weights from what the verifier saw (run_real's order)
        from kinlogic.exam import with_split_search
        from reader import FOCUS_K, focus_fits, with_focus
        base, fits = (lambda b: R(b, k=2 * a.k)), focus_fits(model, vocab)
        if a.split_search:
            inner = with_split_search(inner, fits=fits, base=base)
        if a.focus_search:
            inner = with_focus(inner, model, vocab, dev, base=lambda b: R(b, k=FOCUS_K))
    read = with_exact_clauses(inner)

    def safe(text):
        try:
            return candidates(text, read)
        except Exception:   # a question the pipeline cannot answer is left out, as before
            return None

    items = list(keyed(a.n))
    got = R.answer_all(safe, [t for t, _ in items])
    data = []
    for i, ((text, gold), done) in enumerate(zip(items, got)):
        if done is None:
            continue
        p, results = done
        good = [(r, sc) for r, sc in results if r['status'] not in ('error', 'contradiction', 'none')]
        if not good:
            continue
        top = max(sc for _, sc in good)
        cnt = collections.Counter(r['option'] for r, _ in good)
        data.append([(features(p, r, sc, top, cnt[r['option']] / len(good), j, len(good)), r['option'] == gold, r['option'], sc - top)
                     for j, (r, sc) in enumerate(good)])
        if i % 200 == 0:
            print(f'{i} questions read', flush=True)
    cut = int(len(data) * 0.8)
    w = fit([x for q in data[:cut] for x, *_ in q], [float(y) for q in data[:cut] for _, y, *_ in q])
    v = Verifier(w)

    def acc(qs, weigh):
        ok = 0
        for q in qs:
            votes = collections.Counter()
            for x, y, opt, d in q:
                votes[opt] += weigh(x, d)
            best = max(votes, key=votes.get)
            ok += any(y for x, y, opt, d in q if opt == best)
        return ok / max(1, len(qs))
    rep = {'questions': len(data), 'train': cut, 'heldout_vote': acc(data[cut:], lambda x, d: math.exp(8 * d)),
           'heldout_verifier': acc(data[cut:], lambda x, d: v.prob(x)), 'w': w}
    json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in rep.items() if k != 'w'}))


if __name__ == '__main__':
    main()
