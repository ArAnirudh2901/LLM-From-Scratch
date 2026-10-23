"""Re-read-and-correct rows (self-correction learned from the reader's own mistakes, checked by an exact verifier:
STaSC 2503.08681; "small LMs need strong verifiers", 2404.17140). A reader drafts facts for questions whose true facts
are known (verified pool, verified rewrites, STaR/teacher rows); the engine judges each draft the way it can at test
time ('contradiction': no family fits; 'unsure': the draft's answer has under 60% of the drafts' vote). A row is
question + "draft : <facts> check : <verdict>" -> the true facts. Wrong drafts teach fixing; some right drafts with
'unsure' teach keeping a good reading. exam.answer_corrected uses the same format at test time.

    python build_corrections.py --ckpt runs/v10-r4-pre/best_12000.pt --out data/corrections.jsonl.gz --n 30000
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random

import torch

from kinlogic.exam import with_draft


def verdict_of(lf: str, people, share: float) -> str | None:
    from kinlogic.engine import solve
    from kinlogic.facts import parse
    st, bad = parse(lf)
    if bad:
        return 'contradiction'
    r = solve(st, people=people)
    if not r['ok'] or not (r['questions'] or r.get('claims')):
        return 'contradiction'
    return 'unsure' if share < 0.6 else None   # None: the draft would be kept without a second look


def sources(a, rng):
    rows = []
    for f in sorted(glob.glob(f'{a.pool}/kinds/*.jsonl.gz')):
        got = [json.loads(l) for _, l in zip(range(a.per_kind * 5), gzip.open(f, 'rt'))]
        rows += rng.sample(got, min(a.per_kind, len(got)))
    for f in a.extra:
        for p in glob.glob(f):
            rows += [json.loads(l) for l in gzip.open(p, 'rt')]
    rng.shuffle(rows)
    return [r for r in rows if r.get('lf') and r.get('tokens')]


def main():
    from eval.evaluate import load
    from reader import generate
    from train_reader import answers
    from kinlogic.symgen import real_keys
    import hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pool', default='data/dataset_v6')
    ap.add_argument('--per_kind', type=int, default=3000)
    ap.add_argument('--extra', nargs='*', default=['data/dataset_v6/reworded.jsonl.gz', 'data/star_r2.jsonl.gz', 'data/teach_rows.jsonl.gz'])
    ap.add_argument('--n', type=int, default=30000, help='rows wanted')
    ap.add_argument('--samples', type=int, default=6)
    ap.add_argument('--keep_share', type=float, default=0.25, help='right drafts kept as rows (learn not to over-edit)')
    ap.add_argument('--bs', type=int, default=64)
    a = ap.parse_args()
    rng = random.Random(0)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model, vocab = load(a.ckpt, dev)
    blocked = real_keys()
    rows = [r for r in sources(a, rng) if hashlib.sha1(' '.join(r['tokens']).encode()).hexdigest() not in blocked]
    st, out = collections.Counter(), []
    for i in range(0, len(rows), a.bs):
        if len(out) >= a.n:
            break
        chunk = rows[i:i + a.bs]
        toks = [r['tokens'] for r in chunk]
        with torch.no_grad():
            greedy = generate(model, vocab, toks, dev, k=1)
            drawn = generate(model, vocab, toks, dev, k=a.samples, sample=True, temperature=1.0)
        for r, g, s in zip(chunk, greedy, drawn):
            people = sorted({t for t in r['tokens'] if t[0] == '@'})
            gold = answers(r['lf'], people)
            drafts = [lf for lf, _ in list(g) + list(s)]
            got = [answers(lf, people) for lf in drafts]
            votes = collections.Counter(str(x) for x in got)
            seen = set()
            for lf, ans in zip(drafts, got):
                if lf in seen:
                    continue
                seen.add(lf)
                v = verdict_of(lf, people, votes[str(ans)] / len(got))
                right = ans == gold
                if v is None or (right and rng.random() > a.keep_share):
                    st['kept as is' if v is None else 'right, skipped'] += 1
                    continue
                out.append({'kind': 'correct-' + ('keep' if right else v), 'tokens': with_draft(r['tokens'], lf, v), 'lf': r['lf']})
                st['right (keep row)' if right else f'wrong ({v})'] += 1
        print(f'{min(i + a.bs, len(rows))}/{len(rows)} | rows {len(out)} | {dict(st)}', flush=True)
    with gzip.open(a.out, 'wt') as f:
        f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out[:a.n]))
    print('done', len(out[:a.n]), dict(st))


if __name__ == '__main__':
    main()
