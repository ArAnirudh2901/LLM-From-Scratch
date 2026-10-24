"""Scores a reader on scraped real exam questions (data/real_questions, private): right option, or answer text.

    python -m eval.run_real --ckpt runs/reader-v9/best.pt [--out results.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re

from kinlogic.exam import _relation_option, answer, answer_adaptive, answer_corrected, option_moves
from eval.sites import complete
from kinlogic.explain import match_option


def load_real(pattern='data/real_questions/*.jsonl'):
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            if r.get('url', '').lower().endswith('.pdf'):   # page breaks scramble PDF questions
                continue
            r['options'] = [[l, re.sub(r'^\s*\(?[A-Ea-e1-5][).]\s*\.?\s*', '', t)] for l, t in r.get('options') or []]
            if r['options'] and all(l.isdigit() for l, _ in r['options']):   # "1) ... 5)" with the answer as a letter
                r['options'] = [['abcde'[int(l) - 1], t] for l, t in r['options'] if 1 <= int(l) <= 5]
                if r.get('answer') and r['answer'].isdigit():
                    r['answer'] = 'abcde'[int(r['answer']) - 1]
            key = re.sub(r'[^a-z0-9]+', ' ', (r['question'] + ' ' + ' '.join(t for _, t in r['options'])).lower()).strip()
            if key in seen or complete(r):   # once across sites; only questions answerable from their own text
                continue
            seen.add(key)
            rows.append(r)
    return rows


def split_of(r) -> str:
    """'dev' (error analysis, fixes) or 'test' (locked: scores only), half each by question hash."""
    import zlib
    return 'dev' if zlib.crc32(re.sub(r'\s+', ' ', r['question']).strip().lower().encode()) % 2 == 0 else 'test'


def as_text(r):
    opts = ' '.join(f'({l}) {t}' for l, t in r['options'])
    return f"{r['question']} {opts}".strip()


SYNONYMS = [(r'\bgrand[- ]?(uncle|aunt|nephew|niece)\b', r'great-\1'), (r'\bgrand (father|mother|son|daughter)\b', r'grand\1'),
            (r'\bspouse\b', 'husband or wife')]


def _same_words(t):
    t = t.lower().strip().rstrip('.')
    t = re.sub(r'^(.+?)\s*\((paternal|maternal)\)$', r'\2 \1', t)   # "cousins (paternal)"
    t = re.sub(r'\s*\([^)]*\)', '', t).strip()   # glosses: "maternal uncle (mausa)", "(by marriage)"
    t = re.sub(r'^[a-z]\w* is (?:the |his |her )?(.+?) (?:of|to) [a-z]\w*$', r'\1', t)   # "p is father of q"
    t = re.sub(r'\b(cousin|brother|sister|sibling)s$', r'\1', t)
    for a, b in SYNONYMS:
        t = re.sub(a, b, t)
    return t


def correct(r, res):
    if r.get('answer') and r['options']:
        return res['option'] == r['answer']
    want = r.get('answer_text') or ''
    got = res.get('answer') or ''
    if not want or not got:
        return False
    if match_option(got, [('x', want)]) or match_option(want, [('x', got)]) or _same_words(got) == _same_words(want):
        return True
    mv_w, mv_g = option_moves(re.sub(r"^.*?\bis\s+\w+'s\s+", '', want)), option_moves(got)
    return bool(mv_w and mv_g and len(mv_w) == len(mv_g) and all(a == c and (g is None or h is None or g == h) for (a, g), (c, h) in zip(mv_w, mv_g)))


def run(read, rows, read_more=None, threshold=0.6, fix=False, answer_fn=None):
    from eval.real_clean import broken
    by = collections.defaultdict(lambda: [0, 0])
    sp = collections.defaultdict(lambda: [0, 0])
    misses, oks = [], []
    for r in rows:
        text = as_text(r)
        res = answer_fn(text) if answer_fn else answer_corrected(text, read, threshold) if fix else answer_adaptive(text, read, read_more, threshold) if read_more else answer(text, read)
        ok = correct(r, res)
        oks.append(ok)
        for k in (split_of(r), f'{split_of(r)} fair' if not broken(r) else None):
            if k:
                sp[k][0] += ok
                sp[k][1] += 1
        k = by[r['site']]
        k[0] += ok; k[1] += 1
        if not ok:
            misses.append({'split': split_of(r), 'site': r['site'], 'q': r['question'][:300], 'want': r.get('answer') or r.get('answer_text'),
                           'got': res.get('option') or res.get('answer'), 'facts': res.get('facts', '')})
    total = [sum(v[i] for v in by.values()) for i in range(2)]
    return {'real_acc': total[0] / max(1, total[1]), 'n': total[1],
            'splits': {k: f'{a}/{b} = {a / b:.1%}' for k, (a, b) in sorted(sp.items())},
            'by_site': {k: f'{a}/{b}' for k, (a, b) in sorted(by.items())}, 'ok': oks}, misses


if __name__ == '__main__':
    import torch
    from eval.evaluate import load
    from reader import generate
    from train import get_device
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=None, help='first N questions only (quick comparisons)')
    ap.add_argument('--split', default=None, help="only 'dev' or 'test' questions")
    ap.add_argument('--k', type=int, default=8, help='readings per question (beam width; the engine votes)')
    ap.add_argument('--adaptive', type=float, default=None, help='re-read with beam 2k + 2k samples when the vote share is below this')
    ap.add_argument('--check', action='store_true', help='execution-guided beam: readings no family fits do not count toward k')
    ap.add_argument('--loop_noise', type=float, default=0.0, help='looped reader: relative noise at each later loop start (sampled readings)')
    ap.add_argument('--perturb', type=int, default=0, help='also score N altered copies of each question (eval.perturb: shuffled options, renamed people, new code symbols)')
    ap.add_argument('--loops', type=int, default=None, help='looped reader: most loops to run (default: all it was trained with)')
    ap.add_argument('--halt', action='store_true', help='looped reader: stop early when the engine accepts the reading and the exit gate agrees (reader.answer_looped)')
    ap.add_argument('--correct', action='store_true', help='re-read and correct unsure readings (exam.answer_corrected; needs a reader trained on correction rows)')
    a = ap.parse_args()
    dev = get_device(force_cpu=a.cpu)
    model, vocab = load(a.ckpt, dev)
    model.loop_noise = a.loop_noise   # looped readers: different readings from noise at each later loop
    if a.loops and getattr(model, 'n_loop', 1) > 1:
        model.n_loop = a.loops   # answer_looped and generate stop here; beyond the trained depth repeats the last loop
        model.set_loops(model.n_loop)
    rows = [r for r in load_real() if not a.split or split_of(r) == a.split][:a.limit]
    from kinlogic.codes import _fits
    chk = (lambda lf: _fits(lf) is not None) if a.check else None
    more = (lambda b: [x + y for x, y in zip(generate(model, vocab, b, dev, k=2 * a.k, check=chk), generate(model, vocab, b, dev, k=2 * a.k, sample=True, temperature=1.0))]) if a.adaptive else None
    read = lambda b: generate(model, vocab, b, dev, k=a.k, check=chk)
    from reader import answer_looped
    fn = (lambda text: answer_looped(text, model, vocab, dev, k=a.k)) if a.halt else None
    score, misses = run(read, rows, more, a.adaptive or 0.6, fix=a.correct, answer_fn=fn)
    print(json.dumps({k: v for k, v in score.items() if k != 'ok'}, indent=1))
    if a.perturb:
        from eval.perturb import compare, variants
        from eval.real_clean import broken
        pairs = variants(rows, a.perturb)
        s1, _ = run(read, [v for _, v in pairs], more, a.adaptive or 0.6, fix=a.correct, answer_fn=fn)
        score['perturbed'] = compare(rows, score['ok'], pairs, s1['ok'], split_of, broken)
        print(json.dumps(score['perturbed'], indent=1))
    if a.out:
        json.dump({'score': score, 'misses': misses}, open(a.out, 'w'), indent=1)
