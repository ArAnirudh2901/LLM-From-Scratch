"""Scores a reader on scraped real exam questions (data/real_questions, private): right option, or answer text.

    python -m eval.run_real --ckpt runs/reader-v9/best.pt [--out results.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
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


def run(read, rows, read_more=None, threshold=0.6, fix=False, answer_fn=None, tracer=None, split=split_of):
    from eval.real_clean import broken, category
    from eval.verify_keys import OUT as KEYS, qhash
    keys = {}
    for f in sorted(glob.glob(KEYS.replace('.json', '*.json'))):   # checked answer keys (eval.verify_keys, all shards)
        keys.update(json.load(open(f)))
    by = collections.defaultdict(lambda: [0, 0])
    sp = collections.defaultdict(lambda: [0, 0])
    misses, oks, fairs, ids = [], [], [], []
    solve = lambda text: answer_fn(text) if answer_fn else answer_corrected(text, read, threshold) if fix else answer_adaptive(text, read, read_more, threshold) if read_more else answer(text, read)
    for i, r in enumerate(rows):
        text = as_text(r)
        if tracer:
            tracer.begin(text)
        res = solve(text)
        ok = correct(r, res)
        oks.append(ok)
        fairs.append(not broken(r))
        ids.append(qhash(r))
        for k in (split(r), f'{split(r)} fair' if not broken(r) else None):
            if k:
                sp[k][0] += ok
                sp[k][1] += 1
        v = keys.get(qhash(r)) if not broken(r) else None
        if v and v['status'] in ('confirmed', 'corrected'):   # wrong keys fixed, ambiguous questions left out
            sp[f'{split(r)} verified'][0] += res.get('option') == v['key']
            sp[f'{split(r)} verified'][1] += 1
        if tracer:
            tracer.end(r, res, ok, solve, i=i, split=split(r), site=r['site'], category=category(r), fair=not broken(r),
                       verified_key=v['key'] if v and v['status'] in ('confirmed', 'corrected') else None)
        k = by[r['site']]
        k[0] += ok; k[1] += 1
        if not ok:
            misses.append({'split': split(r), 'site': r['site'], 'q': r['question'][:300], 'want': r.get('answer') or r.get('answer_text'),
                           'got': res.get('option') or res.get('answer'), 'facts': res.get('facts', '')})
    total = [sum(v[i] for v in by.values()) for i in range(2)]
    return {'real_acc': total[0] / max(1, total[1]), 'n': total[1],
            'splits': {k: f'{a}/{b} = {a / b:.1%}' for k, (a, b) in sorted(sp.items())},
            'by_site': {k: f'{a}/{b}' for k, (a, b) in sorted(by.items())}, 'ok': oks, 'fair': fairs, 'ids': ids}, misses


if __name__ == '__main__':
    import torch
    from eval.evaluate import load
    from reader import generate
    from train import get_device
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ensemble', default='', help='more checkpoints, comma-separated: their readings join the vote')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=None, help='first N questions only (quick comparisons)')
    ap.add_argument('--split', default=None, help="only 'dev' or 'test' questions, or 'locked2' (eval.locked: scored once)")
    ap.add_argument('--rescore_reason', default=None, help='locked2 again: why (a new model selected on dev)')
    ap.add_argument('--pattern', default='data/real_questions/*.jsonl', help='question files (data/fresh_online/*.jsonl: questions scraped after training)')
    ap.add_argument('--k', type=int, default=8, help='readings per question (beam width; the engine votes)')
    ap.add_argument('--adaptive', type=float, default=None, help='re-read with beam 2k + 2k samples when the vote share is below this')
    ap.add_argument('--check', action='store_true', help='execution-guided beam: readings no family fits do not count toward k')
    ap.add_argument('--loop_noise', type=float, default=0.0, help='looped reader: relative noise at each later loop start (sampled readings)')
    ap.add_argument('--simplify', action='store_true', help='questions with relative clauses also read with the clauses split into plain sentences (exam.simplify)')
    ap.add_argument('--hybrid', action='store_true', help='model readings also merged with the exact facts of the clauses the rules understand (exam.with_exact_clauses)')
    ap.add_argument('--by_sentence', action='store_true', help='long questions also read sentence by sentence (exam.with_split), one more vote')
    ap.add_argument('--para', type=int, default=0, help='vote over N extra copies of each question (renamed people, new code symbols, options in place)')
    ap.add_argument('--split_search', action='store_true', help='long questions: sentence-level beam search checked by the engine (exam.with_split_search)')
    ap.add_argument('--verifier', default=None, help='weights of the learned verifier (train_verifier.py); weights the vote')
    ap.add_argument('--cover', action='store_true', help='readings that leave a named person out weigh less in the vote (KIN_COVER)')
    ap.add_argument('--perturb', type=int, default=0, help='also score N altered copies of each question (eval.perturb: shuffled options, renamed people, new code symbols)')
    ap.add_argument('--reworded', default=None, help='also score kept reworded copies (eval.reword_real) against their originals')
    ap.add_argument('--loops', type=int, default=None, help='looped reader: most loops to run (default: all it was trained with)')
    ap.add_argument('--halt', action='store_true', help='looped reader: stop early when the engine accepts the reading and the exit gate agrees (reader.answer_looped)')
    ap.add_argument('--correct', action='store_true', help='re-read and correct unsure readings (exam.answer_corrected; needs a reader trained on correction rows)')
    ap.add_argument('--trace', default=None, help='per-question trace (JSONL: route, every reading, engine results) for eval.trace_report')
    ap.add_argument('--trace_shadow', action='store_true', help='with --trace: also answer exactly-read questions with the exact readers off (the model alone)')
    ap.add_argument('--max_new', type=int, default=96, help='reader output budget in tokens (long stories need up to ~150)')
    ap.add_argument('--reserve', type=int, default=0, help='within this many tokens of the budget, finish the statement and write the question (0: off)')
    ap.add_argument('--closure', action='store_true', help="'not related' with unnamed relatives: retry with everyone named in one family")
    ap.add_argument('--core_repair', action='store_true', help='every reading contradicts: answer without each fact of a minimal conflicting set')
    ap.add_argument('--lever', action='store_true', help='verifier probability times the reader weight (LEVER) instead of replacing it')
    ap.add_argument('--trace_pow', type=float, default=0.0, help='vote weight times reading traceability ** this (0: off)')
    a = ap.parse_args()
    dev = get_device(force_cpu=a.cpu)
    model, vocab = load(a.ckpt, dev)
    model.loop_noise = a.loop_noise   # looped readers: different readings from noise at each later loop
    if a.loops and getattr(model, 'n_loop', 1) > 1:
        model.n_loop = a.loops   # answer_looped and generate stop here; beyond the trained depth repeats the last loop
        model.set_loops(model.n_loop)
    import sys
    from eval.locked import check_once, hashes, record
    from eval.verify_keys import qhash
    locked = hashes()   # locked2 questions: never in a dev or v1 run
    if a.split == 'locked2':
        check_once(a.rescore_reason, a.limit)
        rows = [r for r in load_real(a.pattern) if qhash(r) in locked]
        if not rows or len(rows) != len(locked):   # never spend the one locked score on a partial or empty run
            sys.exit(f'locked2: {len(rows)} of {len(locked)} locked questions found (manifest missing or wrong --pattern); nothing scored')
    else:
        rows = [r for r in load_real(a.pattern) if (not a.split or split_of(r) == a.split) and qhash(r) not in locked][:a.limit]
    from kinlogic.codes import _fits
    chk = (lambda lf: _fits(lf) is not None) if a.check else None
    gen = lambda b, m=model, v=vocab, **kw: generate(m, v, b, dev, max_new=a.max_new, reserve=a.reserve, **kw)
    more = (lambda b: [x + y for x, y in zip(gen(b, k=2 * a.k, check=chk), gen(b, k=2 * a.k, sample=True, temperature=1.0))]) if a.adaptive else None
    read = lambda b: gen(b, k=a.k, check=chk)
    if a.ensemble:   # more readers vote through the same engine: each adds its k readings
        others = [load(c, dev) for c in a.ensemble.split(',')]
        one = read
        read = lambda b: [sum(z, []) for z in zip(one(b), *(gen(b, m=m, v=v, k=a.k, check=chk) for m, v in others))]
    if a.by_sentence:
        from kinlogic.exam import with_split
        read = with_split(read)
    if a.split_search:
        from kinlogic.exam import with_split_search
        read = with_split_search(read)
    if a.simplify:
        from kinlogic.exam import with_simplified
        read = with_simplified(read)
    if a.hybrid:
        from kinlogic.exam import with_exact_clauses
        read = with_exact_clauses(read)
        more = with_exact_clauses(more) if more else None
    from reader import answer_looped
    fn = (lambda text: answer_looped(text, model, vocab, dev, k=a.k)) if a.halt else None
    if a.verifier:
        os.environ['KIN_VERIFIER'] = a.verifier
    import kinlogic.exam as E
    E.COVER = E.COVER or a.cover
    E.CLOSURE, E.CORE_REPAIR, E.LEVER, E.TRACE_POW = a.closure, a.core_repair, a.lever, a.trace_pow
    if a.para:   # self-consistency over surface forms: the same question with other names and code symbols
        import random
        import zlib
        from eval.perturb import held_out, variant
        names, by_text = held_out(), {as_text(r): r for r in rows}
        base = fn or (lambda text: answer_adaptive(text, read, more, a.adaptive or 0.6) if more else answer(text, read))

        def fn(text):
            outs, r = [base(text)], by_text.get(text)
            rng = random.Random(zlib.crc32(text.encode()))
            for _ in range(a.para if r else 0):
                try:
                    outs.append(base(as_text(variant(r, rng, names, shuffle=False))))
                except Exception:
                    pass
            votes = collections.Counter()
            for o in outs:
                if o.get('option'):
                    votes[o['option']] += 1.0 + o.get('confidence', 0.0)
            if not votes:
                return outs[0]
            best = votes.most_common(1)[0][0]
            return next(o for o in outs if o.get('option') == best)
    tracer = None
    if a.trace:
        from eval.trace import Tracer
        tracer = Tracer(a.trace, shadow=a.trace_shadow).install()
    score, misses = run(read, rows, more, a.adaptive or 0.6, fix=a.correct, answer_fn=fn, tracer=tracer,
                        split=(lambda r: 'locked2') if a.split == 'locked2' else split_of)
    if a.split == 'locked2':
        record(score, a.ckpt, sys.argv[1:], a.rescore_reason)
    if tracer:
        tracer.close()   # perturbed / reworded copies below are scored untraced
    print(json.dumps({k: v for k, v in score.items() if k not in ('ok', 'fair', 'ids')}, indent=1))
    if a.perturb:
        from eval.perturb import compare, variants
        from eval.real_clean import broken
        pairs = variants(rows, a.perturb)
        s1, _ = run(read, [v for _, v in pairs], more, a.adaptive or 0.6, fix=a.correct, answer_fn=fn)
        score['perturbed'] = compare(rows, score['ok'], pairs, s1['ok'], split_of, broken)
        print(json.dumps(score['perturbed'], indent=1))
    if a.reworded and os.path.exists(a.reworded):
        from eval.reword_real import compare as cmp_rw, pairs_of
        orig, copies = pairs_of(a.reworded, rows)
        ok0 = [score['ok'][rows.index(r)] for r in orig]
        s2, _ = run(read, copies, more, a.adaptive or 0.6, fix=a.correct, answer_fn=fn)
        score['reworded'] = cmp_rw(ok0, s2['ok'])
        print(json.dumps(score['reworded'], indent=1))
    if a.out:
        json.dump({'score': score, 'misses': misses}, open(a.out, 'w'), indent=1)
