"""Builds data/dataset_v8_extra.jsonl.gz from the verified pool by the rules in dataset_spec.py: answer labels
re-weighted toward real exams with a floor for rare answers, surfaces rewritten into exam constructions, rare meaning
types topped up. Writes data/dataset_v8_manifest.json with every count and the audit against real exams, so a reader
(or a later model) can see what the data holds and what is missing.

    python build_dataset_v8.py --pool data/dataset_v6
Train with: train_reader --pool <dataset_v7> --reworded data/dataset_v8_extra.jsonl.gz,... --p_reworded 0.5 --style_p 0.5
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import re
from multiprocessing import Pool

import dataset_spec as SPEC

BASE = lambda k: re.sub(r'\b(?:paternal|maternal|elder|younger) ', '', k or '')


def _label(r):
    from answer_audit import label
    if not r.get('lf') or 'ask ' not in r['lf']:
        return None
    try:
        return label(r['lf'], sorted({t for t in r['tokens'] if t[0] == '@'}))
    except Exception:
        return None


def real_answer_shares():
    from eval.real_clean import broken
    from eval.run_real import load_real
    c = collections.Counter()
    for r in load_real():   # dev + test: answer counts only
        if broken(r) or not r.get('options') or not r.get('answer'):
            continue
        t = re.sub(r'\s*\([^)]*\)', '', dict(r['options']).get(r['answer'], '').lower()).strip().rstrip('.')
        if t and len(t) <= 30 and not re.search(r'\b(?:is|are|and|only|statement|data|none|cannot|both|either|all|his|her|a|four|three|two)\b', t):
            c[BASE(t)] += 1
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}


def _rewrite(job):
    from boost_phrases import boost, restyle
    from train_reader import answers
    r, seed, how = job
    rng = random.Random(seed)
    toks, lf = r['tokens'], r['lf']
    if how == 'clause':   # exam copula first ("Y is the R of X") opens more sentence pairs to merge into a clause
        pre = restyle(list(toks), lf, rng, married=False)
        base = pre[0] if pre else list(toks)
        new = boost(list(base), rng, p=1.0)
        if new:
            return {**r, 'tokens': new, 'kind': 'v8-clause-' + str(r.get('kind'))}, 'clause'
        if pre:
            return {**r, 'tokens': base, 'kind': 'v8-style-' + str(r.get('kind'))}, 'restyle'
        return r, 'plain'
    if how == 'restyle':
        got = restyle(list(toks), lf, rng, married=True)
        if not got:
            return r, 'plain'
        new, new_lf = got
        if new_lf != lf:
            people = sorted({t for t in toks if t[0] == '@'})
            try:
                if answers(new_lf, people) != answers(lf, people):
                    got = restyle(list(toks), lf, rng, married=False)
                    if not got:
                        return r, 'plain'
                    new, new_lf = got
            except Exception:
                return r, 'plain'
        if rng.random() < 0.5:
            new = boost(list(new), rng) or new
        return {**r, 'tokens': new, 'lf': new_lf, 'kind': 'v8-style-' + str(r.get('kind'))}, 'restyle'
    return r, 'plain'


NAMES = list(SPEC.CONSTRUCTIONS)
WEIGHTS = [v[0] for v in SPEC.CONSTRUCTIONS.values()]


def _assign(j):
    """Construction tried on draw j: rare constructions are tried more often since fewer rows fit them."""
    return random.Random(j).choices(NAMES, weights=[w * (1 if 'plain' in n or 'copula' in n or 'question' in n else 3) for n, w in zip(NAMES, WEIGHTS)])[0]


def _construct(job):
    from boost_phrases import CONSTRUCT
    from train_reader import answers
    name, r, seed = job
    rng = random.Random(seed)
    try:
        got = CONSTRUCT[name](list(r['tokens']), r['lf'], rng)
    except Exception:
        return name, None
    if not got:
        return name, None
    toks, lf = got
    if lf != r['lf']:   # facts changed (married to): the engine must give the same answer
        people = sorted({t for t in r['tokens'] if t[0] == '@'})
        try:
            if answers(lf, people) != answers(r['lf'], people):
                return name, None
        except Exception:
            return name, None
    return name, {**r, 'tokens': toks, 'lf': lf, 'kind': 'v8-' + name.split(' (')[0].replace(' ', '-') + '-' + str(r.get('kind'))}


def main():
    from phrase_audit import PAT, real_texts, shares
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='data/dataset_v6')
    ap.add_argument('--out', default='data/dataset_v8_extra.jsonl.gz')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    rng = random.Random(8)
    files = sorted(glob.glob(f'{a.pool}/kinds/*.jsonl.gz')) + sorted(glob.glob(f'{a.pool}/deep/*.jsonl.gz'))
    rows = []
    for f in files:
        rows += [json.loads(l) for _, l in zip(range(SPEC.LABEL_SAMPLE // len(files) + 1), gzip.open(f, 'rt'))]
    rows = [r for r in rows if isinstance(r.get('tokens'), list) and r.get('lf')]
    with Pool(a.workers) as pool:
        labels = pool.map(_label, rows, chunksize=200)
    train = collections.Counter(BASE(l) for l in labels if l)
    ttot = sum(train.values())
    real = real_answer_shares()
    uni = 1.0 / max(1, len(train))
    weight = {}
    for k, n in train.items():
        target = SPEC.ANSWER_BLEND * real.get(k, 0.0) + (1 - SPEC.ANSWER_BLEND) * uni
        weight[k] = min(SPEC.ANSWER_MAX_WEIGHT, max(SPEC.ANSWER_MIN_WEIGHT, target / (n / ttot)))
    w = [weight.get(BASE(l), 1.0) if l else 1.0 for l in labels]
    picked = rng.choices(range(len(rows)), weights=w, k=SPEC.ROWS)
    order = rng.choices(range(len(rows)), weights=w, k=SPEC.ROWS * 4)   # answer-balanced draws, reused across constructions
    with Pool(a.workers) as pool:
        done = pool.map(_construct, [(name, rows[i], 1_000_000 + j) for j, i in enumerate(order)
                                     for name in [_assign(j)]], chunksize=200)
    out, made, quota = [], collections.Counter(), {k: int(v[0] * SPEC.ROWS) for k, v in SPEC.CONSTRUCTIONS.items()}
    for name, r in done:
        if r is not None and made[name] < quota[name]:
            made[name] += 1
            out.append(r)
    picked = [i for i in order[:SPEC.ROWS]]
    got = collections.Counter()   # rare meaning types topped up from the whole sample
    for r in rows:
        t = ' '.join(r['tokens'])
        need = [k for k, q in SPEC.RARE_MEANING_QUOTAS.items() if got[k] < q and PAT[k].search(t)]
        if need and not PAT["X 's R is Y"].search(t):
            for k in need:
                got[k] += 1
            out.append({**r, 'kind': 'v8-rare-' + str(r.get('kind'))})
    rng.shuffle(out)
    with gzip.open(a.out, 'wt') as f:
        f.write(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out))
    after = collections.Counter(BASE(labels[i]) for i in picked if labels[i])
    atot = sum(after.values())
    texts = [' '.join(r['tokens']) for r in out[:40000]]
    rs, ts = shares(real_texts()), shares(texts)
    manifest = {
        'rows': len(out), 'source_pool': a.pool, 'spec': {k: v[0] for k, v in SPEC.CONSTRUCTIONS.items()},
        'constructions_made': dict(made), 'rare_meaning_rows': dict(got),
        'answers_before': {k: round(v / ttot, 4) for k, v in train.most_common()},
        'answers_after': {k: round(v / atot, 4) for k, v in after.most_common()},
        'answers_real': {k: round(v, 4) for k, v in sorted(real.items(), key=lambda x: -x[1])[:40]},
        'phrases_real_vs_v8': {k: [round(rs[k], 4), round(ts[k], 4)] for k in rs},
    }
    json.dump(manifest, open(a.out.replace('_extra.jsonl.gz', '_manifest.json'), 'w'), indent=1)
    print(len(out), 'rows ->', a.out, '| constructions', dict(made), '| rare', sum(got.values()))
    for k in ['cousin', 'uncle', 'aunt', 'nephew', 'brother', 'sister', 'son-in-law', 'daughter-in-law', 'sister-in-law']:
        print(f'  {k:16s} real {real.get(k, 0):5.1%}  before {train.get(k, 0) / ttot:5.1%}  after {after.get(k, 0) / max(1, atot):5.1%}')


if __name__ == '__main__':
    main()
