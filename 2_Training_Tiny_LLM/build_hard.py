"""Hard constructions for the reader: verified generated questions with extra sentences that are true in every family
the engine finds, written the way real exam puzzles write them and the reader misreads (dev error study):

    negation      "@4 does not have a daughter" / "@4 has no sons"          -> @4 has 0 daughter
    counts        "@2 has three children , only one of whom is a boy"       -> @2 has 3 child ; @2 has 1 son
    gender        "the gender of @1 and @3 is the same" / "if ... , how ..." -> @1 same @3
    coordination  "@0 is the brother of @1 and @2" (two sentences merged; facts unchanged)

A row is kept only when the engine gives the same answers with the extra facts as without (the facts are implied, so
they never change the question), so every row is exact. Generated questions only; never evaluation text.

    python build_hard.py --n 30000 --out data/hard_v1.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import random
import re
from multiprocessing import Pool

KINDS = {'puzzle': 45, 'statements': 30, 'nested': 15, 'story': 10}
from kinlogic.facts import QUESTIONS
QUESTION = re.compile(r'^(?:%s)\b(?! *@\d+ *(?:same|opposite)\b)' % '|'.join(map(re.escape, QUESTIONS)))   # 'gender @3' asks; '@1 same @3' states
COORD = ('brother', 'sister', 'son', 'daughter', 'father', 'mother', 'husband', 'wife', 'uncle', 'aunt', 'nephew', 'niece',
         'grandson', 'granddaughter', 'grandfather', 'grandmother', 'cousin', 'brother-in-law', 'sister-in-law')
RELS = ('son', 'daughter', 'brother', 'sister', 'child')
NUM = {0: 'no', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}
PLURAL = {'son': 'sons', 'daughter': 'daughters', 'brother': 'brothers', 'sister': 'sisters', 'child': 'children'}


def _a(word):
    return 'an' if word[0] in 'aeiou' else 'a'   # "an uncle", "an aunt"


def _plural(rel):
    if rel.endswith('-in-law'):
        return rel.replace('-in-law', 's-in-law')   # brothers-in-law
    return PLURAL.get(rel, rel + 's')


def _answers(r):
    return [sorted(a['key'] for a in q['answers']) for q in r.get('questions', [])]


def _solve(lf, people):
    from kinlogic.engine import solve
    from kinlogic.facts import parse
    try:
        r = solve(parse(lf)[0], people=people)
    except Exception:
        return None
    return r if r.get('ok') and r.get('questions') else None


def _always(r, fact):
    from kinlogic.engine import claim_holds
    from kinlogic.facts import parse
    group = parse(fact)[0]
    return all(claim_holds(w, group, r['ids']) is True for w in r['worlds'])


def _count(r, x, rel):
    for n in range(5):
        if _always(r, f'{x} has {n} {rel}'):
            return n
    return None


def _num(n, rng):
    return str(n) if rng.random() < 0.3 else NUM[n]


def has_sentences(r, people, rng):
    """(tokens, facts) for count / negation sentences true in every family."""
    out = []
    for x in rng.sample(people, len(people)):
        rels = list(RELS)
        rng.shuffle(rels)
        for rel in rels:
            n = _count(r, x, rel)
            if n is None:
                continue
            if rel == 'child' and n >= 2:
                m = _count(r, x, 'son')
                if m is not None and 0 < m < n:   # "has three children , only one of whom is a boy"
                    boy = 'male' if rng.random() < 0.5 else ('boy' if m == 1 else 'boys')
                    who = 'one' if m == 1 else NUM[m]
                    toks = [x, 'has', _num(n, rng), 'children', ',', 'only', who, 'of', 'whom', 'is' if m == 1 else 'are',
                            *(['a'] if boy == 'boy' else []), boy, '.']
                    out.append((toks, [f'{x} has {n} child', f'{x} has {m} son']))
                    break
            if n == 0:
                toks = rng.choice([[x, 'does', 'not', 'have', 'a', rel, '.'], [x, 'has', 'no', PLURAL[rel], '.'],
                                   [x, 'does', 'not', 'have', 'any', PLURAL[rel], '.'], [x, "doesn't", 'have', 'a', rel, '.']])
                out.append((toks, [f'{x} has 0 {rel}']))
                break
            if rel != 'child' and n >= 1 and rng.random() < 0.4:
                toks = ([x, 'has', 'only', 'one', rel, '.'] if n == 1 and rng.random() < 0.5
                        else [x, 'has', _num(n, rng), rel if n == 1 else PLURAL[rel], '.'])
                out.append((toks, [f'{x} has {n} {rel}']))
                break
        if len(out) >= 2 or (out and rng.random() < 0.6):   # mostly one: negations are the easiest true fact to find
            break
    return out


def coord_sentence(r, people, rng):
    """'@x is the REL of @y and @z' (both relations proved in every family) or '@x has a R1 and a R2 , @y and @z'."""
    rel_of = collections.defaultdict(list)   # x -> [(rel, y)]: y is x's rel
    pairs = [(x, y) for x in people for y in people if x != y]
    rng.shuffle(pairs)
    for x, y in pairs[:24]:
        for rel in rng.sample(COORD, len(COORD)):
            if _always(r, f'{x} {rel} {y}'):
                rel_of[x].append((rel, y))
                break
    for x in rng.sample(list(rel_of), len(rel_of)):
        got = rel_of[x]
        if len(got) < 2:
            continue
        (r1, y), (r2, z) = got[:2]
        if r1 == r2:   # "has two sisters" states the count too (the generator's convention): only when it is proved
            if _always(r, f'{x} has 2 {r1}'):
                return [x, 'has', 'two', _plural(r1), ',', y, 'and', z, '.'], [f'{x} has 2 {r1}', f'{x} {r1} {y}', f'{x} {r1} {z}']
            return [y, 'and', z, 'are', 'both', x, "'s", _plural(r1), '.'], [f'{x} {r1} {y}', f'{x} {r1} {z}']
        return [x, 'has', _a(r1), r1, 'and', _a(r2), r2, ',', y, 'and', z, ',', 'respectively', '.'], [f'{x} {r1} {y}', f'{x} {r2} {z}']
    inv = collections.defaultdict(list)   # (rel, x) -> [y]: x is y's rel
    for y, lst in rel_of.items():
        for rel, x in lst:
            inv[(rel, x)].append(y)
    for (rel, x), ys in inv.items():
        if len(ys) >= 2:
            y, z = ys[:2]
            return [x, 'is', 'the', rel, 'of', y, 'and', z, '.'], [f'{y} {rel} {x}', f'{z} {rel} {x}']
    return None


def gender_sentence(r, people, rng):
    pairs = [(a, b) for i, a in enumerate(people) for b in people[i + 1:]]
    rng.shuffle(pairs)
    for a, b in pairs[:12]:
        for word in ('same', 'opposite'):
            if _always(r, f'{a} {word} {b}'):
                if word == 'same':
                    toks = rng.choice([['the', 'gender', 'of', a, 'and', b, 'is', 'the', 'same', '.'],
                                       [a, 'and', b, 'are', 'of', 'the', 'same', 'gender', '.'],
                                       [a, 'and', b, 'have', 'the', 'same', 'gender', '.']])
                else:
                    toks = rng.choice([[a, 'and', b, 'are', 'of', 'opposite', 'genders', '.'],
                                       ['the', 'gender', 'of', a, 'is', 'different', 'from', 'that', 'of', b, '.']])
                return toks, [f'{a} {word} {b}'], (a, b, word)
    return None


def coordinate(tokens):
    """'@0 is the brother of @1 . @0 is the brother of @2 .' -> '@0 is the brother of @1 and @2 .' (facts unchanged)."""
    s = ' '.join(tokens)
    pat = re.compile(r'(@\d+) is the ([a-z-]+) of (@\d+) \. \1 is the \2 of (@\d+) \.')
    m = pat.search(s)
    if not m:
        return None
    return (s[:m.start()] + f'{m.group(1)} is the {m.group(2)} of {m.group(3)} and {m.group(4)} .' + s[m.end():]).split()


def augment(ex, rng):
    tokens, lf = list(ex['tokens']), ex['lf']
    if any(t in ('"', '“', '”') for t in tokens) or tokens[-1] != '?':
        return None
    people = sorted({t for t in tokens if re.fullmatch(r'@\d+', t)}, key=lambda t: int(t[1:]))
    r = _solve(lf, people)
    if r is None or not r.get('worlds'):
        return None
    base = _answers(r)
    q0 = max((i + 1 for i, t in enumerate(tokens[:-1]) if t in ('.', '?', '!')), default=0)   # the question sentence
    body, question = tokens[:q0], tokens[q0:]
    adds, cond, used = [], [], []   # adds: (sentence tokens, its facts) in text order; cond: facts of an "if" clause
    for toks, facts in has_sentences(r, people, rng):
        adds.append((toks, facts))
        used.append('negation' if re.search(r' 0 [a-z]+$', facts[0]) else 'count')
    g = gender_sentence(r, people, rng) if rng.random() < 0.6 else None
    if g:
        toks, facts, (a, b, word) = g
        if word == 'same' and rng.random() < 0.4 and question[:1] in (['how'], ['who'], ['what']):
            question = ['if', 'the', 'gender', 'of', a, 'and', b, 'is', 'same', ','] + question   # "If the gender of G and D is same, how ..."
            cond = facts
            used.append('gender-if')
        else:
            adds.append((toks, facts))
            used.append('gender')
    co = coordinate(body) if rng.random() < 0.7 else None
    if co:
        body = co
        used.append('coordination')
    cs = coord_sentence(r, people, rng) if rng.random() < 0.7 else None
    if cs:
        adds.insert(rng.randrange(len(adds) + 1), cs)
        used.append('coordination')
    new_facts = [f for _, fs in adds for f in fs] + cond   # the facts in the order the text states them
    if not new_facts and not co:
        return None
    segs = [x.strip() for x in lf.split(';')]
    at = next((i for i, x in enumerate(segs) if QUESTION.match(x)), len(segs))
    new_lf = ' ; '.join(segs[:at] + new_facts + segs[at:])
    r2 = _solve(new_lf, people)   # implied: same answers and not one possible family removed (a count can miss open genders)
    if r2 is None or _answers(r2) != base or len(r2.get('worlds', [])) != len(r['worlds']):
        return None
    new_tokens = body + [t for a, _ in adds for t in a] + question
    return {'kind': 'hard-' + ex['kind'], 'tokens': new_tokens, 'lf': new_lf, 'text': ' '.join(new_tokens),
            'truth': ex.get('truth'), 'constructs': used}


def _job(args):
    kind, n, seed = args
    from build_dataset import engine_ok
    from kinlogic import generate as gen
    rng = random.Random(seed)
    out, tried = [], 0
    while len(out) < n and tried < n * 30:
        tried += 1
        ex = gen.KINDS[kind](rng)
        if not ex or not engine_ok(ex):
            continue
        got = augment(ex, rng)
        if got:
            out.append(got)
    return kind, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30000)
    ap.add_argument('--seed', type=int, default=15)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--out', default='data/hard_v1.jsonl.gz')
    a = ap.parse_args()
    tot = sum(KINDS.values())
    jobs, chunk = [], 250
    for kind, w in KINDS.items():
        need = round(a.n * w / tot)
        jobs += [(kind, min(chunk, need - i), a.seed * 100003 + len(jobs) + i) for i in range(0, need, chunk)]
    rows, by, cons = [], collections.Counter(), collections.Counter()
    with Pool(a.workers) as p:
        for kind, got in p.imap(_job, jobs):
            rows += got
            by[kind] += len(got)
            cons.update(c for r in got for c in r['constructs'])
    random.Random(a.seed).shuffle(rows)
    with gzip.open(a.out + '.tmp', 'wt') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(a.out + '.tmp', a.out)
    man = {'rows': len(rows), 'by_kind': dict(by), 'constructs': dict(cons), 'seed': a.seed}
    json.dump(man, open(re.sub(r'(\.jsonl)?(\.gz)?$', '', a.out) + '_manifest.json', 'w'), indent=1)
    print(json.dumps(man))


if __name__ == '__main__':
    main()
