"""Construction suite (behavioural testing per sentence type, as CheckList, Ribeiro et al. ACL 2020): held-out
templates for the ways a family fact is written, from exam forms to how people type. Each item carries its own facts;
the gold answer is the engine's answer on those facts for the same prepared question, so it is right by construction,
and a model is scored per type. Templates are written here, not taken from the generator: they measure
generalization, not memory.

    python -m eval.construction_suite --ckpt runs/v10-r6a/best.pt --n 20
"""
from __future__ import annotations

import argparse
import collections
import json
import random

M_REL = ['father', 'son', 'brother', 'husband']
F_REL = ['mother', 'daughter', 'sister', 'wife']
HINDI = {'mama': ('maternal uncle', 'M'), 'chacha': ('paternal uncle', 'M'), 'bua': ('paternal aunt', 'F'), 'mausi': ('maternal aunt', 'F'),
         'nana': ('maternal grandfather', 'M'), 'dada': ('paternal grandfather', 'M'), 'nani': ('maternal grandmother', 'F'), 'dadi': ('paternal grandmother', 'F')}
INFORMAL = {'father': 'dad', 'mother': 'mom', 'brother': 'bro', 'sister': 'sis'}
KINDS = ['copula 2', 'copula 3', 'possessive chain', 'of-chain', 'who clause (nearest)', 'who clause (subject)', 'whose clause',
         'stacked clauses', 'coordination', 'married to', 'if-conditional', 'question first', 'pronoun', 'apposition',
         'speech (pointing)', 'informal first person', 'lowercase no punctuation', 'negation', 'count + only', 'gender from spouse', 'gender from another relation', 'gender unknown', 'names (gendered)']


def _rel(rng):
    return (rng.choice(M_REL), 'M') if rng.random() < 0.5 else (rng.choice(F_REL), 'F')


def make(kind, rng, NM, NF):
    """(question text, facts over template people, {template person: name in text}, formal twin text or None)."""
    (r1, g1), (r2, _), (r3, _) = _rel(rng), _rel(rng), _rel(rng)
    letters = rng.sample('PQRSTJKLMN', 4)
    A, B, C, D = letters
    N = {'@A': A, '@B': B, '@C': C, '@D': D}
    f1, f2, f3 = f'@B {r1} @A', f'@C {r2} @B', f'@D {r3} @C'
    twin = None
    if kind == 'copula 2':
        t, fs = f'{A} is the {r1} of {B}. {B} is the {r2} of {C}. How is {A} related to {C}?', [f1, f2, 'ask @C @A']
    elif kind == 'copula 3':
        t, fs = f'{A} is the {r1} of {B}. {B} is the {r2} of {C}. {C} is the {r3} of {D}. How is {A} related to {D}?', [f1, f2, f3, 'ask @D @A']
    elif kind == 'possessive chain':
        t, fs = f"{A} is {C}'s {r2}'s {r1}. How is {A} related to {C}?", [f'@C {r2} #1', f'#1 {r1} @A', 'ask @C @A']
    elif kind == 'of-chain':
        t, fs = f'{A} is the {r1} of the {r2} of {C}. How is {A} related to {C}?', [f'@C {r2} #1', f'#1 {r1} @A', 'ask @C @A']
    elif kind == 'who clause (nearest)':
        t, fs = f'{A} is the {r1} of {B}, who is the {r2} of {C}. How is {A} related to {C}?', [f1, f2, 'ask @C @A']
    elif kind == 'who clause (subject)':
        t, fs = f'{B}, who is the {r2} of {C}, has a {r1} named {A}. How is {A} related to {C}?', [f2, f1, 'ask @C @A']
    elif kind == 'whose clause':
        t, fs = f'{B}, whose {r1} is {A}, is the {r2} of {C}. How is {A} related to {C}?', [f1, f2, 'ask @C @A']
    elif kind == 'stacked clauses':
        t = f'{A} is the {r1} of {B}, whose {r3} is {D}, who is the {r2} of {C}. How is {D} related to {C}?'
        fs = [f1, f'@B {r3} @D', f'@C {r2} @D', 'ask @C @D']
    elif kind == 'coordination':
        t, fs = f'{A} is the {r1} of {B} and the {r3} of {C}. How is {B} related to {C}?', [f1, f'@C {r3} @A', 'ask @C @B']
    elif kind == 'married to':
        t, fs = f'{A} is married to {B}. {B} is the {r2} of {C}. How is {A} related to {C}?', ['@A spouse @B', f2, 'ask @C @A']
    elif kind == 'if-conditional':
        t, fs = f'If {A} is the {r1} of {B} and {B} is the {r2} of {C}, how is {A} related to {C}?', [f1, f2, 'ask @C @A']
    elif kind == 'question first':
        t, fs = f'How is {A} related to {C} if {A} is the {r1} of {B} and {B} is the {r2} of {C}?', [f1, f2, 'ask @C @A']
    elif kind == 'pronoun':
        p = 'His' if g1 == 'M' else 'Her'
        t, fs = f'{B} has a {r1} named {A}. {p} {r3} is {D}. How is {D} related to {B}?', [f1, f'@A {r3} @D', 'ask @B @D']
    elif kind == 'apposition':
        t, fs = f'{A}, the {r1} of {B}, is the {r3} of {D}. How is {D} related to {B}?', [f1, f'@D {r3} @A', 'ask @B @D']
    elif kind == 'speech (pointing)':
        hs = 'He' if g1 == 'M' else 'She'
        t, fs = f"Pointing to {A}, {C} said, '{hs} is the {r1} of my {r2}.' How is {A} related to {C}?", [f'@C {r2} #1', f'#1 {r1} @A', 'ask @C @A']
    elif kind == 'informal first person':
        t = f"my {INFORMAL.get(r2, r2)}'s {r1}, what is {'he' if g1 == 'M' else 'she'} to me?"
        fs, N = None, {}
        twin = (f"{A} is {C}'s {r2}'s {r1}. How is {A} related to {C}?", [f'@C {r2} #1', f'#1 {r1} @A', 'ask @C @A'], {'@A': A, '@C': C})
    elif kind == 'lowercase no punctuation':
        t = f"{A.lower()} is {B.lower()}'s {r1} {B.lower()} is {C.lower()}'s {r2} how is {A.lower()} related to {C.lower()}"
        fs, N = None, {}
        twin = (f'{A} is the {r1} of {B}. {B} is the {r2} of {C}. How is {A} related to {C}?', [f1, f2, 'ask @C @A'], {'@A': A, '@B': B, '@C': C})
    elif kind == 'hindi kin word':
        h = rng.choice(list(HINDI))
        en, _ = HINDI[h]
        t = f'{A} is the {h} of {C}. How is {A} related to {C}?'
        fs, N = None, {}
        twin = (f'{A} is the {en} of {C}. How is {A} related to {C}?', [f'@C {en} @A', 'ask @C @A'], {'@A': A, '@C': C})
    elif kind == 'negation':
        t = f'{A} is the child of {B}, but {B} is not the father of {A}. {C} is the father of {A}. How is {C} related to {B}?'
        fs = ['@B child @A', '@A not father @B', '@A father @C', 'ask @B @C']
    elif kind == 'count + only':
        t = f'{B} has two sons, {A} and {C}. {D} is the only daughter of {C}. How is {A} related to {D}?'
        fs = ['@B has 2 son', '@B son @A', '@B son @C', '@C only daughter @D', 'ask @D @A']
    elif kind == 'gender from spouse':
        t, fs = f'{A} is married to {B}. {B} is the {r2} of {C}. {C} is the {r3} of {D}. How is {A} related to {D}?', ['@A spouse @B', f2, f3, 'ask @D @A']
    elif kind == 'gender from another relation':
        t, fs = f'{A} is the sibling of {B}. {A} is the {r3} of {D}. {B} is the {r2} of {C}. How is {A} related to {C}?', ['@B sibling @A', f'@D {r3} @A', f2, 'ask @C @A']
    elif kind == 'gender unknown':
        t, fs = f'{A} is the child of {B}. {B} is the sibling of {C}. How is {A} related to {C}?', ['@B child @A', '@C sibling @B', 'ask @C @A']
    elif kind == 'names (gendered)':
        nA = rng.choice(NM if g1 == 'M' else NF)
        nB, nC = rng.choice([n for n in NM + NF if n != nA]), rng.choice([n for n in NM + NF if n != nA])
        if nB == nC:
            nC = nC + 'a'
        N = {'@A': nA, '@B': nB, '@C': nC}
        t, fs = f"{nA} is the {r1} of {nB}. {nB}'s {r2} is {nC}. How is {nA} related to {nC}?", [f1, f'@B {r2} @C', 'ask @C @A']
    else:
        raise ValueError(kind)
    return t, fs, N, twin


def gold(text, facts, names):
    """Engine answer on the item's own facts, template people bound to the prepared question's placeholders."""
    from kinlogic.exam import _answer_one
    from kinlogic.prepare import prepare
    p = prepare(text)
    ids = {e['text']: e['id'] for e in p['ents']}
    lf = ' ; '.join(facts)
    for tag, nm in names.items():
        if tag not in lf:
            continue
        if nm not in ids:
            return None
        lf = lf.replace(tag, ids[nm])
    res = _answer_one(p, lf, lambda b: [''] * len(b), None)
    if res.get('status') in ('contradiction', 'none', 'error') or not res.get('answer'):
        return None   # the random relation words contradict each other: the caller draws again
    return res.get('answer')


def main():
    from eval.evaluate import load
    from kinlogic.exam import answer, with_exact_clauses, with_simplified
    from reader import generate
    from train import get_device
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    try:
        d = json.load(open('kinlogic/data/perturb_names.json'))
        NM, NF = d['M'], d['F']
    except Exception:
        NM, NF = ['Arun', 'Vikram'], ['Asha', 'Neha']
    dev = get_device()
    model, vocab = load(a.ckpt, dev)
    read = with_exact_clauses(with_simplified(lambda b: generate(model, vocab, b, dev, k=8)))
    res, bad = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(list)
    for kind in KINDS:
        for _ in range(a.n):
            for _ in range(40):   # redraw until the item's own facts describe a possible family
                t, fs, names, twin = make(kind, rng, NM, NF)
                want = gold(t, fs, names) if fs else gold(*twin)
                if want:
                    break
            got = answer(t, read).get('answer')
            ok = bool(want) and str(got).strip().lower() == str(want).strip().lower()
            res[kind][0] += ok
            res[kind][1] += 1
            if not ok and len(bad[kind]) < 2:
                bad[kind].append((t, want, got))
    tot = [sum(v[0] for v in res.values()), sum(v[1] for v in res.values())]
    for k in KINDS:
        a_, n = res[k]
        print(f'{k:26s} {a_:3d}/{n:<3d} {a_ / max(1, n):5.0%}' + (f'   e.g. {bad[k][0][0][:90]!r} want {bad[k][0][1]} got {bad[k][0][2]}' if bad[k] else ''))
    print(f'ALL {tot[0]}/{tot[1]} = {tot[0] / max(1, tot[1]):.1%}')
    if a.out:
        json.dump({k: v for k, v in res.items()}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
