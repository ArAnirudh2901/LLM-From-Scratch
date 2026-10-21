"""Natural wording for a run of plain facts, the constructions real questions use and the v10-r1 reader missed:
apposition ("Arjun's father, Dhruv, is Rina's son"), "whose" ("Rina's mother is Saroj, whose husband Bimal is the
son of Hari"), pronouns ("He is ...", "Her mother is ...", "Om and his brother Jai ..."), "their son" and comma lists.
Facts are exactly those of the plain sentences; a pronoun or possessive also states that person's gender.

A pronoun is used only for the previous sentence's subject, and only when no one else in that sentence has the same
gender, so every pronoun has one reading.
"""
from __future__ import annotations

from .generate import F, G, cap, poss, state

PRON = {'M': ('he', 'his'), 'F': ('she', 'her')}
CHILD = {'son', 'daughter', 'child'}
SPOUSE = {'husband', 'wife'}
TOGETHER = ['live in the same town', 'went to the market together', 'run a small shop', 'visited the temple on Sunday',
            'were at the wedding', 'came to the school function', 'travelled to the city last week']


def _simple(f):
    return f['t'] == 'rel' and not f.get('side') and not f.get('only') and not f.get('neg')


def _of(rng, x, w):
    """ "the w of x" or "x's w" """
    return rng.choice([f'the {w} of {x.surface}', f'{poss(x.surface)} {w}'])


def render(rng, facts, joint=None):
    """[(sentence, facts, evidence refs)] for the facts in order (each fact: x's r is y). joint(a, b, c): c is the child
    of both a and b in the true family ("their son" is only said then)."""
    out, i = [], 0
    prev_subj, prev_refs = None, []

    def emit(text, fs, subj, refs, ev=()):
        nonlocal prev_subj, prev_refs
        out.append((cap(text) + '.', fs, list(ev)))
        prev_subj, prev_refs = subj, refs

    def can_pronoun(r):
        return (r is prev_subj and r.kind in ('name', 'letter') and r.g
                and not any(o is not r and o.g == r.g for o in prev_refs))

    while i < len(facts):
        f = facts[i]
        nxt = facts[i + 1] if i + 1 < len(facts) else None
        x, w, y = f['x'], f['r'], f['y']
        pick = rng.random()
        if not _simple(f):
            emit(state(rng, x.surface, y.surface, w, f.get('side'), f.get('only')), [f], y, [x, y])
            i += 1
            continue
        if can_pronoun(y) and pick < 0.5:   # "He is the son of Ved."
            p = PRON[y.g][0]
            emit(rng.choice([f'{p} is {_of(rng, x, w)}', f'{p} is also {_of(rng, x, w)}']), [f], y, [x, y], [y])
            i += 1
            continue
        if can_pronoun(x) and pick < 0.5:   # "Her mother is Saroj."
            emit(f'{PRON[x.g][1]} {w} is {y.surface}', [f], y, [x, y], [x])
            i += 1
            continue
        if nxt and _simple(nxt):
            x2, w2, y2 = nxt['x'], nxt['r'], nxt['y']
            if y2 is y and pick < 0.45:   # apposition: "Arjun's father, Dhruv, is Rina's son"
                t = rng.choice([f'{poss(x.surface)} {w} {y.surface} is {_of(rng, x2, w2)}',
                                f'{poss(x.surface)} {w}, {y.surface}, is {_of(rng, x2, w2)}',
                                f'{y.surface}, {poss(x.surface)} {w}, is {_of(rng, x2, w2)}'])
                emit(t, [f, nxt], y, [x, y, x2])
                i += 2
                continue
            if x2 is y and pick < 0.45:   # "whose": "Rina's mother is Saroj, whose husband Bimal is Hari's son"
                third = facts[i + 2] if i + 2 < len(facts) else None
                if third and _simple(third) and third['y'] is y2 and rng.random() < 0.5:
                    t = f"{poss(x.surface)} {w} is {y.surface}, whose {w2} {y2.surface} is {_of(rng, third['x'], third['r'])}"
                    emit(t, [f, nxt, third], y2, [x, y, y2, third['x']])
                    i += 3
                    continue
                t = rng.choice([f'{poss(x.surface)} {w} is {y.surface}, whose {w2} is {y2.surface}',
                                f'{y.surface} is {_of(rng, x, w)}, and {poss(y.surface)} {w2} is {y2.surface}'])
                emit(t, [f, nxt], y2, [x, y, y2])
                i += 2
                continue
            other = y if x2 is x else x
            if w in SPOUSE and w2 in CHILD and x2 in (x, y) and joint and joint(x, y, y2) and pick < 0.45:   # "their son"
                t = rng.choice([f'{x.surface} and {y.surface} are married, and {y2.surface} is their {w2}',
                                f'{y2.surface} is the {w2} of {x.surface} and {poss(x.surface)} {w} {y.surface}'])
                emit(t, [f, nxt, F(other, w2, y2)], y2, [x, y, y2])
                i += 2
                continue
        if x.kind in ('name', 'letter') and x.g and pick < 0.12:   # "Om and his brother Jai live in the same town."
            emit(f'{x.surface} and {PRON[x.g][1]} {w} {y.surface} {rng.choice(TOGETHER)}', [f], x, [x, y], [x])
            i += 1
            continue
        if pick < 0.2 and i + 2 < len(facts) and all(_simple(g) for g in facts[i:i + 3]):   # comma list
            fs = facts[i:i + 3]
            parts = [f"{poss(g['x'].surface)} {g['r']} is {g['y'].surface}" for g in fs]
            emit(f'{parts[0]}, {parts[1]}, and {parts[2]}', fs, fs[-1]['y'], [r for g in fs for r in (g['x'], g['y'])])
            i += 3
            continue
        emit(state(rng, x.surface, y.surface, w), [f], y, [x, y])
        i += 1
    return out


def add_all(q, rng, facts, joint=None):
    """Adds the rendered sentences to a generate.Q (pronouns become gender facts via evidence)."""
    for text, fs, ev in render(rng, facts, joint):
        q.add(text, fs, evidence=ev)
    return q


__all__ = ['render', 'add_all', 'G']
