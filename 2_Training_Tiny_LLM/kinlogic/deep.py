"""Deep chains: questions whose two people are 1-12 links apart (up to 12 generations), in five formats.

The family is built first (a chain of parent, child, sibling and spouse steps), so every fact and the true relation
are known exactly before any text exists; `check()` then re-solves the written facts with the engine and the two
must agree. Step pairs that collapse into a shorter relation ("father's wife" = mother, "parent's child" = sibling)
are never used, so a d-link chain really is d links.
"""
from __future__ import annotations

import random

from .engine import World, relation_in, solve
from .facts import parse
from .generate import (F, LETTERS, PRON, Q, SPEECH_OPEN, ask_text, cap, gender_hint, implied, options_for, poss,
                       state)

WORD = {'U': ('father', 'mother'), 'D': ('son', 'daughter'), 'B': ('brother', 'sister'), 'S': ('husband', 'wife')}
INV = {'U': 'D', 'D': 'U', 'B': 'B', 'S': 'S'}
BAD = {('U', 'D'), ('D', 'U'), ('B', 'B'), ('S', 'S'), ('U', 'S'), ('S', 'D'), ('B', 'U'), ('D', 'B')}
OPP = {'M': 'F', 'F': 'M'}
FORMATS = ('statements', 'nested', 'speech', 'coded', 'story')
SYMBOLS = ['+', '-', '×', '÷', '@', '#', '$', '%', '&', '*', '!', '©', '★', '◆', '∆', '£', '€', '§']


def word(move, g):
    return WORD[move][0 if g == 'M' else 1]


GREAT = {2: 'grand', 3: 'great-grand', 4: 'great-great-grand', 5: 'great-great-great-grand'}
# Runs of links that one exam word names exactly (brother-in-law is left out: it has two meanings).
RUN = {('U', 'B'): ('uncle', 'aunt'), ('B', 'D'): ('nephew', 'niece'), ('U', 'B', 'D'): ('cousin', 'cousin'),
       ('S', 'U'): ('father-in-law', 'mother-in-law'), ('D', 'S'): ('son-in-law', 'daughter-in-law'),
       ('U', 'U', 'B'): ('great-uncle', 'great-aunt'), ('B', 'D', 'D'): ('grand-nephew', 'grand-niece'),
       **{('U',) * k: (p + 'father', p + 'mother') for k, p in GREAT.items()},
       **{('D',) * k: (p + 'son', p + 'daughter') for k, p in GREAT.items()}}


def segments(rng, ch, p=0.6):
    """The chain cut into runs, each named by one word: [(start index, end index, word)]."""
    out, i, mv = [], 0, ch.moves
    while i < len(mv):
        fits = [k for k in (5, 4, 3, 2) if i + k <= len(mv) and tuple(mv[i:i + k]) in RUN]
        k = fits[0] if fits and rng.random() < p else 1
        end = ch.nodes[i + k]
        out.append((i, i + k, RUN[tuple(mv[i:i + k])][0 if ch.g(end) == 'M' else 1] if k > 1 else word(mv[i], ch.g(end))))
        i += k
    return out


def deep_options(rng, label):
    """Options for a named relation, with near misses (one 'great-' more or less, the next cousin degree)."""
    if not label or "'s " in label:
        return None
    label = label.replace('paternal ', '').replace('maternal ', '')   # the words rarely fix the side; exams leave it out
    near = {label}
    if 'great-' in label or 'grand' in label:
        near |= {label.replace('great-', '', 1), 'great-' + label, label.replace('grand', 'great-grand', 1)}
    if 'cousin' in label:
        near |= {label.replace('removed', '').replace('once', '').strip() or 'cousin', 'second-cousin', label + ' once removed'}
    near = [cap(x.strip()) for x in near if x.strip()]
    opts = list(dict.fromkeys(near))[:4]
    if len(opts) < 4:
        return options_for(rng, label)
    if cap(label) not in opts:
        opts[-1] = cap(label)
    rng.shuffle(opts)
    return opts


def moves_for(rng, d):
    """d steps in the shapes exams use (80%): straight up or down the generations, uncle and nephew lines
    (great-great-uncle, grand-nephew), cousins (third cousin twice removed), each with at most one in-law step at
    either end; the rest a free walk through parents, children, siblings and spouses."""
    for _ in range(500):
        u = rng.random()
        if u < 0.8:
            core = d - (1 if d >= 2 and rng.random() < 0.3 else 0)       # room for one spouse step
            shape = rng.random()
            if shape < 0.35 or core < 2:
                mv = [rng.choice('UD')] * core
            elif shape < 0.6:
                a = rng.randint(1, core - 1)
                mv = ['U'] * a + ['B'] + ['D'] * (core - a - 1) if core - a - 1 >= 0 else ['U'] * core
            else:
                k = rng.randint(1, core - 1)
                mv = (['U'] * (core - 1) + ['B']) if rng.random() < 0.5 else (['B'] + ['D'] * (core - 1))
                mv = mv if k else mv
            if len(mv) < d:
                mv = (['S'] + mv) if rng.random() < 0.5 else (mv + ['S'])
        else:
            mv = []
            for _ in range(d):
                ok = [m for m in 'UDBS' if not mv or (mv[-1], m) not in BAD]
                mv.append(rng.choices(ok, [3 if m in 'UD' else 1 for m in ok])[0])
        if len(mv) == d and all(p not in BAD for p in zip(mv, mv[1:])) and mv.count('S') <= 2:
            return mv
    return None


class Chain:
    """A real family holding the chain (plus the parents a sibling step needs), as an engine World."""

    def __init__(self, rng, moves):
        self.w, self.people, self.level, self.moves = World(), [], {}, moves
        w = self.w

        def person(g, lev):
            i = w.add(f'@p{len(self.people)}')
            w.gen[i] = g
            self.people.append(i)
            self.level[i] = lev
            return i

        cur = person(rng.choice('MF'), 0)
        self.nodes = [cur]
        for m in moves:
            lev = self.level[cur]
            if m == 'S':
                nxt = person(OPP[w.gen[cur]], lev)
                ok = w.marry(cur, nxt)
            elif m == 'U':
                g = rng.choice('MF')
                nxt = person(g, lev - 1)
                ok = w.set_parent(cur, nxt, 'fa' if g == 'M' else 'mo')
            elif m == 'D':
                nxt = person(rng.choice('MF'), lev + 1)
                ok = w.set_parent(nxt, cur, 'fa' if w.gen[cur] == 'M' else 'mo')
            else:   # a sibling shares both parents
                fa, mo = w.father(cur), w.mother(cur)
                ok = True
                if fa < 0:
                    fa = person('M', lev - 1)
                    ok = ok and w.set_parent(cur, fa, 'fa')
                if mo < 0:
                    mo = person('F', lev - 1)
                    ok = ok and w.set_parent(cur, mo, 'mo')
                if w.spouse(fa) < 0:
                    ok = ok and w.marry(fa, mo)
                nxt = person(rng.choice('MF'), lev)
                ok = ok and w.set_parent(nxt, fa, 'fa') and w.set_parent(nxt, mo, 'mo')
            if not ok:
                raise ValueError('chain does not fit')
            cur = nxt
            self.nodes.append(cur)
        if not w.normalize():
            raise ValueError('chain does not fit')

    def g(self, i):
        return self.w.gen[i]

    def truth(self, a, b):
        return relation_in(self.w, a, b)


# ---------- formats ----------

def _ask(rng, q: Q, ch: Chain, kind):
    a, b = ch.nodes[0], ch.nodes[-1]
    x, y = (a, b) if rng.random() < 0.5 else (b, a)
    rx, ry = q.refs[x], q.refs[y]
    if not implied(q, ry) and rng.random() < 0.75:   # exams state the gender the answer needs
        gender_hint(q, ry)
    q.add(ask_text(rng, q.say(rx, True), q.say(ry, True)), [{'t': 'ask', 'x': rx, 'y': ry}])
    t = ch.truth(x, y)
    opts = deep_options(rng, t.get('label') or t['key']) if t['key'] != 'path' and rng.random() < 0.35 else None
    return q.finish(kind, opts, {'truth': t['key'], 'base': t.get('base'), 'depth': len(ch.moves), 'y_gender': ch.g(y),
                                 'x_gender': ch.g(x)}, noise=len(ch.moves) <= 4)


def _clause(rng, q: Q, ch: Chain, seg):
    """One run as a statement, forwards ("B is A's grandfather") or, for single links, inverted ("A is B's son")."""
    i, j, w = seg
    a, b = ch.nodes[i], ch.nodes[j]
    ra, rb = q.ref(a), q.ref(b)
    if j == i + 1 and rng.random() < 0.3:
        w = word(INV[ch.moves[i]], ch.g(a))
        return state(rng, rb.surface, ra.surface, w), F(rb, w, ra)
    return state(rng, ra.surface, rb.surface, w), F(ra, w, rb)


def f_statements(rng, ch: Chain):
    q = Q(rng, ch)
    clauses = [_clause(rng, q, ch, seg) for seg in segments(rng, ch)]
    if rng.random() < 0.35 and all(f['t'] == 'rel' for _, f in clauses):   # natural prose, in chain order
        from .prose import add_all
        add_all(q, rng, [f for _, f in clauses], joint=lambda a, b, c: None not in (a.node, b.node, c.node) and
                {ch.father(c.node), ch.mother(c.node)} == {a.node, b.node})
        return _ask(rng, q, ch, 'deep-statements')
    if rng.random() < 0.4:
        rng.shuffle(clauses)
    if len(clauses) > 1 and rng.random() < 0.4:   # "A is the mother of B, B is the father of C and C is ..."
        text = ', '.join(c for c, _ in clauses[:-1]) + ' and ' + clauses[-1][0]
        q.add(cap(text) + '.', [f for _, f in clauses])
    else:
        for c, f in clauses:
            q.add(cap(c) + '.', [f])
    return _ask(rng, q, ch, 'deep-statements')


def _words(segs):
    return [w for _, _, w in segs]


def _phrase(rng, anchor: str, words, mine=False):
    """'the son of the brother of Varun's mother', "Varun's mother's brother's son" or a mix ('my' for a speaker)."""
    own = (lambda ws: 'my ' + "'s ".join(ws)) if mine else (lambda ws: poss(anchor) + ' ' + "'s ".join(ws))
    style = rng.random()
    if style < 0.25 and len(words) <= 6:
        return own(words)
    k = rng.randint(1, len(words)) if style < 0.6 else 1
    outer = list(reversed(words[k:]))
    return ('the ' + ' of the '.join(outer) + ' of ' if outer else '') + own(words[:k])


def _chain_facts(q: Q, ch: Chain, segs, first, last):
    refs = [first] + [q.anon(ch.nodes[j]) for _, j, _ in segs[:-1]] + [last]
    return [F(refs[k], w, refs[k + 1]) for k, (_, _, w) in enumerate(segs)]


def f_nested(rng, ch: Chain):
    q = Q(rng, ch)
    segs = segments(rng, ch)
    r0 = q.ref(ch.nodes[0])
    np_ = _phrase(rng, r0.surface, _words(segs))
    rt = q.ref(ch.nodes[-1])
    q.add(rng.choice([f'{rt.surface} is {np_}.', f'{cap(np_)} is {rt.surface}.']), _chain_facts(q, ch, segs, r0, rt))
    return _ask(rng, q, ch, 'deep-nested')


def f_speech(rng, ch: Chain):
    q = Q(rng, ch)
    segs = segments(rng, ch)
    rs = q.ref(ch.nodes[0])
    rt = q.desc(ch.nodes[-1])
    pr = PRON[ch.g(ch.nodes[-1])][0]
    utter = f'"{cap(pr)} is {_phrase(rng, "", _words(segs), mine=True)}."'
    opener = rng.choice(SPEECH_OPEN).format(P=q.say(rt), S=rs.surface, U=utter)
    q.add(opener, _chain_facts(q, ch, segs, rs, rt), evidence=[rt])
    return _ask(rng, q, ch, 'deep-speech')


def f_coded(rng, ch: Chain):
    """'A @ B means A is the father of B' (or reversed: "'A @ B' means 'B is the father of A'") and an expression
    through the whole chain."""
    q = Q(rng, ch, style='letters')
    refs = [q.ref(n) for n in ch.nodes]
    d = len(ch.moves)
    order = list(range(d + 1)) if rng.random() < 0.5 else list(range(d, -1, -1))
    flipped = rng.random() < 0.3
    pairs = []   # (left, right, word, fact) for each adjacent pair of the expression
    for j in range(d):
        u, v = order[j], order[j + 1]
        x, y = (v, u) if flipped else (u, v)            # the code says "x is the word of y"
        i = min(u, v)
        m = ch.moves[i] if x == i + 1 else INV[ch.moves[i]]
        w = word(m, ch.g(ch.nodes[x]))
        pairs.append((refs[u], refs[v], w, F(refs[y], w, refs[x])))
    used = sorted({w for _, _, w, _ in pairs})
    sym = dict(zip(used, rng.sample(SYMBOLS, len(used))))
    taken = {r.surface for r in refs}
    pair = next((p for p in rng.sample([('A', 'B'), ('P', 'Q'), ('X', 'Y'), ('M', 'N'), ('L', 'K')], 5) if not set(p) & taken),
                None) or tuple(rng.sample([c for c in LETTERS if c not in taken], 2))
    A, B = pair
    defs = [f"'{A} {sym[w]} {B}' means '{B} is the {w} of {A}'" if flipped else f'{A} {sym[w]} {B} means {A} is the {w} of {B}'
            for w in used]
    rng.shuffle(defs)
    expr = refs[order[0]].surface + ''.join(f' {sym[w]} {r.surface}' for _, r, w, _ in pairs)
    lead = rng.choice(['If ', 'In a certain code, ', 'Given that '])
    joined = ', '.join(defs[:-1]) + (' and ' if len(defs) > 1 else '') + defs[-1]
    q.add(f'{lead}{joined}.', [])
    q.add(rng.choice([f'If {expr},', f'In the expression {expr},', f"Consider the expression '{expr}'."]), [f for *_, f in pairs])
    return _ask(rng, q, ch, 'deep-coded')


STORY = {
    'U': ['{X} went to the market with {xp} {w}, {Y}.', '{X} called {xp} {w} {Y} after dinner.',
          '{X} visited {xp} {w} {Y} on Sunday.', '{X} gifted a watch to {xp} {w} {Y}.'],
    'D': ['{X} took {xp} {w} {Y} to school.', '{X} bought a bicycle for {xp} {w} {Y}.',
          '{X} taught {xp} {w} {Y} to swim.', '{X} and {xp} {w} {Y} planted a tree.'],
    'B': ['{X} and {xp} {w} {Y} watched a cricket match together.', '{X} helped {xp} {w} {Y} with homework.',
          '{X} played chess with {xp} {w} {Y}.'],
    'S': ['{X} and {xp} {w} {Y} went on a holiday.', '{X} celebrated the anniversary with {xp} {w} {Y}.',
          '{X} cooked dinner for {xp} {w} {Y}.'],
}
FILLER = ['{X} loves gardening.', '{X} works in a bank.', 'It rained heavily that evening.', '{X} is fond of music.',
          'Everyone enjoyed the festival.', '{X} reads the newspaper every morning.', '{X} lives in Pune.']


def f_story(rng, ch: Chain):
    """Sentences with pronouns ("Ravi took his daughter Meera to school"), shuffled, with sentences that state no
    family link at all."""
    q = Q(rng, ch, style=rng.choice(['indian', 'world', 'mixed']))
    items = []
    for i, m in enumerate(ch.moves):
        a, b = ch.nodes[i], ch.nodes[i + 1]
        if rng.random() < 0.35:
            a, b, m = b, a, INV[m]
        ra, rb = q.ref(a), q.ref(b)
        w = word(m, ch.g(b))
        t = rng.choice(STORY[m]).format(X=ra.surface, Y=rb.surface, w=w, xp=PRON[ch.g(a)][2])
        items.append((t, [F(ra, w, rb)], [ra]))
    names = [r.surface for r in q.refs.values()]
    for _ in range(rng.choice([0, 1, 1, 2])):
        items.append((rng.choice(FILLER).format(X=rng.choice(names)), [], []))
    if rng.random() < 0.5:
        rng.shuffle(items)
    for t, fs, ev in items:
        q.add(t, fs, evidence=ev)
    return _ask(rng, q, ch, 'deep-story')


RENDER = {'statements': f_statements, 'nested': f_nested, 'speech': f_speech, 'coded': f_coded, 'story': f_story}


def make(rng, fmt: str, d: int) -> dict | None:
    mv = moves_for(rng, d)
    if mv is None:
        return None
    try:
        ch = Chain(rng, mv)
    except ValueError:
        return None
    return RENDER[fmt](rng, ch)


def sample(rng, depths=range(1, 13), formats=FORMATS) -> dict:
    """One deep question for training (depths weighted to the long end)."""
    depths = list(depths)
    while True:
        ex = make(rng, rng.choice(list(formats)), rng.choices(depths, [1 + d for d in depths])[0])
        if ex is not None:
            return ex


def check(ex) -> str | None:
    """None when the written facts give the true relation, else why not. The engine re-solves the facts; when the
    text leaves the asked person's gender open ("child"), it re-solves with that gender added and must then get the
    exact relation. Sets ex['answer'] to what the text itself supports."""
    st, bad = parse(ex['lf'])
    if bad:
        return 'unparsed facts'
    people = sorted({t for t in ex['tokens'] if t[0] == '@'})

    def got(stmts):
        r = solve(stmts, people=people, max_steps=20000)
        if not r['ok'] or not r['questions'] or not r['questions'][-1]['answers']:
            return None, None
        q = r['questions'][-1]
        keys = {a.get('key') for a in q['answers']} | {a.get('base') for a in q['answers']} | {q.get('neutral')}
        return keys - {None}, q.get('neutral') or q['answers'][0].get('key')

    keys, ans = got(st)
    if keys is None:
        return 'engine found no answer'
    ex['answer'] = ans
    if ex['truth'] in keys or ex.get('base') in keys:
        return None
    x, y = st[-1].get('x'), st[-1].get('y')
    sex = {'M': 'male', 'F': 'female'}
    keys2, _ = got(st + parse(f"{x} {sex[ex['x_gender']]} ; {y} {sex[ex['y_gender']]}")[0])
    if keys2 and (ex['truth'] in keys2 or ex.get('base') in keys2):
        return None
    if _neutral(ex['truth']) in {_neutral(k) for k in keys | (keys2 or set())}:   # same path; the text leaves
        return None                                                                 # someone on it ungendered
    return f"engine says {sorted(keys)[:3]}, truth {ex['truth']}"


NEUTRAL = {'father': 'parent', 'mother': 'parent', 'son': 'child', 'daughter': 'child', 'husband': 'spouse',
           'wife': 'spouse', 'brother': 'sibling', 'sister': 'sibling'}


def _neutral(key):
    import re
    return re.sub(r"\b(father|mother|son|daughter|husband|wife|brother|sister)\b", lambda m: NEUTRAL[m.group(1)], key or '')
