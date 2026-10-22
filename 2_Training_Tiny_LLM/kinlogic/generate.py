"""Synthetic blood-relation exam questions with their exact facts: the reader model's training data.

Every question is written from a random family, so its facts are true and consistent. Each sentence is rendered
together with the facts it states (in the fact language, text order), then the text goes through prepare() and the
people are mapped to its placeholders. Question kinds follow what bank (IBPS/SBI/RBI), SSC, CAT, CLAT and UPSC CSAT
papers ask: statement chains, nested descriptions, pointing/introducing speech, coded symbols, closed-family puzzles,
data sufficiency, statements to verify, stories, plus Hindi kinship words.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .engine import World, relation_in
from .names import INDIAN_F, INDIAN_M, LETTERS, SURNAMES, WORLD_F, WORLD_M
from .prepare import prepare

# ---------- relation words for runs of moves ----------
# Moves: U parent, D child, S spouse, B sibling. Value: (male word, female word, neutral word or None).
WORDS = {
    ('U',): ('father', 'mother', 'parent'), ('D',): ('son', 'daughter', 'child'),
    ('S',): ('husband', 'wife', 'spouse'), ('B',): ('brother', 'sister', 'sibling'),
    ('U', 'U'): ('grandfather', 'grandmother', 'grandparent'), ('D', 'D'): ('grandson', 'granddaughter', 'grandchild'),
    ('U', 'B'): ('uncle', 'aunt', None), ('B', 'D'): ('nephew', 'niece', None), ('U', 'B', 'D'): ('cousin', 'cousin', 'cousin'),
    ('S', 'U'): ('father-in-law', 'mother-in-law', 'parent-in-law'), ('D', 'S'): ('son-in-law', 'daughter-in-law', None),
    ('S', 'B'): ('brother-in-law', 'sister-in-law', None), ('B', 'S'): ('brother-in-law', 'sister-in-law', None),
    ('U', 'U', 'U'): ('great-grandfather', 'great-grandmother', 'great-grandparent'),
    ('D', 'D', 'D'): ('great-grandson', 'great-granddaughter', 'great-grandchild'),
    ('U', 'U', 'B'): ('great-uncle', 'great-aunt', None), ('B', 'D', 'D'): ('grand-nephew', 'grand-niece', None),
}
INVERSE = {('U',): ('D',), ('D',): ('U',), ('S',): ('S',), ('B',): ('B',), ('U', 'U'): ('D', 'D'), ('D', 'D'): ('U', 'U'),
           ('U', 'B'): ('B', 'D'), ('B', 'D'): ('U', 'B'), ('U', 'B', 'D'): ('U', 'B', 'D'), ('S', 'U'): ('D', 'S'),
           ('D', 'S'): ('S', 'U'), ('S', 'B'): ('B', 'S'), ('B', 'S'): ('S', 'B'), ('U', 'U', 'U'): ('D', 'D', 'D'),
           ('D', 'D', 'D'): ('U', 'U', 'U'), ('U', 'U', 'B'): ('B', 'D', 'D'), ('B', 'D', 'D'): ('U', 'U', 'B')}
PLURAL = {'brother': 'brothers', 'sister': 'sisters', 'sibling': 'siblings', 'cousin': 'cousins', 'son': 'sons',
          'daughter': 'daughters', 'child': 'children'}
# Hindi words a question may use instead: (moves, genders of each reached person) -> word.
HINDI = {
    (('U', 'B'), ('F', 'M')): 'mama', (('U', 'B'), ('M', 'M')): 'chacha', (('U', 'B'), ('M', 'F')): 'bua',
    (('U', 'B'), ('F', 'F')): 'mausi', (('U', 'U'), ('M', 'M')): 'dada', (('U', 'U'), ('M', 'F')): 'dadi',
    (('U', 'U'), ('F', 'M')): 'nana', (('U', 'U'), ('F', 'F')): 'nani', (('B', 'S'), ('M', 'F')): 'bhabhi',
    (('B', 'S'), ('F', 'M')): 'jija', (('B', 'D'), ('M', 'M')): 'bhatija', (('B', 'D'), ('F', 'M')): 'bhanja',
    (('B', 'D'), ('M', 'F')): 'bhatiji', (('B', 'D'), ('F', 'F')): 'bhanji', (('S', 'U'), (None, 'M')): 'sasur',
    (('S', 'U'), (None, 'F')): 'saas', (('D', 'S'), ('F', 'M')): 'damaad', (('D', 'S'), ('M', 'F')): 'bahu',
}


def word_for(moves: tuple, g, neutral=False):
    m, f, n = WORDS[moves]
    if neutral and n:
        return n
    return m if g == 'M' else f


# ---------- people and facts ----------

@dataclass(eq=False)
class Ref:
    """Someone a question mentions: a family member (node) with how the text refers to them."""
    node: int | None
    g: str | None
    kind: str = 'name'       # name | letter | desc | anon (never named in the text)
    surface: str = ''
    head: str = ''           # for descriptions: man, woman, boy, girl, lady...

    def __repr__(self):
        return self.surface or f'<{self.kind}{self.node}>'


def F(x, r, y, side=None, only=False, neg=False):
    return {'t': 'rel', 'x': x, 'r': r, 'y': y, 'side': side, 'only': only, 'neg': neg}


def G(x, g):
    return {'t': 'gender', 'x': x, 'g': g}


class Family:
    """A random family (2-4 generations of married couples and their children) as an engine World."""

    def __init__(self, rng: random.Random, size: int = 10, gens: int = 3):
        self.w, self.people, self.level = World(), [], {}
        w = self.w

        def person(g, lev):
            i = w.add(f'@p{len(self.people)}')
            w.gen[i] = g
            self.people.append(i)
            self.level[i] = lev
            return i

        top = rng.randint(0, 1)
        h, wf = person('M', top), person('F', top)
        w.marry(h, wf)
        queue = [(h, wf, top)]
        while queue and len(self.people) < size:
            fa, mo, lev = queue.pop(0)
            if lev + 1 >= gens + top:
                continue
            for _ in range(rng.choice([1, 1, 2, 2, 2, 3, 3, 4])):
                if len(self.people) >= size:
                    break
                c = person(rng.choice('MF'), lev + 1)
                w.set_parent(c, fa, 'fa'); w.set_parent(c, mo, 'mo')
                if lev + 2 < gens + top and rng.random() < 0.75 and len(self.people) < size:
                    s = person('F' if w.gen[c] == 'M' else 'M', lev + 1)
                    w.marry(c, s)
                    queue.append((c, s, lev + 1) if w.gen[c] == 'M' else (s, c, lev + 1))
                elif rng.random() < 0.3 and len(self.people) < size:
                    s = person('F' if w.gen[c] == 'M' else 'M', lev + 1)
                    w.marry(c, s)
        w.normalize()

    def g(self, i):
        return self.w.gen[i]

    def hops(self, a, b):
        """Shortest path a -> b as hops [(move, node)], a parent-then-child pair merged into a sibling hop B."""
        p = self.w.path(a, b) or []
        out, i = [], 0
        while i < len(p):
            m, _, node = p[i]
            if m == 'U' and i + 1 < len(p) and p[i + 1][0] == 'D':
                out.append(('B', p[i + 1][2]))
                i += 2
            else:
                out.append((m, node))
                i += 1
        return out

    def truth(self, a, b):
        return relation_in(self.w, a, b)


def chunks(hops, rng, p_long=0.35):
    """Groups hops into runs that have one relation word ("uncle" for parent + sibling)."""
    out, i = [], 0
    while i < len(hops):
        fits = [k for k in (3, 2, 1) if i + k <= len(hops) and tuple(m for m, _ in hops[i:i + k]) in WORDS]
        longer = [k for k in fits if k > 1]
        k = rng.choice(longer) if longer and rng.random() < p_long else 1
        out.append(hops[i:i + k])
        i += k
    return out


# ---------- names ----------

class Names:
    """Hands out distinct names in one style per question: Indian, international, mixed or letters."""

    def __init__(self, rng: random.Random, style: str | None = None):
        self.rng = rng
        self.style = style or rng.choices(['indian', 'world', 'mixed', 'letters'], [45, 20, 15, 20])[0]
        self.used = set()
        self.letters = [c for c in LETTERS]
        rng.shuffle(self.letters)
        if rng.random() < 0.5:   # exams often use a run of letters (A, B, C... or P, Q, R...)
            start = rng.choice('ADJLPST')
            run = [c for c in LETTERS if c >= start] + [c for c in LETTERS if c < start]
            self.letters = run

    def __call__(self, g):
        if self.style == 'letters':
            for c in self.letters:
                if c not in self.used:
                    self.used.add(c)
                    return c
        pools = {'indian': (INDIAN_M, INDIAN_F), 'world': (WORLD_M, WORLD_F)}
        style = self.style if self.style in pools else self.rng.choice(['indian', 'world'])
        pool = pools[style][0 if g == 'M' else 1]
        for _ in range(50):
            n = self.rng.choice(pool)
            if n not in self.used:
                self.used.add(n)
                return n
        n = self.rng.choice(SURNAMES)
        self.used.add(n)
        return n


# ---------- building one question ----------

def an(word):
    return ('an ' if word[0] in 'aeiou' else 'a ') + word


def poss(s):
    return s + "'s"


def cap(s):
    return s[:1].upper() + s[1:]


PRON = {'M': ('he', 'him', 'his'), 'F': ('she', 'her', 'her')}
HEADS = {'M': ['man', 'man', 'boy', 'gentleman'], 'F': ['woman', 'lady', 'girl', 'woman']}


class Q:
    """One question being written: sentences, and the facts each states, in order."""

    def __init__(self, rng: random.Random, fam: Family, names: Names | None = None, style: str | None = None):
        self.rng, self.fam = rng, fam
        self.names = names or Names(rng, style)
        self.refs, self.sentences, self.facts = {}, [], []
        self.gendered, self.descs = set(), []

    def ref(self, node) -> Ref:
        """The person as a name (or letter) in the text."""
        r = self.refs.get(node)
        if r is None:
            g = self.fam.g(node)
            n = self.names(g)
            r = Ref(node, g, 'letter' if len(n) == 1 else 'name', n)
            self.refs[node] = r
        return r

    def anon(self, node=None) -> Ref:
        return Ref(node, self.fam.g(node) if node is not None else None, 'anon')

    def desc(self, node, head=None) -> Ref:
        """The person as a description ("a man" first, "the man" after)."""
        g = self.fam.g(node)
        r = Ref(node, g, 'desc', head=head or self.rng.choice(HEADS[g]))
        r.said = False
        self.refs[node] = r
        return r

    def say(self, r: Ref, definite=False) -> str:
        if r.kind != 'desc':
            return r.surface
        if not r.said:
            r.said = True
            self.descs.append(r)
            return ('the ' + r.head) if definite else an(r.head)
        return 'the ' + r.head

    def two_descs(self, a, b):
        """Descriptions for two people with different nouns, so "the man" stays clear."""
        ra = self.desc(a)
        heads = [h for h in HEADS[self.fam.g(b)] if h != ra.head] + ['person']
        return ra, self.desc(b, self.rng.choice(heads[:-1] or heads))

    def add(self, sentence: str, facts=(), evidence=()):
        """A sentence; `evidence`: people whose gender it shows other than by a relation word (pronoun, noun)."""
        self.sentences.append(sentence)
        for r in evidence:
            if r not in self.gendered and r.g:
                self.gendered.add(r)
                self.facts.append(G(r, r.g))
        self.facts += list(facts)

    def finish(self, kind: str, options=None, extra=None, noise=True) -> dict | None:
        rng = self.rng
        text = ' '.join(s.strip() for s in self.sentences if s.strip())
        if noise and rng.random() < 0.12:   # real questions drop words: "the son of only son my grandfather"
            import re
            text = re.sub(r'\b(the|of) (?!man\b|woman\b|boy\b|girl\b|lady\b|gentleman\b|person\b)',
                          lambda m: '' if rng.random() < 0.25 else m.group(0), text)
        if rng.random() < 0.12:
            text = rng.choice(['Question {n}: ', 'Q{n}. ', 'Blood Relations Question {n}: ', 'Question {n}. ',
                               'Directions: Study the following information carefully and answer the question given below. ',
                               'Read the information carefully and answer the question. ']).format(n=rng.randint(1, 40)) + text
        if options:
            labels = rng.choice([['(a)', '(b)', '(c)', '(d)', '(e)'], ['a)', 'b)', 'c)', 'd)', 'e)'], ['1.', '2.', '3.', '4.', '5.'],
                                 ['A.', 'B.', 'C.', 'D.', 'E.'], ['(1)', '(2)', '(3)', '(4)', '(5)']])
            text += ' ' + ' '.join(f'{l} {o}' for l, o in zip(labels, options))
        if rng.random() < 0.02 and not any(r.kind == 'letter' for r in self.refs.values()):
            text = text.lower()
        p = prepare(text)
        by_text = {e['text'].lower(): e['id'] for e in p['ents'] if e['kind'] in ('name', 'letter')}
        descs = [e for e in p['ents'] if e['kind'] == 'desc']
        if len(descs) != len(self.descs) or any(d['text'].split()[-1].lower() != r.head for d, r in zip(descs, self.descs)):
            return None
        ids = {}
        for d, r in zip(descs, self.descs):
            ids[r] = d['id']
        n_new = 0
        out = []
        for f in self.facts:
            if f['t'] in ('claim', 'part'):
                out.append(dict(f))
                continue
            g = dict(f)
            for k in ('x', 'y'):
                r = f.get(k)
                if not isinstance(r, Ref):
                    continue
                if r not in ids:
                    if r.kind == 'anon':
                        n_new += 1
                        ids[r] = f'#{n_new}'
                    else:
                        e = by_text.get(r.surface.lower())
                        if e is None:
                            return None
                        ids[r] = e
                g[k] = ids[r]
            out.append(g)
        ex = {'kind': kind, 'text': text, 'tokens': p['tokens'], 'lf': _format(out), 'ents': p['ents'],
              'names': {ids[r]: r.surface for r in ids if r.kind in ('name', 'letter')}}
        if extra:
            ex.update(extra)
        return ex


# ---------- sentence forms ----------

def rel_phrase(word, side=None, only=False):
    return ' '.join(w for w in ('only' if only else None, side, word) if w)


AGE = {'brother': ['elder', 'younger'], 'sister': ['elder', 'younger'], 'son': ['elder', 'younger', 'eldest', 'youngest'],
       'daughter': ['elder', 'younger', 'eldest', 'youngest']}


def state(rng, x: str, y: str, word: str, side=None, only=False) -> str:
    """ "y is x's word" in one of the ways exam papers say it ("elder"/"younger" may be added: no new fact)."""
    rp = rel_phrase(word, side, only)
    if word in AGE and not only and not side and rng.random() < 0.08:
        rp = f'{rng.choice(AGE[word])} {word}'
    forms = [f'{y} is the {rp} of {x}', f'{y} is the {rp} of {x}', f"{y} is {poss(x)} {rp}", f"{poss(x)} {rp} is {y}",
             f'The {rp} of {x} is {y}']
    if not only:
        forms += [f'{x} has {an(rp)} named {y}', f'{x} has {an(rp)}, {y}']
    return rng.choice(forms)


def ask_text(rng, x: str, y: str) -> str:
    """A question asking how y is related to x."""
    return rng.choice([
        f'How is {y} related to {x}?', f'How is {y} related to {x}?', f'How is {y} related to {x}?',
        f'What is {y} to {x}?', f'How is {y} connected to {x}?', f'{cap(y)} is related to {x} as what?',
        f'What is the relation of {y} to {x}?', f'What is the relationship of {y} with {x}?',
        f'In what way is {y} related to {x}?', f'Who is {y} to {x}?', f'How is {y} related with {x}?',
        f'Find the relation of {y} with {x}.', f'Which of the following is the relation of {y} to {x}?',
        f'What is the relation between {y} and {x}?', f'{cap(y)} is {poss(x)}?', f'Then, how is {y} related to {x}?',
        f'How are {y} and {x} related?', f'What relation does {y} have with {x}?',
        f'{cap(y)} is ____ of {x}.', f'{cap(y)} is {poss(x)} ____.', f'{cap(y)} is the ____ of {x}.',   # fill in the blank
    ])


def fact_for(q: Q, hop_run, start, end, neutral=False, side_p=0.3, invert=None):
    """The fact for one run of hops from start to end, written forwards (end is start's word) or inverted
    (start is end's word). Returns (sentence parts (x, y, word, side), fact)."""
    rng, fam = q.rng, q.fam
    moves = tuple(m for m, _ in hop_run)
    invert = rng.random() < 0.35 if invert is None else invert
    if invert:
        moves2 = INVERSE[moves]
        word = word_for(moves2, fam.g(start), neutral)
        a, b = end, start
        first = fam.hops(end, start)[0][1]
        mv = moves2
    else:
        word = word_for(moves, fam.g(end), neutral)
        a, b = start, end
        first = hop_run[0][1]
        mv = moves
    side = None
    if mv[0] == 'U' and len(mv) > 1 and rng.random() < side_p:
        side = 'paternal' if fam.g(first) == 'M' else 'maternal'
    return a, b, word, side


def pick_pair(rng, fam: Family, lo=1, hi=6):
    for _ in range(40):
        a, b = rng.sample(fam.people, 2)
        n = len(fam.hops(a, b))
        if lo <= n <= hi:
            return a, b
    return None


PRONOUN_HINTS = ['{X} told {his} teacher about the family trip.', '{X} lost {his} keys at the market yesterday.',
                 '{X} says that {his} favourite subject is mathematics.', 'When {X} got the job, {he} called home first.',
                 '{X} wore {his} new watch to the wedding.', '{X} walks {his} dog every morning.',
                 '{X} keeps {his} old letters in a box.', 'Last year {X} finished {his} studies at college.']
SPOUSE_HINTS = ['{X} {w} cooks for the whole family.', '{X} {w} works at a bank.', '{X} {w} is a doctor.',
                '{X} {w} planted a mango tree in the garden.']


def gender_hint(q: Q, r: Ref):
    """A sentence giving someone's gender outright."""
    rng = q.rng
    young = r.node is not None and q.fam.level.get(r.node, 0) >= max(q.fam.level.values())
    words = {'M': ['male', 'a male', 'a man', 'a male member'], 'F': ['female', 'a female', 'a woman', 'a lady', 'a female member']}[r.g]
    if young:
        words += ['a boy'] if r.g == 'M' else ['a girl']
    pick = rng.random()
    his = {'M': 'his', 'F': 'her'}[r.g]
    sp = q.fam.w.spouse(r.node) if r.node is not None and r.kind in ('name', 'letter') else -1
    if r.kind in ('name', 'letter') and pick < 0.35:   # gender from a pronoun in an everyday sentence
        q.add(rng.choice(PRONOUN_HINTS).format(X=r.surface, his=his, he={'M': 'he', 'F': 'she'}[r.g]), [], evidence=[r])
    elif sp is not None and sp >= 0 and pick < 0.5:   # from a spouse word: "Jordan's wife cooks for the family"
        w = 'wife' if r.g == 'M' else 'husband'
        q.add(rng.choice(SPOUSE_HINTS).format(X=poss(r.surface), w=w), [F(r, w, q.anon(sp))])
    else:
        q.add(f'{q.say(r)} is {rng.choice(words)}.', [G(r, r.g)])
    q.gendered.add(r)


def implied(q: Q, r: Ref) -> bool:
    return r in q.gendered or any(f['t'] == 'rel' and f['y'] is r and KIN_GENDER.get(f['r']) for f in q.facts)


def _kin_gender():
    from .kin import KIN
    return {w: ps[0][-1].g for w, ps in KIN.items() if all(p[-1].g == ps[0][-1].g for p in ps)}


KIN_GENDER = _kin_gender()


def options_for(rng, truth_label, pool=None):
    pool = pool or ['Brother', 'Sister', 'Father', 'Mother', 'Son', 'Daughter', 'Uncle', 'Aunt', 'Nephew', 'Niece',
                    'Cousin', 'Grandfather', 'Grandmother', 'Grandson', 'Granddaughter', 'Husband', 'Wife',
                    'Father-in-law', 'Mother-in-law', 'Brother-in-law', 'Sister-in-law', 'Son-in-law',
                    'Daughter-in-law', 'Maternal uncle', 'Paternal uncle', 'Cannot be determined']
    opts = {cap(truth_label)} if truth_label else set()
    while len(opts) < 4:
        opts.add(rng.choice(pool))
    opts = list(opts)
    rng.shuffle(opts)
    if rng.random() < 0.3:
        opts.append(rng.choice(['None of these', 'Cannot be determined', 'Data inadequate']))
    return opts


# ---------- question kinds ----------

def _family(rng, lo=4, hi=12):
    return Family(rng, size=rng.randint(lo, hi), gens=rng.choice([2, 3, 3, 3, 4]))


def _sym(rng, q: Q, ra: Ref, rb: Ref, word):
    """Symmetric statements: "A and B are brothers", "A and B are a married couple"."""
    x, y = q.say(ra), q.say(rb)
    if word in ('brother', 'sister') and ra.g == rb.g == ('M' if word == 'brother' else 'F'):
        return f'{x} and {y} are {PLURAL[word]}', [F(ra, word, rb), F(rb, word, ra)]
    if word == 'sibling' or (word in ('brother', 'sister') and rng.random() < 0.5):
        return f'{x} and {y} are siblings', [F(ra, 'sibling', rb)]
    if word in ('husband', 'wife', 'spouse'):
        return rng.choice([(f'{x} is married to {y}', [F(ra, 'spouse', rb)]), (f'{x} and {y} are a married couple', [F(ra, 'spouse', rb)]),
                           (f'{x} married {y}', [F(ra, 'spouse', rb)])])
    if word == 'cousin':
        return f'{x} and {y} are cousins', [F(ra, 'cousin', rb)]
    return None


def q_statements(rng) -> dict | None:
    """ "A is the brother of B. B is the daughter of C. ... How is C related to A?" (with distractors)."""
    fam = _family(rng)
    pair = pick_pair(rng, fam, 1, 6)
    if not pair:
        return None
    a, b = pair
    q = Q(rng, fam)
    hops = fam.hops(a, b)
    runs = chunks(hops, rng)
    hide = rng.random() < 0.08
    clauses, cur = [], a
    hindi_ok = rng.random() < 0.06
    use_prose = not hindi_ok and rng.random() < 0.35   # apposition, "whose", pronouns, lists (kinlogic/prose.py)
    chain = []
    for k, run in enumerate(runs):
        end = run[-1][1]
        x, y, word, side = fact_for(q, run, cur, end, neutral=hide and rng.random() < 0.5)
        rx, ry = q.ref(x), q.ref(y)
        moves = tuple(m for m, _ in (run if (x, y) == (cur, end) else fam.hops(x, y)))
        genders = tuple(fam.g(n) for _, n in fam.hops(x, y))
        hw = HINDI.get((moves, genders)) or HINDI.get((moves, (None, genders[-1])))
        if hindi_ok and hw:
            facts = _hindi_facts(q, rx, ry, hw, fam.hops(x, y))
            clauses.append((rng.choice([f'{ry.surface} is {poss(rx.surface)} {hw}', f'{ry.surface} is the {hw} of {rx.surface}']), facts))
        elif use_prose:
            chain.append(F(rx, word, ry, side))
        else:
            sym = _sym(rng, q, rx, ry, word) if rng.random() < 0.15 else None
            clauses.append(sym or (state(rng, rx.surface, ry.surface, word, side), [F(rx, word, ry, side)]))
        cur = end
    # distractors: true facts about people near the chain
    for _ in range(rng.choice([0, 0, 0, 1, 1, 2, 3])):
        known = [r.node for r in q.refs.values()]
        n = rng.choice(known)
        nb = [m for m in fam.people if m not in known and 1 <= len(fam.hops(n, m)) <= 2]
        if not nb:
            break
        m = rng.choice(nb)
        run = fam.hops(n, m)
        if tuple(h for h, _ in run) not in WORDS:
            continue
        x, y, word, side = fact_for(q, run, n, m)
        rx, ry = q.ref(x), q.ref(y)
        clauses.append((state(rng, rx.surface, ry.surface, word, side), [F(rx, word, ry, side)]))
    w = fam.w
    for x in [r.node for r in list(q.refs.values())]:   # "A, B and C are sisters": each is the next one's sister
        sibs = [x] + [z for z in w.siblings(x) if fam.g(z) == fam.g(x)]
        if len(sibs) >= 3 and rng.random() < 0.25:
            rs_ = [q.ref(z) for z in sibs[:4]]
            word = 'brother' if fam.g(x) == 'M' else 'sister'
            clauses.append((f'{_listing(rng, [r.surface for r in rs_])} are {PLURAL[word]}',
                            [F(rs_[i], word, rs_[(i + 1) % len(rs_)]) for i in range(len(rs_))]))
            break
    for x in [r.node for r in list(q.refs.values())]:   # "X has two sons, Y and Z" / "Y and Z are the parents of X"
        if rng.random() < 0.12:
            for word in ('son', 'daughter'):
                kids = w.relatives(x, word)
                if 2 <= len(kids) <= 3:   # "has two sons" means exactly two: all are listed, and the count is a fact
                    rx, ks = q.ref(x), [q.ref(k) for k in kids]
                    clauses.append((f'{rx.surface} has {NUM[len(ks)]} {PLURAL[word]}, {_listing(rng, [k.surface for k in ks])}',
                                    [{'t': 'has', 'x': rx, 'n': len(ks), 'r': word, 'side': None}] + [F(rx, word, k) for k in ks]))
                    break
        elif rng.random() < 0.06 and w.father(x) >= 0 and w.mother(x) >= 0:
            rx, rf, rm = q.ref(x), q.ref(w.father(x)), q.ref(w.mother(x))
            kid = 'son' if fam.g(x) == 'M' else 'daughter'
            if w.spouse(w.father(x)) == w.mother(x) and rng.random() < 0.6:   # "X and Y are married, and Z is their son"
                clauses.append((rng.choice([f'{rf.surface} and {rm.surface} are married, and {rx.surface} is their {kid}',
                                            f'{rf.surface} and his wife {rm.surface} have {an(kid)}, {rx.surface}']),
                                [F(rf, 'wife', rm), F(rf, kid, rx), F(rm, kid, rx)]))
            else:
                clauses.append((rng.choice([f'{rf.surface} and {rm.surface} are the parents of {rx.surface}',
                                            f"{poss(rx.surface)} parents are {rf.surface} and {rm.surface}"]),
                                [F(rx, 'parent', rf), F(rx, 'parent', rm)]))
    if rng.random() < 0.4:
        rng.shuffle(clauses)
    clauses = _relative_clauses(rng, clauses)
    ra, rb = q.ref(a), q.ref(b)
    if rng.random() < 0.5:
        ra, rb = rb, ra
    target = rb
    style = rng.random()
    if use_prose:   # the chain in order (pronouns refer back), then the other sentences
        from .prose import add_all
        add_all(q, rng, chain, joint=lambda p1, p2, c: _joint(fam, p1, p2, c))
        for c, fs in clauses:
            q.add(cap(c) + '.', fs)
        if not implied(q, target) and rng.random() < 0.6:
            gender_hint(q, target)
        q.add(ask_text(rng, ra.surface, rb.surface), [{'t': 'ask', 'x': ra, 'y': rb}])
    elif style < 0.3 and len(clauses) <= 4:   # "If A is ...; B is ... and C is ..., how is D related to A?"
        parts = [c for c, _ in clauses]
        joined = '; '.join(parts[:-1]) + (' and ' if len(parts) > 1 else '') + parts[-1] if len(parts) > 1 else parts[0]
        facts = [f for _, fs in clauses for f in fs]
        q.add(f'If {joined}, {ask_text(rng, ra.surface, rb.surface)[0].lower() + ask_text(rng, ra.surface, rb.surface)[1:]}'
              if False else f'If {joined}, then how is {rb.surface} related to {ra.surface}?', facts + [{'t': 'ask', 'x': ra, 'y': rb}])
    else:
        for c, fs in clauses:
            q.add(cap(c) + '.', fs)
        if not implied(q, target) and rng.random() < 0.6:
            gender_hint(q, target)
        q.add(ask_text(rng, ra.surface, rb.surface), [{'t': 'ask', 'x': ra, 'y': rb}])
    t = fam.truth(ra.node, rb.node)
    return q.finish('statements', options_for(rng, t['label']) if rng.random() < 0.35 else None, {'truth': t['key']})


def _joint(fam, p1, p2, c):
    """c is the child of both p1 and p2 in the true family."""
    w = fam.w
    return None not in (p1.node, p2.node, c.node) and {w.father(c.node), w.mother(c.node)} == {p1.node, p2.node}


def _relative_clauses(rng, clauses):
    """ "A is the father of B. B is the sister of C." -> "A is the father of B, who is the sister of C." """
    out, i = [], 0
    while i < len(clauses):
        c1, f1 = clauses[i]
        if i + 1 < len(clauses) and len(f1) == 1 and len(clauses[i + 1][1]) == 1 and rng.random() < 0.3:
            c2, f2 = clauses[i + 1]
            a, b = f1[0], f2[0]
            # c1 says "a.y is the R of a.x"; c2 says "b.y is the R2 of b.x" with b.y == a.x
            if a['t'] == b['t'] == 'rel' and b['y'] is a['x'] and c1 == f"{a['y'].surface} is the {rel_phrase(a['r'], a['side'])} of {a['x'].surface}":
                out.append((f"{c1}, who is the {rel_phrase(b['r'], b['side'])} of {b['x'].surface}", f1 + f2))
                i += 2
                continue
        out.append((c1, f1))
        i += 1
    return out


def _hindi_facts(q: Q, rx: Ref, ry: Ref, word, hops):
    """Canonical facts for a Hindi kinship word: one fact when English has a word for it, else two."""
    fam = q.fam
    one = {'mama': 'uncle', 'chacha': 'uncle', 'bua': 'aunt', 'mausi': 'aunt', 'dada': 'grandfather', 'dadi': 'grandmother',
           'nana': 'grandfather', 'nani': 'grandmother', 'sasur': 'father-in-law', 'saas': 'mother-in-law',
           'damaad': 'son-in-law', 'bahu': 'daughter-in-law'}
    if word in one:
        side = None
        if word in ('mama', 'chacha', 'bua', 'mausi', 'dada', 'dadi', 'nana', 'nani'):
            side = 'paternal' if fam.g(hops[0][1]) == 'M' else 'maternal'
        return [F(rx, one[word], ry, side)]
    mid = q.anon(hops[0][1])
    first = {'bhabhi': 'brother', 'jija': 'sister', 'bhatija': 'brother', 'bhatiji': 'brother', 'bhanja': 'sister', 'bhanji': 'sister'}[word]
    second = {'bhabhi': 'wife', 'jija': 'husband', 'bhatija': 'son', 'bhatiji': 'daughter', 'bhanja': 'son', 'bhanji': 'daughter'}[word]
    return [F(rx, first, mid), F(mid, second, ry)]


PRONOUN_ANCHORS = {'my', 'his', 'her', 'your', 'their'}


def _np(rng, anchor: str, words: list[str], style=None) -> str:
    """A description of someone reached from `anchor` through `words` (innermost first):
    ["mother", "brother", "son"] -> "the son of the brother of Varun's mother" / "Varun's mother's brother's son".
    anchor may be a pronoun ("my", "his")."""
    pron = anchor in PRONOUN_ANCHORS
    own = (lambda w: f'{anchor} {w}') if pron else (lambda w: f'{poss(anchor)} {w}')
    style = style or rng.choice(['of', 'of', 'poss', 'mixed'])
    if len(words) == 1:
        return own(words[0]) if pron or rng.random() < 0.6 else f'the {words[0]} of {anchor}'
    if style == 'poss':
        return own(words[0]) + "'s " + "'s ".join(words[1:])
    k = 1 if style == 'of' else rng.randint(1, len(words) - 1)
    if style == 'of' and not pron and rng.random() < 0.4:
        inner = f'the {words[0]} of {anchor}'
    else:
        inner = own(words[0]) + ''.join(f"'s {w}" for w in words[1:k])
    for w in words[k:]:
        inner = f'the {w} of {inner}'
    return inner


def _np_facts(q: Q, anchor: Ref, runs_words, last: Ref | None):
    """Facts for a description chain from anchor: one unnamed person per inner step; the last step reaches
    `last` (or a new unnamed person when None). Returns (facts, the person described)."""
    facts, cur = [], anchor
    for k, (word, side, only, node) in enumerate(runs_words):
        nxt = last if k == len(runs_words) - 1 and last is not None else q.anon(node)
        facts.append(F(cur, word, nxt, side, only))
        cur = nxt
    return facts, cur


def _run_words(q: Q, start, runs, side_p=0.25):
    """Words for consecutive hop runs from start, with "only" when that relative is the only one."""
    fam, rng = q.fam, q.rng
    out, cur = [], start
    for run in runs:
        moves = tuple(m for m, _ in run)
        end = run[-1][1]
        word = word_for(moves, fam.g(end))
        side = None
        if moves[0] == 'U' and len(moves) > 1 and rng.random() < side_p:
            side = 'paternal' if fam.g(run[0][1]) == 'M' else 'maternal'
        only = False
        if word in ('son', 'daughter', 'brother', 'sister') and rng.random() < 0.45:
            only = len(fam.w.relatives(cur, word)) == 1
        out.append((word, side, only, end))
        cur = end
    return out


def _phrase(words):
    return [rel_phrase(w, s, o) for w, s, o, _ in words]


def q_nested(rng) -> dict | None:
    """ "Santosh is the son of the brother of Varun's mother. How is Santosh related to Varun?" and the
    first-person / description-in-question variants."""
    fam = _family(rng, 5, 12)
    pair = pick_pair(rng, fam, 2, 5)
    if not pair:
        return None
    a, b = pair
    q = Q(rng, fam)
    runs = chunks(fam.hops(a, b), rng, p_long=0.2)
    if len(runs) < 2:
        return None
    words = _run_words(q, a, runs)
    if any(n in q.refs for _, _, _, n in words[:-1]):
        return None
    mode = rng.random()
    if mode < 0.15:   # "How is the father of my mother's sister's son related to me?"
        me = q.anon(a)
        np_ = _np(rng, 'my', _phrase(words))
        facts, target = _np_facts(q, me, words, None)
        q.add(f'How is {np_} related to me?', facts + [{'t': 'ask', 'x': me, 'y': target}])
        return q.finish('nested', None, {'truth': fam.truth(a, b)['key']})
    ra = q.ref(a)
    np_ = _np(rng, ra.surface, _phrase(words))
    if mode < 0.35:   # the description is in the question: "How is Varun's mother's brother related to Ravi?"
        c = rng.choice([p for p in fam.people if p not in (a, b)] or [None])
        if c is None or not fam.hops(c, b):
            return None
        if c in [n for _, _, _, n in words]:
            return None
        for text, fs in _chain_statements(q, c, a):   # the text must connect the other person too
            q.add(cap(text) + '.', fs)
        if any(n in q.refs for _, _, _, n in words):   # the description must not reach a named person
            return None
        rc = q.ref(c)
        facts, target = _np_facts(q, ra, words, None)
        if rng.random() < 0.5:
            q.add(f'How is {np_} related to {rc.surface}?', facts + [{'t': 'ask', 'x': rc, 'y': target}])
            return q.finish('nested', None, {'truth': fam.truth(c, b)['key']})
        q.add(f'How is {rc.surface} related to {np_}?', facts + [{'t': 'ask', 'x': target, 'y': rc}])
        return q.finish('nested', None, {'truth': fam.truth(b, c)['key']})
    rb = q.ref(b)
    facts, _ = _np_facts(q, ra, words, rb)
    q.add(rng.choice([f'{rb.surface} is {np_}.', f'{cap(np_)} is {rb.surface}.']), facts)
    if not implied(q, rb) and rng.random() < 0.3:
        gender_hint(q, rb)
    if rng.random() < 0.5:
        q.add(ask_text(rng, ra.surface, rb.surface), [{'t': 'ask', 'x': ra, 'y': rb}])
        t = fam.truth(a, b)
    else:
        q.add(ask_text(rng, rb.surface, ra.surface), [{'t': 'ask', 'x': rb, 'y': ra}])
        t = fam.truth(b, a)
    return q.finish('nested', options_for(rng, t['label']) if rng.random() < 0.4 else None, {'truth': t['key']})


SPEECH_OPEN = [
    'Pointing to {P}, {S} said, {U}', 'Pointing towards {P}, {S} said, {U}', 'Pointing at {P}, {S} said, {U}',
    'Pointing to a photograph of {P}, {S} said, {U}', 'Looking at a portrait of {P}, {S} said, {U}',
    'Introducing {P}, {S} said, {U}', 'Introducing {P} to the guests, {S} said, {U}', 'Showing {P}, {S} said, {U}',
    'While introducing {P}, {S} said, {U}', '{S} said, pointing to {P}, {U}', 'Pointing to {P} in a photograph, {S} said, {U}',
]


def q_speech(rng) -> dict | None:
    """Pointing / introducing / photograph questions: the speaker's "my", the person's "he/his"."""
    fam = _family(rng, 5, 12)
    pair = pick_pair(rng, fam, 1, 5)
    if not pair:
        return None
    s, p = pair
    q = Q(rng, fam, style=rng.choices(['indian', 'world', 'mixed'], [70, 15, 15])[0])
    kinds = (rng.random() < 0.3, rng.random() < 0.65)
    if all(kinds):
        rs, rp = q.two_descs(s, p)
    else:
        rs = q.desc(s) if kinds[0] else q.ref(s)
        rp = q.desc(p) if kinds[1] else q.ref(p)
    he, him, his = PRON[fam.g(p)]
    addressee = rng.random() < 0.2   # the speaker talks to the person: "you", "your"
    if addressee:
        he, his = 'you', 'your'
    elif rp.kind == 'desc' and rng.random() < 0.25:
        his = f'this {rp.head}' + "'s"   # "This man's son's sister is my mother-in-law"
    hops = fam.hops(s, p)
    quote = rng.choice([('"', '"'), ("'", "'"), ('“', '”')])
    form = rng.random()
    facts, ev = [], [rp] if rp.kind == 'desc' else []
    if form < 0.45 or len(hops) < 2:   # "He is the son of my father's only brother."
        words = _run_words(q, s, chunks(hops, rng, 0.3))
        np_ = _np(rng, 'my', _phrase(words))
        utter = f"{cap(he)} {'are' if addressee else 'is'} {np_}."
        f2, _ = _np_facts(q, rs, words, rp)
        facts += f2
        if not addressee:
            ev.append(rp)
    else:   # "His mother is the only daughter of my father." (meet in the middle)
        k = rng.randint(1, len(hops) - 1)
        m = hops[k - 1][1]
        s_runs = chunks(fam.hops(s, m), rng, 0.3)
        p_runs = chunks(fam.hops(p, m), rng, 0.3)
        if not p_runs or not s_runs:
            return None
        pw = _run_words(q, p, p_runs, 0)
        sw = _run_words(q, s, s_runs)
        subj = (his[:-2] + "'s " if his.endswith("'s") else his + ' ') + "'s ".join(_phrase(pw))
        f_subj, head = _np_facts(q, rp, pw, None)
        pred = _np(rng, 'my', _phrase(sw))
        f_pred, _ = _np_facts(q, rs, sw, head)
        utter = f'{cap(subj)} is {pred}.'
        facts += f_subj + f_pred
        if not addressee and not his.startswith('this'):
            ev.append(rp)
    open_ = rng.choice(SPEECH_OPEN)
    if addressee:
        open_ = rng.choice(['{S} said to {P}, {U}', '{S} told {P}, {U}', 'Looking at {P}, {S} said, {U}'])
    side = []   # someone the speaker talks to or introduces to: "said to her husband"
    if not addressee and rng.random() < 0.2:
        _, _, sh = PRON[fam.g(s)]
        rel = [(m, mv) for m in fam.people if m not in (s, p) and len(mv := fam.hops(s, m)) == 1 and mv[0][0] in ('S', 'D', 'U')]
        if rel and rng.random() < 0.7:
            m, mv = rng.choice(rel)
            word = word_for(('S',) if mv[0][0] == 'S' else (mv[0][0],), fam.g(m))
            side = [F(rs, word, q.anon(m))]
        else:
            word = rng.choice(['friend', 'neighbour', 'colleague'])
            side = [F(rs, word, Ref(None, None, 'anon'))]
        if '{S} said' in open_:
            open_ = open_.replace('{S} said', '{S} said to ' + sh + ' ' + word)
        elif open_.startswith('Introducing {P},'):
            open_ = open_.replace('Introducing {P},', 'Introducing {P} to ' + sh + ' ' + word + ',')
        else:
            side = []
        if side:
            q.gendered.discard(rs)
    bare_photo = rp.kind == 'desc' and not addressee and not side and rng.random() < 0.2
    if bare_photo:   # the person is first named in the question: "the man in the photograph"
        open_ = rng.choice(['Pointing to a photograph, {S} said, {U}', 'Looking at a photograph, {S} said, {U}',
                            'Showing a photograph to a friend, {S} said, {U}'])
        ev = [x for x in ev if x is not rp]
    order = [rs, rp] if bare_photo or open_.index('{S}') < open_.index('{P}') else [rp, rs]
    said = {r: q.say(r) for r in order if r.kind == 'desc' and not (bare_photo and r is rp)}
    if rs.kind == 'desc':
        ev.append(rs)
    ev = sorted(set(ev), key=order.index)   # gender facts in the order the text mentions people
    if side and rs not in ev:
        ev.append(rs)   # "her husband": the speaker's gender
        ev = sorted(set(ev), key=order.index)
    sentence = open_.format(P=said.get(rp, rp.surface), S=said.get(rs, rs.surface), U=quote[0] + utter + quote[1])
    q.add(cap(sentence), side + (facts if not bare_photo else []), evidence=ev)
    if bare_photo:
        ref_p = q.say(rp, definite=True) + ' in the photograph'
    else:
        ref_p = q.say(rp) if rp.kind == 'desc' else rp.surface
        if rp.kind == 'desc' and 'photograph' in open_ and rng.random() < 0.5:
            ref_p = f'{ref_p} in the photograph'
    ref_s = q.say(rs) if rs.kind == 'desc' else rs.surface
    if bare_photo:   # facts come from the quote, which the reader links to the question's "the man"
        q.facts += [G(rp, rp.g)] + facts
        q.gendered.add(rp)
    near = [m for m in fam.people if m not in (s, p) and len(fam.hops(s, m)) == 1 and m not in q.refs]
    if near and not bare_photo and rng.random() < 0.15:   # "How is Tanya's husband related to the man?"
        m = rng.choice(near)
        mv = tuple(h for h, _ in fam.hops(s, m))
        word = word_for(mv, fam.g(m))
        rm = q.anon(m)
        q.add(ask_text(rng, ref_p, f"{poss(ref_s)} {word}"), [F(rs, word, rm), {'t': 'ask', 'x': rp, 'y': rm}])
        t = fam.truth(p, m)
    elif rng.random() < 0.65:
        q.add(ask_text(rng, ref_s, ref_p), [{'t': 'ask', 'x': rs, 'y': rp}])
        t = fam.truth(s, p)
    else:
        q.add(ask_text(rng, ref_p, ref_s), [{'t': 'ask', 'x': rp, 'y': rs}])
        t = fam.truth(p, s)
    return q.finish('speech', options_for(rng, t['label']) if rng.random() < 0.4 else None, {'truth': t['key']})


def q_introduce(rng) -> dict | None:
    """ "A woman introduces a man as the son of the brother of her mother." (reported, with her/his)"""
    fam = _family(rng, 5, 12)
    pair = pick_pair(rng, fam, 1, 5)
    if not pair:
        return None
    s, p = pair
    q = Q(rng, fam)
    kinds = (rng.random() < 0.6, rng.random() < 0.6)
    if all(kinds):
        rs, rp = q.two_descs(s, p)
    else:
        rs = q.desc(s) if kinds[0] else q.ref(s)
        rp = q.desc(p) if kinds[1] else q.ref(p)
    _, _, his = PRON[fam.g(s)]
    words = _run_words(q, s, chunks(fam.hops(s, p), rng, 0.3))
    np_ = _np(rng, his, _phrase(words))
    facts, _ = _np_facts(q, rs, words, rp)
    verb = rng.choice(['introduces', 'introduced'])
    q.add(cap(f'{q.say(rs)} {verb} {q.say(rp)} as {np_}.'), facts, evidence=[x for x in (rs, rp) if x.kind == 'desc'] + [rs])
    ref_s, ref_p = q.say(rs), q.say(rp)
    if rng.random() < 0.6:
        q.add(ask_text(rng, ref_s, ref_p), [{'t': 'ask', 'x': rs, 'y': rp}])
        t = fam.truth(s, p)
    else:
        q.add(ask_text(rng, ref_p, ref_s), [{'t': 'ask', 'x': rp, 'y': rs}])
        t = fam.truth(p, s)
    return q.finish('introduce', options_for(rng, t['label']) if rng.random() < 0.4 else None, {'truth': t['key']})


SYMBOLS = ['+', '-', '×', '÷', '*', '@', '#', '$', '%', '&', '!', '^', '=', '/']
CODE_WORDS = ['father', 'mother', 'son', 'daughter', 'brother', 'sister', 'husband', 'wife', 'parent', 'child', 'sibling',
              'spouse', 'father-in-law', 'mother-in-law', 'son-in-law', 'daughter-in-law', 'brother-in-law',
              'sister-in-law', 'grandfather', 'grandmother', 'grandson', 'granddaughter', 'uncle', 'aunt', 'nephew', 'niece']


def q_coded(rng) -> dict | None:
    """ "If A + B means A is the brother of B, ... how is P related to S in P + Q - R × S?" and the
    in-law codes of bank mains ("B#R!P*S&H: how is R's husband related to H?")."""
    fam = _family(rng, 6, 12)
    pair = pick_pair(rng, fam, 2, 7)
    if not pair:
        return None
    a, b = pair
    runs = chunks(fam.hops(a, b), rng, p_long=0.3 if rng.random() < 0.5 else 0)
    if len(runs) < 2 or len(runs) > 5:
        return None
    seq = [a] + [r[-1][1] for r in runs]
    neutral = rng.random() < 0.35
    # "X op Y" means X is the R of Y: R is the word for the path from Y to X
    words = []
    for x, y in zip(seq, seq[1:]):
        moves = tuple(m for m, _ in fam.hops(y, x))
        if moves not in WORDS:
            return None
        words.append(word_for(moves, fam.g(x), neutral and rng.random() < 0.6))
    table = list(dict.fromkeys(words))
    while len(table) < rng.randint(max(len(table), 3), 6):
        w = rng.choice(CODE_WORDS)
        if w not in table:
            table.append(w)
    rng.shuffle(table)
    syms = rng.sample(SYMBOLS, len(table))
    code = dict(zip(table, syms))
    q = Q(rng, fam, Names(rng, 'letters'))
    va, vb = rng.choice([('A', 'B'), ('P', 'Q'), ('X', 'Y'), ('M', 'N'), ('A', 'B')])
    q.names.used |= {va, vb}
    defs = []
    for w in table:
        sym = code[w]
        form = rng.choice([f"'{va} {sym} {vb}' means '{va} is the {w} of {vb}'", f'{va} {sym} {vb} means {va} is the {w} of {vb}',
                           f"{va} {sym} {vb} means {va} is {poss(vb)} {w}", f'{va}{sym}{vb} means {va} is the {w} of {vb}',
                           f"'{va} {sym} {vb}' means '{va} is {poss(vb)} {w}'"])
        defs.append(form)
    joiner = rng.choice(['; ', ', ', '. '])
    intro = rng.choice(['If ', 'In a certain code language, ', 'Given that ', '', 'Read the following: '])
    body = joiner.join(defs[:-1]) + (rng.choice([' and ', joiner]) + defs[-1] if len(defs) > 1 else defs[0])
    q.add(cap(intro + body) + '.')
    refs = [q.ref(n) for n in seq]
    facts = [F(refs[i + 1], words[i], refs[i]) for i in range(len(words))]
    expr_sep = rng.choice([' ', ''])
    expr = refs[0].surface + ''.join(f'{expr_sep}{code[w]}{expr_sep}{r.surface}' for w, r in zip(words, refs[1:]))
    x, y = (refs[0], refs[-1]) if rng.random() < 0.5 else (refs[-1], refs[0])
    mode = rng.random()
    if mode < 0.15:   # which expression shows ...? (the options are expressions; the claim is the target)
        t = fam.truth(x.node, y.node)
        side = t.get('side') if rng.random() < 0.5 else None
        if t['key'] == 'path' or t['base'] not in KIN_GENDER and t['base'] not in ('cousin',):
            return None
        q.add(f'Which of the following expressions shows that {y.surface} is the {rel_phrase(t["base"], side)} of {x.surface}?',
              [{'t': 'claim', 'n': 1}, F(x, t['base'], y, side)])
        return q.finish('coded', None, {'truth': None})
    if mode < 0.35 and len(seq) >= 2:   # relative of someone in the expression: "how is R's husband related to H?"
        c = rng.choice(seq[1:-1] or seq)
        nb = [m for m in fam.people if m not in seq and len(fam.hops(c, m)) == 1]
        if not nb:
            return None
        m = rng.choice(nb)
        mv = tuple(h for h, _ in fam.hops(c, m))
        w2 = word_for(mv, fam.g(m))
        rc, anon = q.ref(c), q.anon(m)
        other = rng.choice([r for r in refs if r is not rc])
        q.add(f'If {expr}, then how is {poss(rc.surface)} {w2} related to {other.surface}?',
              facts + [F(rc, w2, anon), {'t': 'ask', 'x': other, 'y': anon}])
        return q.finish('coded', None, {'truth': fam.truth(other.node, m)['key']})
    form = rng.choice([f'If {expr}, then how is {y.surface} related to {x.surface}?',
                       f"In the expression '{expr}', how is {y.surface} related to {x.surface}?",
                       f'If the expression is {expr}, how is {y.surface} related to {x.surface}?',
                       f'What does {expr} mean? How is {y.surface} related to {x.surface}?',
                       f'Which of the following is true about {expr}? How is {y.surface} related to {x.surface}?'])
    q.add(form, facts + [{'t': 'ask', 'x': x, 'y': y}])
    t = fam.truth(x.node, y.node)
    return q.finish('coded', options_for(rng, t['label']) if rng.random() < 0.4 else None, {'truth': t['key']})


def _format(stmts):
    """Fact text with group markers ("part 1", "claim 2") where the generator placed them."""
    from .facts import format_stmt
    return ' ; '.join(f"{s['t']} {s['n']}" if s['t'] in ('claim', 'part') else format_stmt(s) for s in stmts)


NUM = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']


def _num(rng, n):
    return NUM[n] if n < len(NUM) and rng.random() < 0.75 else str(n)


def _listing(rng, names):
    if len(names) == 2:   # never "Jeff, and Liam"
        return f'{names[0]} and {names[1]}'
    return ', '.join(names[:-1]) + rng.choice([' and ', ', and ']) + names[-1] if len(names) > 1 else names[0]


def _clue(q: Q, x, y):
    """A true fact between two family members with one relation word (None if they are not that close)."""
    run = q.fam.hops(x, y)
    if not run or tuple(m for m, _ in run) not in WORDS:
        return None
    a, b, word, side = fact_for(q, run, x, y)
    ra, rb = q.ref(a), q.ref(b)
    return state(q.rng, ra.surface, rb.surface, word, side), [F(ra, word, rb, side)]


def q_puzzle(rng) -> dict | None:
    """Closed-family puzzles: "A family of eight members A to H has three married couples... Who is the mother of U?" """
    fam = Family(rng, size=rng.randint(5, 10), gens=rng.choice([2, 3, 3]))
    q = Q(rng, fam, Names(rng, 'letters' if rng.random() < 0.8 else None))
    order = fam.people[:]
    rng.shuffle(order)
    refs = [q.ref(m) for m in order]
    n = len(refs)
    w = fam.w
    couples = sum(1 for m in fam.people if w.spouse(m) > m) if True else 0
    levels = len(set(fam.level.values()))
    males = sum(1 for m in fam.people if fam.g(m) == 'M')
    head = [{'t': 'members', 'n': n}]
    extra = []
    if rng.random() < 0.6:
        extra.append((f'{_num(rng, couples)} married couple{"s" if couples != 1 else ""}', {'t': 'couples', 'n': couples}))
    if rng.random() < 0.4:
        extra.append((f'{_num(rng, levels)} generations', {'t': 'generations', 'n': levels}))
    if rng.random() < 0.25:
        extra.append((f'{_num(rng, males)} male members', {'t': 'males', 'n': males}))
        if rng.random() < 0.6:
            extra.append((f'{_num(rng, n - males)} female members', {'t': 'females', 'n': n - males}))
    names = [r.surface for r in refs]
    lst = _listing(rng, names) if rng.random() < 0.85 or q.names.style != 'letters' else f'{names[0]} to {names[-1]}'
    intro = rng.choice([f'There are {_num(rng, n)} members {lst} in a family', f'A family consists of {_num(rng, n)} members {lst}',
                        f'{lst} are {_num(rng, n)} members of a family', f'In a family of {_num(rng, n)} members, {lst}',
                        f'{cap(_num(rng, n))} persons {lst} belong to a family'])
    if extra:
        intro += rng.choice([' with ', '. There are ', ' and there are ']) + _listing(rng, [e for e, _ in extra])
    q.add(cap(intro) + '.', head + [f for _, f in extra])
    # clues: each member linked to someone already placed
    placed, clues = [order[0]], []
    for m in order[1:]:
        cands = sorted((p for p in placed if 1 <= len(fam.hops(p, m)) <= 3), key=lambda p: (len(fam.hops(p, m)), rng.random()))
        for p in cands:
            c = _clue(q, p, m)
            if c:
                clues.append(c)
                break
        else:
            return None   # someone the clues could not link: a real puzzle connects everyone
        placed.append(m)
    by = {r.node: r for r in refs}
    for _ in range(rng.choice([0, 1, 1, 2, 3])):   # extra clue kinds
        kind = rng.choice(['gender', 'gender', 'not', 'only', 'has', 'unmarried', 'opposite'])
        x = rng.choice(order)
        rx = by[x]
        if kind == 'gender':
            word = rng.choice({'M': ['male', 'a male', 'a male member', 'a man'], 'F': ['female', 'a female', 'a female member', 'a lady']}[rx.g])
            clues.append((f'{rx.surface} is {word}', [G(rx, rx.g)]))
        elif kind == 'not':
            y = rng.choice(order)
            run = fam.hops(y, x)
            if y == x or not run or tuple(h for h, _ in run) not in WORDS:
                continue
            mv = tuple(h for h, _ in run)
            wrong = WORDS[mv][1 if rx.g == 'M' else 0]
            clues.append((rng.choice([f'{rx.surface} is not the {wrong} of {by[y].surface}', f'{rx.surface} is not {poss(by[y].surface)} {wrong}']),
                          [F(by[y], wrong, rx, neg=True)]))
        elif kind == 'only':
            for word in ('son', 'daughter'):
                kids = w.relatives(x, word)
                if len(kids) == 1:
                    clues.append((f'{by[kids[0]].surface} is the only {word} of {rx.surface}', [F(rx, word, by[kids[0]], only=True)]))
                    break
        elif kind == 'has':
            kids = w.children(x)
            if kids and rng.random() < 0.5:
                clues.append((rng.choice([f'{rx.surface} has {_num(rng, len(kids))} child{"ren" if len(kids) != 1 else ""}',
                                          f'{rx.surface} is the {"father" if rx.g == "M" else "mother"} of {_num(rng, len(kids))} child{"ren" if len(kids) != 1 else ""}']),
                              [G(rx, rx.g)] * ('the ' in '' or False) + [{'t': 'has', 'x': rx, 'n': len(kids), 'r': 'child', 'side': None}]))
            elif kids:
                sons = [k for k in kids if fam.g(k) == 'M']
                if not sons:
                    clues.append((f'{rx.surface} has only daughters', [{'t': 'has', 'x': rx, 'n': 0, 'r': 'son', 'side': None}]))
                elif len(sons) == len(kids):
                    clues.append((f'{rx.surface} has only sons', [{'t': 'has', 'x': rx, 'n': 0, 'r': 'daughter', 'side': None}]))
        elif kind == 'unmarried' and w.spouse(x) < 0 and not w.children(x):
            clues.append((f'{rx.surface} is unmarried', [{'t': 'attr', 'x': rx, 'a': 'unmarried'}]))
        elif kind == 'opposite':
            y = rng.choice(order)
            if y != x:
                same = fam.g(x) == fam.g(y)
                a, b, word = rx.surface, by[y].surface, 'same' if same else 'opposite'
                clues.append((rng.choice([f'{a} and {b} are of the {word} gender', f"{poss(a)} and {poss(b)} gender is {word}",
                                          f'{a} and {b} are of {word} gender', f"{a}'s gender is the {word} as {b}'s" if same else f'{a} and {b} have opposite genders']),
                              [{'t': 'same' if same else 'opposite', 'x': rx, 'y': by[y]}]))
    if rng.random() < 0.5:
        rng.shuffle(clues)
    if not any(w.children(m) and w.spouse(m) < 0 for m in fam.people) and rng.random() < 0.3:   # real puzzles say this a lot
        clues.insert(0, (rng.choice(['There is no single parent in the family', 'No single person in the family is a parent',
                                     'None of the members is a single parent', 'There are no single parents in the family',
                                     'No one in the family is a single parent']), [{'t': 'singles', 'n': 0}]))
    i = 0
    while i < len(clues):   # sometimes two clues share a sentence
        if i + 1 < len(clues) and rng.random() < 0.35:
            (c1, f1), (c2, f2) = clues[i], clues[i + 1]
            last = c1.split()[-1]
            if c2.startswith(last + ' is ') and rng.random() < 0.6:   # "N is the mother-in-law of G, who is the sister of I"
                q.add(cap(c1) + ', who ' + c2[len(last) + 1:] + '.', f1 + f2)
                i += 2
                continue
            c2 = c2[0].lower() + c2[1:] if c2.startswith('The ') else c2
            q.add(cap(c1) + rng.choice([' and ', '; ', ', while ']) + c2 + '.', f1 + f2)
            i += 2
        else:
            q.add(cap(clues[i][0]) + '.', clues[i][1])
            i += 1
    truth = _puzzle_question(q, order, by)
    return q.finish('puzzle', None, {'truth': truth})


def _puzzle_question(q: Q, order, by):
    rng, fam, w = q.rng, q.fam, q.fam.w
    kind = rng.choice(['ask', 'ask', 'who', 'who', 'count', 'gender', 'pair'])
    x, y = rng.sample(order, 2)
    rx, ry = by[x], by[y]
    if kind == 'who':
        word = rng.choice(['father', 'mother', 'son', 'daughter', 'brother', 'sister', 'husband', 'wife', 'grandfather',
                           'grandmother', 'uncle', 'aunt', 'son-in-law', 'daughter-in-law', 'father-in-law', 'mother-in-law'])
        if word in ('husband', 'wife') and rng.random() < 0.4:
            word = 'spouse'
            q.add(f'{rx.surface} is married to whom?', [{'t': 'who', 'x': rx, 'r': 'spouse', 'side': None}])
        else:
            q.add(rng.choice([f'Who is the {word} of {rx.surface}?', f'Who among the following is the {word} of {rx.surface}?',
                              f'Who is {poss(rx.surface)} {word}?']), [{'t': 'who', 'x': rx, 'r': word, 'side': None}])
        return {'who': sorted(by[m].surface for m in w.relatives(x, word))}
    elif kind == 'count':
        c = rng.choice(['male', 'female', 'child', 'son', 'daughter', 'couples'])
        if c in ('male', 'female'):
            q.add(f'How many {c} members are there in the family?', [{'t': 'count', 'what': c}])
            return {'count': sum(1 for m in order if fam.g(m) == ('M' if c == 'male' else 'F'))}
        if c == 'couples':
            q.add('How many married couples are there in the family?', [{'t': 'count', 'what': 'couples'}])
            return {'count': sum(1 for m in order if w.spouse(m) > m)}
        q.add(f'How many {PLURAL[c]} does {rx.surface} have?', [{'t': 'count', 'what': 'rel', 'x': rx, 'r': c, 'side': None}])
        return {'count': len(w.relatives(x, c))}
    elif kind == 'gender':
        q.add(rng.choice([f'What is the gender of {rx.surface}?', f'Is {rx.surface} male or female?']), [{'t': 'gender?', 'x': rx}])
        return {'gender': 'male' if fam.g(x) == 'M' else 'female'}
    elif kind == 'pair':
        what = rng.choice(['female', 'male', 'couple', 'brother', 'sister'])
        text = {'female': 'Which of the following is a pair of females?', 'male': 'Which of the following is a pair of males?',
                'couple': 'Which of the following is a married couple?', 'brother': 'Which of the following pairs are brothers?',
                'sister': 'Which of the following pairs are sisters?'}[what]
        q.add(text, [{'t': 'pair', 'what': what}])
        return None
    q.add(ask_text(rng, rx.surface, ry.surface), [{'t': 'ask', 'x': rx, 'y': ry}])
    return fam.truth(x, y)['key']


def _chain_statements(q: Q, a, b, p_long=0.3):
    """Statements (text, facts) linking a to b through named people."""
    rng, fam = q.rng, q.fam
    out, cur = [], a
    for run in chunks(fam.hops(a, b), rng, p_long):
        end = run[-1][1]
        x, y, word, side = fact_for(q, run, cur, end)
        rx, ry = q.ref(x), q.ref(y)
        out.append((state(rng, rx.surface, ry.surface, word, side), [F(rx, word, ry, side)]))
        cur = end
    return out


def q_sufficiency(rng) -> dict | None:
    """Data sufficiency: a question and statements I and II (or CSAT's numbered statements 1-4)."""
    fam = _family(rng, 5, 11)
    pair = pick_pair(rng, fam, 2, 5)
    if not pair:
        return None
    a, b = pair
    q = Q(rng, fam)
    stmts = _chain_statements(q, a, b)
    if len(stmts) < 2:
        return None
    k = rng.choice([2, 2, 2, 3, 4]) if len(stmts) >= 3 else 2
    groups = [[] for _ in range(k)]
    for i, s in enumerate(stmts):
        groups[i % k if rng.random() < 0.7 else rng.randrange(k)].append(s)
    groups = [g for g in groups if g]
    ra, rb = q.ref(a), q.ref(b)
    ask = {'t': 'ask', 'x': ra, 'y': rb}
    if len(groups) == 2 and rng.random() < 0.8:
        labels = rng.choice([['I.', 'II.'], ['Statement I:', 'Statement II:'], ['(I)', '(II)'], ['I:', 'II:']])
        q.add(rng.choice(['Question: ', '']) + ask_text(rng, ra.surface, rb.surface) + rng.choice([' Statements:', '']), [ask])
        opts = ['Only I is sufficient', 'Only II is sufficient', 'Either I or II is sufficient', 'Both I and II together are sufficient',
                'Neither I nor II is sufficient']
    else:
        labels = [f'{i + 1}.' for i in range(len(groups))]
        q.add(rng.choice([f'What is the relation of {rb.surface} to {ra.surface}?', ask_text(rng, ra.surface, rb.surface)]), [ask])
        opts = ['Statements 1 and 2 are sufficient', 'Statements 1, 2 and 3 are sufficient', 'All the statements together are sufficient',
                'All the statements together are not sufficient']
    for n, (lab, g) in enumerate(zip(labels, groups), 1):
        q.add(f'{lab} ' + '. '.join(cap(t) for t, _ in g) + '.', [{'t': 'part', 'n': n}] + [f for _, fs in g for f in fs])
    return q.finish('sufficiency', opts if rng.random() < 0.6 else None, {'truth': fam.truth(a, b)['key']})


ROMAN = ['I', 'II', 'III', 'IV', 'V']


def q_claims(rng) -> dict | None:
    """ "Which of the following statements is definitely true? I. ... II. ..." (claims are checked, not assumed)."""
    fam = _family(rng, 5, 10)
    pair = pick_pair(rng, fam, 2, 4)
    if not pair:
        return None
    a, b = pair
    q = Q(rng, fam)
    for t, fs in _chain_statements(q, a, b):
        q.add(cap(t) + '.', fs)
    people = [r.node for r in list(q.refs.values())]
    claims = []
    for _ in range(rng.randint(2, 5)):
        x, y = rng.sample(people, 2)
        c = _clue(q, x, y)
        if not c:
            continue
        text, facts = c
        if rng.random() < 0.45:   # make it false: the other gender's word
            f = facts[0]
            other = {v: k for k, v in [('father', 'mother'), ('son', 'daughter'), ('brother', 'sister'), ('husband', 'wife'),
                                        ('uncle', 'aunt'), ('nephew', 'niece'), ('grandfather', 'grandmother'), ('grandson', 'granddaughter')]}
            other.update({k: v for v, k in other.items()})
            if f['r'] in other:
                f = {**f, 'r': other[f['r']]}
                text = state(rng, f['x'].surface, f['y'].surface, f['r'], f['side'])
                facts = [f]
        claims.append((text, facts))
    if len(claims) < 2:
        return None
    q.add(rng.choice(['Which of the following statements is definitely true?', 'How many of the following statements are definitely true?',
                      'Which of the following is/are true?', 'Which of the following statements is false?',
                      'Which of the following can be concluded?', 'Which of the following statements cannot be definitely true?']))
    for n, (t, fs) in enumerate(claims, 1):
        q.add(f'{ROMAN[n - 1]}. {cap(t)}.', [{'t': 'claim', 'n': n}] + fs)
    opts = rng.sample(['Only I', 'Only II', 'Only I and II', 'Only II and III', 'All of these', 'None of these', 'Only III', 'One', 'Two', 'Three'], 4)
    return q.finish('claims', opts if rng.random() < 0.6 else None, {'truth': None})


STORY = [
    '{X} took {his} {r} {Y} to the park.', '{X} and {his} {r} {Y} went shopping on Sunday.', '{X} called {his} {r} {Y} last night.',
    '{X} bought a gift for {his} {r} {Y}.', '{X} cooked dinner for {his} {r} {Y}.', '{X} went to the movies with {his} {r} {Y}.',
    '{Y} is the {r} of {X} and they live together.', '{X} visited {his} {r} {Y} in the hospital.', "{X}'s {r} {Y} loves to sing.",
    '{X} and {his} {r}, {Y}, played chess all evening.', '{X} drove {his} {r} {Y} to school.', '{X} was proud of {his} {r} {Y}.',
    '{X} helped {his} {r} {Y} with homework.', '{Y}, the {r} of {X}, baked a cake.', '{X} celebrated the birthday of {his} {r} {Y}.',
    '{X} asked {his} {r} {Y} for advice.', '{X} spent the weekend with {his} {r} {Y}.', '{X} wrote a letter to {his} {r} {Y}.',
]
FRIENDLY = ['{X} and {his} friend {Y} went fishing.', '{X} met {his} colleague {Y} at the office.',
            '{X} played tennis with {his} neighbour {Y}.', '{X} studied with {his} classmate {Y}.']
PLACES = ['Paris', 'London', 'Delhi', 'Mumbai', 'Goa', 'Shimla', 'Jaipur', 'Chennai', 'Tokyo', 'Kerala', 'Dubai', 'Agra',
          'Kolkata', 'Bangalore', 'Manali', 'Ooty', 'Pune', 'Rome', 'Sydney', 'Boston']
TRAVEL = ['{X} visited {P} last summer.', '{X} flew to {P} for work.', 'The family went to {P} on vacation.', '{X} grew up in {P}.']


def q_story(rng) -> dict | None:
    """Everyday stories (like CLUTRR): relations mentioned in passing, friends and places as distractions."""
    fam = _family(rng, 4, 12)
    pair = pick_pair(rng, fam, 2, 6)
    if not pair:
        return None
    a, b = pair
    q = Q(rng, fam, Names(rng, rng.choice(['indian', 'world', 'mixed'])))
    lines, cur = [], a
    for run in chunks(fam.hops(a, b), rng, 0.3):
        end = run[-1][1]
        x, y, word, side = fact_for(q, run, cur, end, side_p=0.1)
        rx, ry = q.ref(x), q.ref(y)
        tpl = rng.choice(STORY)
        lines.append((tpl.format(X=rx.surface, Y=ry.surface, r=rel_phrase(word, side), his=PRON[rx.g][2]),
                      [F(rx, word, ry, side)], [rx] if '{his}' in tpl else []))   # gender only when a pronoun states it
        cur = end
    for _ in range(rng.choice([0, 1, 1, 2])):
        kind = rng.random()
        if kind < 0.4:
            n = rng.choice(list(q.refs.values()))
            friend = Ref(None, rng.choice('MF'), 'name', q.names(rng.choice('MF')))
            tpl = rng.choice(FRIENDLY)
            word = 'friend' if 'friend' in tpl else 'colleague' if 'colleague' in tpl else 'neighbour' if 'neighbour' in tpl else 'classmate'
            lines.append((tpl.format(X=n.surface, Y=friend.surface, his=PRON[n.g][2]), [F(n, word, friend)], [n]))
        else:
            n = rng.choice(list(q.refs.values()))
            lines.append((rng.choice(TRAVEL).format(X=n.surface, P=rng.choice(PLACES)), [], []))
    rng.shuffle(lines) if rng.random() < 0.3 else None
    for text, facts, ev in lines:
        q.add(text, facts, evidence=ev)
    ra, rb = q.ref(a), q.ref(b)
    if rng.random() < 0.5:
        ra, rb = rb, ra
    q.add(ask_text(rng, ra.surface, rb.surface), [{'t': 'ask', 'x': ra, 'y': rb}])
    return q.finish('story', None, {'truth': fam.truth(ra.node, rb.node)['key']})


def q_identity(rng) -> dict | None:
    """ "His father is the only son of my father." / "...my father's son. I have no brother or sister.":
    the description loops back to the speaker."""
    fam = _family(rng, 5, 12)
    w = fam.w
    speakers = []
    for s in fam.people:
        par = w.father(s) if rng.random() < 0.6 else w.mother(s)
        if par < 0:
            continue
        word = 'son' if fam.g(s) == 'M' else 'daughter'
        if len(w.relatives(par, word)) == 1:
            speakers.append((s, par, word))
    if not speakers:
        return None
    s, par, word = rng.choice(speakers)
    near = [p for p in fam.people if p != s and 1 <= len(fam.hops(p, s)) <= 3 and p != par]
    if not near:
        return None
    p = rng.choice(near)
    q = Q(rng, fam, style=rng.choice(['indian', 'indian', 'world', 'mixed']))
    rs = q.ref(s) if rng.random() < 0.6 else q.desc(s)
    rp = q.desc(p) if rng.random() < 0.6 else q.ref(p)
    if rs.kind == rp.kind == 'desc' and rs.head == rp.head:
        return None
    pw = _run_words(q, p, chunks(fam.hops(p, s), rng, 0.3), 0)
    if any(n in q.refs for _, _, _, n in pw[:-1]):
        return None
    he, _, his = PRON[fam.g(p)]
    pword = 'father' if fam.g(par) == 'M' else 'mother'
    subj = his + ' ' + "'s ".join(_phrase(pw))
    f_subj, head = _np_facts(q, rp, pw, None)
    mid = q.anon(par)
    nosib = rng.random() < 0.3 and not w.siblings(s)
    if nosib:
        pred = rng.choice([f'my {pword}\'s {word}', f'the {word} of my {pword}'])
        tail = ' ' + rng.choice(['I have no brother or sister.', 'I have no siblings.', 'I am the only child of my parents.'])
        f_pred = [F(rs, pword, mid), F(mid, word, head)]
    else:
        pred = rng.choice([f'the only {word} of my {pword}', f'my {pword}\'s only {word}'])
        tail = ''
        f_pred = [F(rs, pword, mid), F(mid, word, head, only=True)]
    quote = rng.choice([('"', '"'), ("'", "'")])
    open_ = rng.choice(SPEECH_OPEN)
    order = [rs, rp] if open_.index('{S}') < open_.index('{P}') else [rp, rs]
    said = {r: q.say(r) for r in order if r.kind == 'desc'}
    ev = sorted({r for r in (rs, rp) if r.kind == 'desc'} | {rp}, key=order.index)
    utter = f'{cap(subj)} is {pred}.{tail}'
    facts = f_subj + f_pred + ([{'t': 'attr', 'x': rs, 'a': 'no-sibling'}] if nosib else [])
    q.add(cap(open_.format(P=said.get(rp, rp.surface), S=said.get(rs, rs.surface), U=quote[0] + utter + quote[1])), facts, evidence=ev)
    ref_s, ref_p = q.say(rs), q.say(rp)
    if rng.random() < 0.6:
        q.add(ask_text(rng, ref_s, ref_p), [{'t': 'ask', 'x': rs, 'y': rp}])
        t = fam.truth(s, p)
    else:
        q.add(ask_text(rng, ref_p, ref_s), [{'t': 'ask', 'x': rp, 'y': rs}])
        t = fam.truth(p, s)
    return q.finish('identity', options_for(rng, t['label']) if rng.random() < 0.4 else None, {'truth': t['key']})


# ---------- the training mix ----------

def q_deep(rng) -> dict | None:
    """Chains of 1-12 links in five formats (kinlogic/deep.py)."""
    from . import deep
    return deep.sample(rng)


MIX = {'statements': 20, 'nested': 10, 'speech': 12, 'identity': 6, 'introduce': 6, 'coded': 12, 'puzzle': 12,
       'sufficiency': 6, 'claims': 6, 'story': 10, 'deep': 18}
KINDS = {k: globals()[f'q_{k}'] for k in MIX}


def sample(rng: random.Random, mix: dict | None = None) -> dict:
    """One question of a kind drawn from the mix (retries until a kind produces one)."""
    mix = mix or MIX
    kinds, weights = list(mix), list(mix.values())
    while True:
        ex = KINDS[rng.choices(kinds, weights)[0]](rng)
        if ex is not None:
            return ex


def write_shard(path: str, n: int, seed: int, mix: dict | None = None):
    """n questions as JSON lines {kind, tokens, lf, text}."""
    import gzip
    import json
    rng = random.Random(seed)
    with gzip.open(path, 'wt') as f:
        for _ in range(n):
            ex = sample(rng, mix)
            f.write(json.dumps({'kind': ex['kind'], 'tokens': ex['tokens'], 'lf': ex['lf'], 'text': ex['text']}) + '\n')


def main():
    import argparse
    import os
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(description='Write generated questions as gzipped JSON-lines shards.')
    ap.add_argument('--out', default='data/gen')
    ap.add_argument('--n', type=int, default=200000, help='questions in total')
    ap.add_argument('--shards', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(f'{a.out}/shard-{a.seed}-{i:03d}.jsonl.gz', a.n // a.shards, a.seed * 1000 + i) for i in range(a.shards)]
    with Pool(a.workers) as pool:
        pool.starmap(write_shard, jobs)
    print(f'wrote {a.n} questions to {a.out}')


if __name__ == '__main__':
    main()
