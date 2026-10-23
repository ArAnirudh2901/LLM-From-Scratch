"""Exact reading of formulaic relation questions: when every sentence of a question is one of the fixed forms exam
papers use ("X is the mother of Y", "Y's brother is X", "X and Y are a married couple", "There are eight members ...
in a family of three generations", "How is X related to Y?"), code writes the facts; otherwise None and the model
reads the question. Like kinlogic/codes.py, a reading is returned only when the whole question is understood.

Works on prepare()'s tokens (people are placeholders @k; words are lowercase).
"""
from __future__ import annotations

import re

from .kin import KIN

E = r'(@\d+)'
REL = r'((?:only )?(?:paternal |maternal )?(?:elder |younger |eldest |youngest )?[a-z][a-z-]*)'
NUM = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
       'twelve': 12, 'one': 1, 'a': 1, 'an': 1, 'only': 1}
GWORD = {'male': 'male', 'man': 'male', 'boy': 'male', 'gentleman': 'male', 'female': 'female', 'woman': 'female',
         'girl': 'female', 'lady': 'female'}
PLURAL = {'brothers': 'brother', 'sisters': 'sister', 'siblings': 'sibling', 'sons': 'son', 'daughters': 'daughter'}


def _rel(word):
    """(fact text 'only paternal uncle', ok) for a relation phrase; None if not a relation."""
    w = word.replace('elder ', '').replace('younger ', '').replace('eldest ', '').replace('youngest ', '')
    only = w.startswith('only ')
    w = w[5:] if only else w
    side = None
    for s in ('paternal', 'maternal'):
        if w.startswith(s + ' '):
            side, w = s, w[len(s) + 1:]
    if w not in KIN:
        return None
    return ' '.join(x for x in ('only' if only else None, side, w) if x)


def _num(w):
    return int(w) if w.isdigit() else NUM.get(w)


def clause(c: str):
    """Facts for one clause, or None when it is not a known form. 'x REL y' facts mean y is x's REL."""
    c = c.strip()
    m = re.fullmatch(rf'{E} is (?:the |a |an |also the )?{REL} (?:of|to) {E}(?: and (?:the |a |an )?{REL} of {E})?', c)
    if m:
        x, r1, y, r2, y2 = m.groups()
        a = _rel(r1)
        if a is None:
            return None
        out = [f'{y} {a} {x}']
        if r2:
            b = _rel(r2)
            if b is None:
                return None
            out.append(f'{y2} {b} {x}')
        return out
    m = re.fullmatch(rf'{E} is (?:the |a |an )?{REL} of {E} and {E}', c)   # "X is the mother of A and B"
    if m and _rel(m.group(2)):
        x, r, y, z = m.groups()
        return [f'{y} {_rel(r)} {x}', f'{z} {_rel(r)} {x}']
    m = re.fullmatch(rf"{E} is {E} 's {REL}", c) or re.fullmatch(rf"{E} 's {REL} is {E}", c)
    if m:
        g = m.groups()
        x, y, r = (g[0], g[1], g[2]) if c.split()[1] == 'is' else (g[2], g[0], g[1])
        a = _rel(r)
        return [f'{y} {a} {x}'] if a else None
    m = re.fullmatch(rf'{E} has (a |an |only one |one )?{REL} (?:named |called |, )?{E}', c)
    if m and _rel(m.group(3)) and not _rel(m.group(3)).startswith('only'):
        y, det, r, x = m.groups()
        base = _rel(r)
        out = [f'{y} {base} {x}']
        if det and det.strip() in ('only one', 'one'):   # "has only one child, Aman": the count is a fact too
            out.insert(0, f'{y} has 1 {base}')
        return out
    m = re.fullmatch(rf'{E} is married to {E}', c) or re.fullmatch(rf'{E} and {E} are (?:a )?(?:married couple|married|husband and wife|couple)', c)
    if m:
        return [f'{m.group(1)} spouse {m.group(2)}']
    m = re.fullmatch(rf'{E} and {E} are (brothers|sisters|siblings)', c)
    if m:
        w = PLURAL[m.group(3)]
        return [f'{m.group(1)} {w} {m.group(2)}', f'{m.group(2)} {w} {m.group(1)}'] if w != 'sibling' else [f'{m.group(1)} sibling {m.group(2)}']
    m = re.fullmatch(rf'{E} is (?:a |an )?(male|female|man|woman|boy|girl|lady|gentleman)(?: member)?', c)
    if m:
        return [f'{m.group(1)} {GWORD[m.group(2)]}']
    m = re.fullmatch(rf'{E} is (unmarried|married)', c)
    if m:
        return [f'{m.group(1)} {m.group(2)}']
    m = re.fullmatch(rf'{E} is not (?:the )?{REL} of {E}', c)
    if m and _rel(m.group(2)):
        return [f'{m.group(3)} not {_rel(m.group(2))} {m.group(1)}']
    return None


HEAD = [   # closed-family puzzle sentences
    (rf'(?:there are|in a family of|a family (?:consists|comprises) of|a family has) (\w+) (?:members|persons|people)(?: [@\d ,and]+)?(?: in (?:a|the) family)?(?: of (\w+) generations?)?', ('members', 'generations')),
    (rf'(?:[@\d ,and]+ )?are (\w+) (?:members|persons) of a family(?: of (\w+) generations?)?', ('members', 'generations')),
    (rf'there (?:are|is) (?:only )?(\w+) married couples?(?: in the family)?', ('couples',)),
    (rf'(?:it is )?a family of (\w+) generations?', ('generations',)),
    (rf'there are (\w+) generations?(?: in the family)?', ('generations',)),
    (rf'(?:there is no single parent|no single person in the family is a parent|no one in the family is a single parent|there are no single parents)(?: in the family)?', ('singles0',)),
]


INSTRUCTION = re.compile(r'^(?:(?:please )?(?:answer|refer|read|study|consider|go through|carefully read|directions?)\b.*\b(?:questions?|information|data|below|following|passage)\b.*|directions?)$')
NUMKEY = [('members', r'(?:members?|persons?|people|individuals)'), ('couples', r'(?:married )?couples?'), ('generations', r'generations?'),
          ('females', r'(?:females?|female members?|ladies|women)'), ('males', r'(?:males?|male members?|men)')]
FILLER = {'there', 'are', 'is', 'in', 'a', 'the', 'family', 'of', 'with', 'and', 'consisting', 'consists', 'belong', 'belongs',
          'to', 'which', 'has', 'have', 'only', 'total', 'members', 'member', 'persons', 'person', 'people', 'it', ',', '-', 'all',
          'from', 'this', 'that', 'comprising', 'comprises', 'includes', 'including', 'family members'}


def header(c: str):
    """Closed-family facts from a header sentence in any order ("there are nine members in three generations and three
    married couples", "X , Y , ... are six family members"); None if any other word is left."""
    out, rest = [], ' ' + c + ' '
    if re.search(r'no single (?:parent|person (?:in the family )?is (?:a|the) parent)|no one in the family is a single parent|there are no single parents', rest):
        out.append('singles 0')
        rest = re.sub(r'(?:there is )?no single (?:parent|person (?:in the family )?is (?:a|the) parent)(?: in the family)?|no one in the family is a single parent|there are no single parents(?: in the family)?', ' ', rest)
    for key, pat in NUMKEY:
        m = re.search(rf' (\w+) {pat} ', rest)
        if m and _num(m.group(1)) is not None:
            out.append(f'{key} {_num(m.group(1))}')
            rest = rest[:m.start()] + ' ' + rest[m.end():]
    m = re.search(r' family of (\w+) ', rest)   # "a family of six" (members) / "a family of three generations" handled above
    if m and _num(m.group(1)) is not None and not any(o.startswith('members') for o in out):
        out.append(f'members {_num(m.group(1))}')
        rest = rest[:m.start()] + ' ' + rest[m.end():]
    left = [w for w in rest.split() if not re.fullmatch(E, w) and w not in FILLER]
    return out if out and not left else None


def head(c: str):
    for pat, keys in HEAD:
        m = re.fullmatch(pat, c)
        if m:
            if keys == ('singles0',):
                return ['singles 0']
            out = []
            for k, g in zip(keys, m.groups()):
                if g is not None:
                    n = _num(g)
                    if n is None:
                        return None
                    out.append(f'{k} {n}')
            return out
    return None


ASK = [
    (rf'how is {E} (?:related|connected) (?:to|with) {E}', lambda a, b: f'ask {b} {a}'),
    (rf'(?:what|who) is {E} (?:to|of) {E}', lambda a, b: f'ask {b} {a}'),
    (rf'what is the relation(?:ship)? (?:of|between) {E} (?:to|with|and) {E}', lambda a, b: f'ask {b} {a}'),
    (rf'{E} is (?:the )?____ (?:of|to) {E}', lambda a, b: f'ask {b} {a}'),
    (rf"{E} is {E} 's (?:____|\?)", lambda a, b: f'ask {b} {a}'),
    (rf'how are {E} and {E} related', lambda a, b: f'ask {b} {a}'),
]


def question(c: str):
    c = re.sub(r'^(?:then ,? |so ,? )', '', c.strip())
    for pat, f in ASK:
        m = re.fullmatch(pat, c)
        if m:
            return f(*m.groups())
    m = re.fullmatch(rf'who is (?:the )?{REL} of {E}', c) or re.fullmatch(rf"who is {E} 's {REL}", c)
    if m:
        g = m.groups()
        p, r = (g[1], g[0]) if c.split()[2] != "@" and not g[0].startswith('@') else (g[0], g[1])
        r = _rel(r)
        return f'who {p} {r}' if r else None
    m = re.fullmatch(r'how many (male|female) members are (?:there )?in the family', c) or re.fullmatch(r'how many (males|females) are there in the family', c)
    if m:
        return f"count {m.group(1).rstrip('s')}"
    return None


def read(tokens, why=None) -> str | None:
    """Facts for a formulaic question, or None unless every sentence is understood. why: a list that receives the
    first clause not understood (diagnostics)."""
    text = ' '.join(tokens)
    if re.search(r'\b(?:means|denotes|implies|indicates|represents)\b', text) or re.search(r'@\d+ [^\w\s\'".,?;:()] @\d+', text):
        return None   # coded questions (definitions, or an expression whose definitions were cut off): kinlogic/codes.py
    text = re.sub(r'\s*\?\s*$', '', text)
    parts = [p for p in re.split(r' [.;:] | , (?=and |while |whereas )|\? ', text) if p.strip()]
    parts = [p for p in parts if not INSTRUCTION.match(p.strip())]   # "answer the questions based on the information given below"
    facts, ask = [], None
    for k, p in enumerate(parts):
        p = re.sub(r'^(?:and|while|whereas) ', '', p.strip())
        p = re.sub(r'^\( ?[a-z0-9] ?\) ', '', p)   # "(a) X is ..." clue labels
        if k == len(parts) - 1:
            q = question(p)
            if q:
                ask = q
                continue
        got = clause(p) or head(p) or header(p)
        if got is None:
            sub = re.split(r' , (?:who|which) is ', p)   # "N is the mother-in-law of G , who is the sister of I"
            if len(sub) == 2:
                first = clause(sub[0])
                last = re.findall(E, sub[0])[-1:] if first else []
                second = clause(f'{last[0]} is {sub[1]}') if last else None
                if first and second:
                    facts += first + second
                    continue
            if why is not None:
                why.append(p)
            return None
        facts += got
    if ask is None or not facts:
        return None
    return ' ; '.join(facts + [ask])
