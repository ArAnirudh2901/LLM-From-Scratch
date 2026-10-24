"""Exact reading of formulaic relation questions: when every sentence of a question is one of the fixed forms exam
papers use ("X is the mother of Y", "Y's brother is X", "X and Y are a married couple", "There are eight members ...
in a family of three generations", "How is X related to Y?"), code writes the facts; otherwise None and the model
reads the question. Like kinlogic/codes.py, a reading is returned only when the whole question is understood.

Works on prepare()'s tokens (people are placeholders @k; words are lowercase).
"""
from __future__ import annotations

import re

from .kin import KIN

E = r'([@#]\d+)'   # #k: an unnamed person ("the one, who ...")
REL = r'((?:only )?(?:paternal |maternal )?(?:elder |younger |eldest |youngest )?[a-z][a-z-]*)'
NUM = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
       'twelve': 12, 'one': 1, 'a': 1, 'an': 1, 'only': 1}
GWORD = {'male': 'male', 'man': 'male', 'boy': 'male', 'gentleman': 'male', 'female': 'female', 'woman': 'female',
         'girl': 'female', 'lady': 'female'}
PLURAL = {'brothers': 'brother', 'sisters': 'sister', 'siblings': 'sibling', 'sons': 'son', 'daughters': 'daughter'}


from .facts import PROFS as _P
PROFS = set(_P)
PROF = rf"({'|'.join(sorted(PROFS))})"


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
        a, b = _rel(r1), _rel(r2) if r2 else None
        if a and (b or not r2):   # else not a relation word ("is married to"): other forms below
            return [f'{y} {a} {x}'] + ([f'{y2} {b} {x}'] if r2 else [])
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
    m = re.fullmatch(rf'{E} has (a |an |only one |one )?{REL} (?:, )?(?:named |called )?{E}', c)
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
    m = re.fullmatch(rf'{E} is (?:an? )?(unmarried|married)(?: (?:member|person))?(?: (?:of|in) the family)?', c)
    if m:
        return [f'{m.group(1)} {m.group(2)}']
    m = re.fullmatch(rf'{E} is (?:a |the )?(male|female) (?:member|person)(?: (?:of|in) the family)?', c)
    if m:
        return [f'{m.group(1)} {m.group(2)}']
    m = re.fullmatch(rf'{E} is not (?:a |the )?(male|female)(?: (?:member|person))?(?: (?:of|in) the family)?', c)
    if m:
        return [f"{m.group(1)} {'female' if m.group(2) == 'male' else 'male'}"]
    m = re.fullmatch(rf'{E} (?:does not have any|has no|does not have an?) (brother|sister|sibling|child)s?', c) or re.fullmatch(rf'{E} (?:does not have any|has no) (children)', c)
    if m:
        return [f"{m.group(1)} no-{ {'children': 'child'}.get(m.group(2), m.group(2)) }"]
    m = re.fullmatch(rf'(?:the )?genders? of {E} and {E} (?:is|are) (?:the )?(same|different|opposite)', c) or re.fullmatch(rf'{E} and {E} are (?:of )?(?:the )?(same|different|opposite) genders?', c)
    if m:
        return [f"{m.group(1)} {'same' if m.group(3) == 'same' else 'opposite'} {m.group(2)}"]
    m = re.fullmatch(rf'{E} is (?:the |a |an )?(unmarried|married) {REL} of {E}', c)
    if m and _rel(m.group(3)):
        return [f'{m.group(4)} {_rel(m.group(3))} {m.group(1)}', f'{m.group(1)} {m.group(2)}']
    m = re.fullmatch(rf'{E} has (?:only )?(\w+) (child|children|son|sons|daughter|daughters)(?: and no (son|daughter|children|sons|daughters))?', c)
    if m and _num(m.group(2)) is not None and m.group(2) not in ('a', 'an', 'only'):   # "X has only one son and no daughter"
        one = lambda w: {'children': 'child', 'child': 'child'}.get(w, PLURAL.get(w, w))
        return [f'{m.group(1)} has {_num(m.group(2))} {one(m.group(3))}'] + ([f'{m.group(1)} has 0 {one(m.group(4))}'] if m.group(4) else [])
    m = re.fullmatch(rf'{E} has only (sons|daughters)', c)   # no child of the other kind
    if m:
        return [f"{m.group(1)} has 0 {'daughter' if m.group(2) == 'sons' else 'son'}"]
    m = re.fullmatch(rf'{E} is not married to {E}', c)
    if m:
        return [f'{m.group(1)} not spouse {m.group(2)}']
    m = re.fullmatch(rf'{E} has (?:only )?(\w+) (children|sons|daughters|brothers|sisters|siblings)(?: ,)?(?: named| called| -| :)? ((?:{E} , )*{E} and {E})(?: -)?', c)
    if m and _num(m.group(2)) is not None:
        x, n, w, names = m.group(1), _num(m.group(2)), m.group(3), re.findall(E, m.group(4))
        r = {'children': 'child'}.get(w, PLURAL.get(w))
        return [f'{x} has {n} {r}'] + [f'{x} {r} {y}' for y in names] if len(names) == n else None
    m = re.fullmatch(rf'{E} and {E} are (?:a )?(?:married couple|husband and wife|couple)(?: ,)? (?:having|with|who have|and have) (\w+) (children|sons|daughters)(?: ,)?(?: named| called| -| :)? ((?:{E} , )*{E} and {E})', c)
    if m and _num(m.group(3)) is not None:
        a, b, n, w = m.group(1), m.group(2), _num(m.group(3)), m.group(4)
        names = re.findall(E, m.group(5))
        r = {'children': 'child'}.get(w, PLURAL.get(w))
        if len(names) != n:
            return None
        return [f'{a} spouse {b}', f'{a} has {n} {r}'] + [f'{p} {r} {y}' for y in names for p in (a, b)]
    m = re.fullmatch(rf'{E} is not (?:the )?{REL} of {E}', c)
    if m and _rel(m.group(2)):
        return [f'{m.group(3)} not {_rel(m.group(2))} {m.group(1)}']
    m = re.fullmatch(rf"{E} is not {E} 's {REL}", c)
    if m and _rel(m.group(3)):
        return [f'{m.group(2)} not {_rel(m.group(3))} {m.group(1)}']
    m = re.fullmatch(rf'{E} (?:belongs? to|is (?:in|from|of)) (?:the )?{ORD} generation', c)   # "who belongs to the 3 rd generation"
    if m and _ord(m.group(2)):
        return [f'{m.group(1)} gen {_ord(m.group(2))}']
    m = re.fullmatch(rf'{E} is (?:an? |the )?(?:(married|unmarried) )?(male|female|man|woman|lady|gentleman|boy|girl)(?: member| person)? (?:in|of|from) (?:the )?{ORD} generation', c)
    if m and _ord(m.group(4)):   # "X is married female in 1st generation"
        return [f'{m.group(1)} {GWORD[m.group(3)]}'] + ([f'{m.group(1)} {m.group(2)}'] if m.group(2) else []) + [f'{m.group(1)} gen {_ord(m.group(4))}']
    m = re.fullmatch(rf'{E} is (?:an? )?(unmarried|married) (male|female|man|woman|lady|gentleman|boy|girl)(?: member| person)?', c)
    if m:   # "who is unmarried female"
        return [f'{m.group(1)} {m.group(2)}', f'{m.group(1)} {GWORD[m.group(3)]}']
    m = re.fullmatch(rf'{E} is (unmarried|married|(?:an? )?(?:male|female)) and (?:is )?(?:the |a |an )?{REL} (?:of|to) {E}', c)
    if m and _rel(m.group(3)):   # "X is unmarried and aunt of Y"
        w = m.group(2).split()[-1]
        return [f'{m.group(1)} {w}', f'{m.group(4)} {_rel(m.group(3))} {m.group(1)}']
    m = re.fullmatch(rf'neither {E} nor {E} (?:has|have) (?:a |any )?(sibling|brother|sister)s?', c)
    if m:
        return [f'{m.group(1)} no-{m.group(3)}', f'{m.group(2)} no-{m.group(3)}']
    m = re.fullmatch(rf'(?:the )?genders? of ((?:{E} , )+{E} and {E}) (?:is|are) (?:the )?same', c)
    if m:   # "the gender of A , B and C is same"
        ps = re.findall(E, m.group(1))
        return [f'{ps[0]} same {q}' for q in ps[1:]]
    m = re.fullmatch(rf'(?:the )?genders? of {E} and {E} (?:is|are) (?:the )?same as (?:that of )?{E}', c)
    if m:
        return [f'{m.group(1)} same {m.group(3)}', f'{m.group(2)} same {m.group(3)}']
    m = re.fullmatch(rf'{E} and {E} are (?:the )?(sons|daughters|children|brothers|sisters) of {E}', c)
    if m:   # "A and B are the sons of C"
        r = {'children': 'child'}.get(m.group(3), PLURAL.get(m.group(3)))
        return [f'{m.group(4)} {r} {m.group(1)}', f'{m.group(4)} {r} {m.group(2)}']
    m = re.fullmatch(rf'{E} and {E} are (?:the )?married couples?', c)
    if m:
        return [f'{m.group(1)} spouse {m.group(2)}']
    m = re.fullmatch(r'there (?:are|is) ((?:(?:a|an|one|only one|\w+) (?:fathers?|mothers?|brothers?|sisters?)(?: ,| and| &| , and)? ?)+?)(?: in the (?:group|family))?', c)
    if m:   # "there are two fathers , three brothers and a mother in the group"
        items = re.findall(r'(a|an|one|only one|\w+) (father|mother|brother|sister)s?\b', m.group(1))
        out = []
        for n, r in items:
            k = 1 if n in ('a', 'an', 'only one') else _num(n)
            if k is None:
                return None
            out.append(f'{r}s {k}')
        return out if out and len({o.split()[0] for o in out}) == len(out) else None
    m = re.fullmatch(rf'{E} has (\w+) (sons?|daughters?|brothers?|sisters?|children) and (\w+) (sons?|daughters?|brothers?|sisters?)', c)
    if m and _num(m.group(2)) is not None and _num(m.group(4)) is not None:   # "X has two sons and one daughter"
        one = lambda w: {'children': 'child'}.get(w, PLURAL.get(w, w))
        return [f'{m.group(1)} has {_num(m.group(2))} {one(m.group(3))}', f'{m.group(1)} has {_num(m.group(4))} {one(m.group(5))}']
    m = re.fullmatch(rf'{E} has (\w+) (brothers|sisters|sons|daughters)', c)   # "X has three brothers"
    if m and _num(m.group(2)) not in (None, 1):
        return [f'{m.group(1)} has {_num(m.group(2))} {PLURAL[m.group(3)]}']
    m = re.fullmatch(rf'{E} and {E} are (?:the )?(only |two )?(daughters|sons|children) of (?:mr |mrs |ms )?{E}', c)
    if m:   # "X and Y are the only daughters of Z" / "two children of Mrs Z"
        r = {'children': 'child'}.get(m.group(4), PLURAL.get(m.group(4)))
        return [f'{m.group(5)} {r} {m.group(1)}', f'{m.group(5)} {r} {m.group(2)}'] + ([f'{m.group(5)} has 2 {r}'] if m.group(3) else [])
    m = re.fullmatch(rf"{E} is {E} 's {REL} and {E} 's {REL}", c)   # "X is A 's mother and B 's aunt"
    if m and _rel(m.group(3)) and _rel(m.group(5)):
        return [f'{m.group(2)} {_rel(m.group(3))} {m.group(1)}', f'{m.group(4)} {_rel(m.group(5))} {m.group(1)}']
    m = re.fullmatch(rf'{E} is (?:the |a )?{REL} of {E} and has no (son|daughter|brother|sister|child)(?:ren|s)?', c)
    if m and _rel(m.group(2)):   # "X is the father of Y and has no daughter"
        return [f'{m.group(3)} {_rel(m.group(2))} {m.group(1)}', f'{m.group(1)} has 0 {m.group(4)}']
    m = re.fullmatch(rf'{E} married to {E}', c)
    if m:
        return [f'{m.group(1)} spouse {m.group(2)}']
    m = re.fullmatch(rf'{E} is married to {E} and has (\w+) (sons|daughters|children) ,? {E} and {E}', c)
    if m and _num(m.group(3)) == 2:   # "... is married to Z and has two sons A and B"
        r = {'children': 'child'}.get(m.group(4), PLURAL.get(m.group(4)))
        x = m.group(1)
        return [f'{x} spouse {m.group(2)}', f'{x} has 2 {r}', f'{x} {r} {m.group(5)}', f'{x} {r} {m.group(6)}']
    m = re.fullmatch(rf'{E} is neither married to {E} nor (?:to )?{E}', c)
    if m:
        return [f'{m.group(1)} not spouse {m.group(2)}', f'{m.group(1)} not spouse {m.group(3)}']
    m = re.fullmatch(rf'{E} is (?:a |an |the )?{PROF}', c)   # "X is a doctor"
    if m:
        return [f'{m.group(1)} prof {m.group(2)}']
    m = re.fullmatch(rf'{E} is (?:a |the )?single parent', c)   # a parent with no spouse
    if m:
        return [f'{m.group(1)} unmarried']
    return None


ORD = r'(\d+ ?(?:st|nd|rd|th)?|first|second|third|fourth|fifth|1 st|2 nd|3 rd|4 th)'
ORDS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5}


def _ord(w):
    w = w.replace(' ', '')
    m = re.match(r'(\d+)', w)
    return int(m.group(1)) if m else ORDS.get(w)


HEAD = [   # closed-family puzzle sentences
    (rf'(?:there are|in a family of|a family (?:consists|comprises) of|a family has) (\w+) (?:members|persons|people)(?: [@\d ,and]+)?(?: in (?:a|the) family)?(?: of (\w+) generations?)?', ('members', 'generations')),
    (rf'(?:[@\d ,and]+ )?are (\w+) (?:members|persons) of a family(?: of (\w+) generations?)?', ('members', 'generations')),
    (rf'there (?:are|is) (?:only )?(\w+) married couples?(?: in the family)?', ('couples',)),
    (rf'(?:it is )?a family of (\w+) generations?', ('generations',)),
    (rf'there are (\w+) generations?(?: in the family)?', ('generations',)),
    (rf'(?:there is no single parent|no single person in the family is a parent|no one in the family is a single parent|there are no single parents)(?: in the family)?', ('singles0',)),
]


INSTRUCTION = re.compile(r'^(?:in this type of questions?\b.*|(?:please )?(?:answer|refer|read|study|consider|go through|carefully read|directions?)\b.*\b(?:questions?|information|data|below|following|passage)\b.*|directions?)$')
NUMKEY = [('members', r'(?:(?:family )?members?|persons?|people|individuals)'), ('couples', r'(?:married )?couples?'), ('generations', r'generations?'),
          ('females', r'(?:females?|female members?|ladies|women)'), ('males', r'(?:males?|male members?|men)')]
FILLER = {'there', 'are', 'is', 'in', 'a', 'the', 'family', 'of', 'with', 'and', 'consisting', 'consists', 'belong', 'belongs',
          'to', 'which', 'has', 'have', 'only', 'total', 'members', 'member', 'persons', 'person', 'people', 'it', ',', '-', 'all',
          'from', 'this', 'that', 'comprising', 'comprises', 'includes', 'including', 'family members', '(', ')', 'consist',
          'consisting', 'live', 'lives', 'living', 'present', 'together'}


def header(c: str):
    """Closed-family facts from a header sentence in any order ("there are nine members in three generations and three
    married couples", "X , Y , ... are six family members"); None if any other word is left."""
    out, rest = [], ' ' + c + ' '
    rest = re.sub(r'no single (?:member|one) (?:in the family )?is (?:a |the )?parent|no single parent has a child', 'no single parent', rest)
    if re.search(r'no single (?:parent|person (?:in the family )?is (?:a |the )?parent)|no one in the family is a single parent|there are no single parents', rest):
        out.append('singles 0')
        rest = re.sub(r'(?:there is )?no single (?:parent|person (?:in the family )?is (?:a |the )?parent)(?: in the family)?|no one in the family is a single parent|there are no single parents(?: in the family)?', ' ', rest)
    for key, pat in NUMKEY:
        m = re.search(rf' (\w+) {pat} ', rest)
        if m and _num(m.group(1)) is not None and m.group(1) not in ('a', 'an', 'only'):   # "X is a female member": a gender
            out.append(f'{key} {_num(m.group(1))}')
            rest = rest[:m.start()] + ' ' + rest[m.end():]
    m = re.search(r' family of (\w+) ', rest)   # "a family of six" (members) / "a family of three generations" handled above
    if m and _num(m.group(1)) is not None and not any(o.startswith('members') for o in out):
        out.append(f'members {_num(m.group(1))}')
        rest = rest[:m.start()] + ' ' + rest[m.end():]
    left = [w for w in rest.split() if not re.fullmatch(E, w) and w not in FILLER]
    return out if out and not left else None


def chain(c: str, ch):
    """'NP is NP' with possessive / 'of' chains ("X 's son is Y 's brother", "X is the brother of Y 's husband"):
    each step a fact through an unnamed person #k (ch numbers them across the question)."""
    from .speech import _np
    m = re.fullmatch(r'(.+?) is (.+)', c)
    if not m or re.search(r'\b(?:not|no|any|married|unmarried|who|which|and|or|but|whose)\b', c):
        return None
    n0 = len(ch.facts)
    left = _np(m.group(1).split(), {}, ch)
    if left is None or _np(m.group(2).split(), {}, ch, end=left) is None or any(isinstance(f, tuple) for f in ch.facts):
        del ch.facts[n0:]
        return None
    out = ch.facts[n0:]
    return out if out else None


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
    (rf'how {E} is (?:related|connected) (?:to|with) {E}', lambda a, b: f'ask {b} {a}'),
    (rf"what is {E} 's relation(?:ship)? (?:to|with) {E}", lambda a, b: f'ask {b} {a}'),
    (rf'find (?:out )?the relation(?:ship)? between {E} and {E}', lambda a, b: f'ask {b} {a}'),
]


def question(c: str, ch=None):
    c = re.sub(r'^(?:then(?: ,)? |so(?: ,)? |in terms of relationship(?: ,)? )', '', c.strip())
    c = re.sub(r'^how is ([@#]\d+) is ', r'how is \1 ', c)   # "how is X is related to Y"
    c = re.sub(r'^(?:to find|find out) (?=what|who|how)', '', c)
    m = re.fullmatch(rf'who are (?:the )?(cousins|brothers|sisters|sons|daughters|children) of {E}', c)   # "who are the cousins of X"
    if m:
        return f"who {m.group(2)} {({'cousins': 'cousin', 'children': 'child'}).get(m.group(1), PLURAL.get(m.group(1)))}"
    c = re.sub(r'^who among the following (is|works as) ', r'who \1 ', c)
    m = re.fullmatch(rf'who (?:is|works as) (?:a |an |the )?{PROF}', c)   # "who is the doctor?"
    if m:
        return f'who prof {m.group(1)}'
    m = re.fullmatch(rf"what is (?:the )?profession of {E}|what is {E} 's profession", c)
    if m:
        return f'prof {m.group(1) or m.group(2)}'
    m = re.fullmatch(rf"who is (?:the )?{PROF} 's {REL}", c)   # "who is the lawyer 's wife?"
    if m and ch is not None and _rel(m.group(2)) and not _rel(m.group(2)).startswith('only'):
        k = ch.new()
        return f'{k} prof {m.group(1)} ; who {k} {_rel(m.group(2))}'
    m = re.fullmatch(rf"how is {E} (?:related|connected) (?:to|with) {E} 's {REL}", c)   # "how is X related to Y 's wife"
    if m and ch is not None and _rel(m.group(3)) and not _rel(m.group(3)).startswith('only'):
        k = ch.new()
        return f'{m.group(2)} {_rel(m.group(3))} {k} ; ask {k} {m.group(1)}'
    m = re.fullmatch(rf"who is (?:the )?{REL} of {E} 's {REL}", c)   # "who is the son of X 's sister"
    if m and ch is not None and _rel(m.group(1)) and _rel(m.group(3)) and not _rel(m.group(3)).startswith('only'):
        k = ch.new()
        return f'{m.group(2)} {_rel(m.group(3))} {k} ; who {k} {_rel(m.group(1))}'
    m = re.fullmatch(rf"how is {E} 's {REL} (?:related|connected) (?:to|with) {E}", c)   # someone unnamed is asked about
    if m and ch is not None and _rel(m.group(2)) and not _rel(m.group(2)).startswith('only'):
        k = ch.new()
        return f'{m.group(1)} {_rel(m.group(2))} {k} ; ask {m.group(3)} {k}'
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


def _part(p: str, ch, depth: int = 0):
    """Facts for one sentence: a known form, or forms joined by commas and who-clauses ("X , who is the father of Y ,
    is married to Z", "the one , who is married to X is the father of Y", "A is the brother of B , C is the wife of D ,
    who is the only son of E"); None unless every piece is understood."""
    p = p.strip()
    got = clause(p) or head(p) or header(p) or chain(p, ch)
    if got or depth > 6:
        return got
    p = re.sub(r'^([@#]\d+) too ', r'\1 ', p)   # "X too has a daughter named Y"
    m = re.fullmatch(rf"{E} , whose {REL} is {E} ,? (.+)", p)   # "X , whose husband is Y , is the mother of Z"
    if m and _rel(m.group(2)) and not _rel(m.group(2)).startswith('only'):
        rest = _part(f'{m.group(1)} {m.group(4)}', ch, depth + 1)
        if rest:
            return [f'{m.group(1)} {_rel(m.group(2))} {m.group(3)}'] + rest
    m = re.fullmatch(rf"(.+ ){E} , whose {REL} is {E}", p)   # "X 's mother is Y , whose husband is Z"
    if m and _rel(m.group(3)) and not _rel(m.group(3)).startswith('only'):
        first = _part(f'{m.group(1)}{m.group(2)}', ch, depth + 1)
        if first:
            return first + [f'{m.group(2)} {_rel(m.group(3))} {m.group(4)}']
    m = re.fullmatch(r'((?:[@#]\d+ , )+[@#]\d+)(?: ,)? and ([@#]\d+) are (?:all )?(sisters|brothers|siblings|cousins)', p)   # three or more
    if m:
        xs = re.findall(r'[@#]\d+', m.group(1)) + [m.group(2)]
        r = {'sisters': 'sister', 'brothers': 'brother', 'siblings': 'sibling', 'cousins': 'cousin'}[m.group(3)]
        return [f'{a} {r} {b}' for a in xs for b in xs if a != b]
    m = re.fullmatch(r'([@#]\d+) and ([@#]\d+) are (cousins|spouses|a married couple|husband and wife)', p)
    if m:
        a, b, w = m.groups()
        return {'cousins': [f'{a} cousin {b}', f'{b} cousin {a}'], 'spouses': [f'{a} spouse {b}'], 'a married couple': [f'{a} spouse {b}'],
                'husband and wife': [f'{b} husband {a}', f'{a} wife {b}']}[w]
    m = re.fullmatch(rf'(.+?)(?: ,)? (?:but|however)(?: ,)? (.+)', p)   # "X is the father of Y but Y is not his son"
    if m and not re.search(r' (?:but|however) ', m.group(1)):
        first = _part(m.group(1), ch, depth + 1)
        subj = re.match(rf'{E} ', m.group(1))
        if first and subj:
            sec = m.group(2)
            sec = re.sub(r"\b(?:his|her)\b", f"{subj.group(1)} 's", sec)
            if sec.startswith('not '):   # "X is Y 's cousin but not the brother of Z"
                sec = f'{subj.group(1)} is {sec}'
            second = _part(sec, ch, depth + 1)
            if second:
                return first + second
    m = re.fullmatch(rf"{E}(?: ,)? {E} 's {REL}(?: ,)? has (?:a|an|one) (son|daughter) {E} and (?:a|an|one) (son|daughter) {E}", p)
    if m and _rel(m.group(3)):   # "X , Y 's sister , has a son A and a daughter B"
        x, y, r, r1, a1, r2, a2 = m.groups()
        return [f'{y} {_rel(r)} {x}', f'{x} {r1} {a1}', f'{x} {r2} {a2}']
    m = re.fullmatch(rf'{E} has (?:a|an) {REL}', p)   # "X has a daughter": someone unnamed
    if m and _rel(m.group(2)) and not _rel(m.group(2)).startswith('only'):
        return [f'{m.group(1)} {_rel(m.group(2))} {ch.new()}']
    m = re.fullmatch(rf"{E} is not (?:the )?{REL} of {E} 's {REL}", p)   # "X is not the granddaughter of Y 's sister"
    if m and _rel(m.group(2)) and _rel(m.group(4)) and not _rel(m.group(4)).startswith('only'):
        k = ch.new()
        return [f'{m.group(3)} {_rel(m.group(4))} {k}', f'{k} not {_rel(m.group(2))} {m.group(1)}']
    m = re.fullmatch(rf'{E} has (?:a|an) {REL} and (?:a|an) {REL}', p)   # "X has a brother and a niece"
    if m and _rel(m.group(2)) and _rel(m.group(3)) and not any(_rel(g).startswith('only') for g in m.groups()[1:]):
        return [f'{m.group(1)} {_rel(m.group(2))} {ch.new()}', f'{m.group(1)} {_rel(m.group(3))} {ch.new()}']
    m = re.fullmatch(rf'{E} has (only one|one|a|an) (child|son|daughter)(?: ,)? who (.+)', p)   # "X has only one child , who belongs to ..."
    if m:
        x, det, r, rest = m.groups()
        k = ch.new()
        pred = _part(f'{k} {rest}', ch, depth + 1)
        if pred:
            return ([f'{x} has 1 {r}'] if 'one' in det else []) + [f'{x} {r} {k}'] + pred
    m = re.fullmatch(rf"{E} 's {REL} has (?:a|an) {REL}", p)   # "X 's spouse has a brother"
    if m and _rel(m.group(2)) and _rel(m.group(3)) and not _rel(m.group(3)).startswith('only'):
        k1, k2 = ch.new(), ch.new()
        return [f'{m.group(1)} {_rel(m.group(2))} {k1}', f'{k1} {_rel(m.group(3))} {k2}']
    m = re.fullmatch(rf"{E} is married to (?:the {REL} of {E}|{E} 's {REL})", p)   # "X is married to the mother of Y"
    if m:
        r, y = (m.group(2), m.group(3)) if m.group(2) else (m.group(5), m.group(4))
        if _rel(r) and not _rel(r).startswith('only'):
            k = ch.new()
            return [f'{y} {_rel(r)} {k}', f'{m.group(1)} spouse {k}']
    m = re.fullmatch(rf'{E} (?:is|,) (?:an? |the )?{PROF}(?: ,)? and (?:is )?(.+)', p) or re.fullmatch(rf'{E} (?:is )?(.+?) and is (?:an? |the )?{PROF}', p)
    if m and re.search(r'[@#]\d+ (?:is|has|was) ', m.group(2) if m.group(3) in PROFS else m.group(3)):
        m = None   # "A has a wife B and C is ... and is an engineer": the profession is C's, read clause by clause
    if m:   # "X is a doctor and the father of Y"
        prof, rest = (m.group(2), m.group(3)) if m.group(2) in PROFS else (m.group(3), m.group(2))
        got = _part(f'{m.group(1)} is {rest}' if not rest.startswith(('is ', 'has ')) else f'{m.group(1)} {rest}', ch, depth + 1)
        if got:
            return [f'{m.group(1)} prof {prof}'] + got
    m = re.fullmatch(rf'{E}(?: ,)? the {PROF}(?: ,)? (.+)', p)   # "D , the salesman , is married to ..."
    if m:
        got = _part(f'{m.group(1)} {m.group(3)}', ch, depth + 1)
        if got:
            return [f'{m.group(1)} prof {m.group(2)}'] + got
    m = re.fullmatch(rf'(?:the )?{PROF} is married to (?:the )?(?:(lady|female|male|gentleman) )?{PROF}', p)   # "the doctor is married to the lawyer"
    if m:
        k1, k2 = ch.new(), ch.new()
        g = {'lady': ['female'], 'female': ['female'], 'male': ['male'], 'gentleman': ['male']}.get(m.group(2) or '', [])
        return [f'{k1} prof {m.group(1)}', f'{k2} prof {m.group(3)}', f'{k1} spouse {k2}'] + [f'{k2} {x}' for x in g]
    m = re.fullmatch(rf'{E} is married to (?:the |a )?(?:(lady|female|male|gentleman) )?{PROF}', p)   # "D is married to the lady teacher"
    if m:
        k = ch.new()
        g = {'lady': ['female'], 'female': ['female'], 'male': ['male'], 'gentleman': ['male']}.get(m.group(2) or '', [])
        return [f'{k} prof {m.group(3)}', f'{m.group(1)} spouse {k}'] + [f'{k} {x}' for x in g]
    m = re.fullmatch(rf"{E} 's {REL} {E}(?: ,)? (is .+|has .+)", p)   # "X 's brother Y , is the grandfather of ..."
    if m and _rel(m.group(2)):
        rest = _part(f'{m.group(3)} {m.group(4)}', ch, depth + 1)
        if rest:
            return [f'{m.group(1)} {_rel(m.group(2))} {m.group(3)}'] + rest
    m = re.fullmatch(r"(.+ 's .+?) is (.+)", p)   # "X 's mother is sister of Y and daughter of Z"
    if m and not re.search(r'\b(?:who|which|not)\b', p):
        from .speech import _np
        n0 = len(ch.facts)
        node = _np(m.group(1).split(), {}, ch)
        if node is not None and not any(isinstance(f, tuple) for f in ch.facts):
            got = clause(f'{node} is {m.group(2)}')
            if got:
                made = ch.facts[n0:]
                del ch.facts[n0:]
                return made + got
        del ch.facts[n0:]
    pieces = re.split(r' and (?=[@#]\d+ (?:is|has|was) )', p)   # "A is the mother of B and C is the son of D"
    if len(pieces) > 1:
        out = []
        for x in pieces:
            got = _part(x, ch, depth + 1)
            if not got:
                out = None
                break
            out += got
        if out:
            return out
    m = re.match(r'^(.*?)(?: ,)? (?:who|which) (.+)$', p)
    if m:
        left, right = m.group(1).strip(), m.group(2).strip()
        anon = re.search(r'\bthe (?:one|person)$', left)
        if anon:
            subj = ch.new()
            left = left[:anon.start()] + subj
        else:
            ents = re.findall(E, left)
            if not ents or not re.search(rf'{E}$', left):
                return None
            subj = ents[-1]
        cuts = [mm.start() for mm in re.finditer(r' , | (?=is |has |does )', right)]
        for i in [None] + cuts:   # who-predicate alone, or followed by the rest of the main sentence
            pred, rest = (right, None) if i is None else (right[:i], right[i:].lstrip(' ,').strip())
            a = _part(f'{subj} {pred}', ch, depth + 1)
            if not a:
                continue
            if rest is None:
                b = _part(left, ch, depth + 1) if left != subj else []
            elif left == subj:
                b = _part(f'{subj} {rest}', ch, depth + 1)
            else:
                b = _part(f'{left} , {rest}', ch, depth + 1) or _part(f'{left} {rest}', ch, depth + 1)
            if b is not None and (b or left == subj):
                return b + a
        return None
    pieces = p.split(' , ')
    if len(pieces) > 1:
        out = []
        for x in pieces:
            got = _part(x, ch, depth + 1)
            if not got:
                return None
            out += got
        return out
    return None


def read(tokens, why=None) -> str | None:
    """Facts for a formulaic question, or None unless every sentence is understood. why: a list that receives the
    first clause not understood (diagnostics)."""
    text = ' '.join(tokens)
    if re.search(r'\b(?:means|denotes|implies|indicates|represents)\b', text) or re.search(r'@\d+ [^\w\s\'".,?;:()] @\d+', text):
        return None   # coded questions (definitions, or an expression whose definitions were cut off): kinlogic/codes.py
    text = re.sub(r'\s*\?\s*$', '', text)
    hon = []   # "Mrs. X" / "Mr X": the title gives the gender

    def _title(m):
        hon.append(f"{m.group(2)} {'female' if m.group(1) in ('mrs', 'ms', 'miss', 'smt') else 'male'}")
        return m.group(2)
    text = re.sub(r'\b(mrs|ms|miss|smt|mr|shri|sri) (?:\. )?([@#]\d+)', _title, text)
    text = re.sub(r"\b(\w+)(?:-| )in(?:-| )law\b", r'\1-in-law', text)
    text = re.sub(r" [‟’′`] s\b", " 's", text)
    text = re.sub(r"\b(does|do|is|has) n ' t\b|\b(does|do|is|has)n ' t\b", lambda m: (m.group(1) or m.group(2)) + ' not', text)
    parts = [p for p in re.split(r' [.;:] | , (?=and |while |whereas )|\? ', text) if p.strip()]
    parts = [p for p in parts if not INSTRUCTION.match(p.strip())]   # "answer the questions based on the information given below"
    if parts:   # "..., then what is the relation of X with Y" / "how is X related to Y if Z has no siblings"
        last = parts[-1]
        m = re.fullmatch(r'(.+?)(?: ,)? then ((?:what|who|how) .+)', last)
        if m:
            parts[-1:] = [m.group(1), m.group(2)]
        m = re.fullmatch(r'((?:what|who|how) .+?)(?: ,)? if (.+)', parts[-1])
        if m:
            parts[-1:] = [m.group(2), m.group(1)]
        m = re.fullmatch(r'(.+?)(?: ,)? if (.+?)(?: ,)? ((?:what|who|how) .+)', parts[-1])
        if m:
            parts[-1:] = [m.group(1), m.group(2), m.group(3)]
    from .speech import _Chain
    facts, ask, ch = [], None, _Chain()
    for k, p in enumerate(parts):
        p = re.sub(r'^(?:and|while|whereas) ', '', p.strip())
        p = re.sub(r'^\( ?[a-z0-9] ?\) ', '', p)
        p = re.sub(r'^(?:pick|choose|select) the (?:right|correct) (?:choice|option|answer) from the (?:given )?options(?: ,)? (?:if )?', '', p)   # "(a) X is ..." clue labels
        if k == len(parts) - 1:
            if re.fullmatch(r'(?:then(?: ,)? )?which (?:of the following|one|among the following)[^?]*\b(?:true|correct|false|incorrect)', p.strip()):
                break   # "which of the following statements is true?": the exam layer checks the options
            q = question(p, ch)
            if q:
                ask = q
                continue
        got = _part(p, ch)
        if got is None:
            if why is not None:
                why.append(p)
            return None
        facts += got
    if ask is None and facts and re.fullmatch(r'(?:then(?: ,)? )?which (?:of the following|one|among the following)[^?]*\b(?:true|correct|false|incorrect)', parts[-1].strip()):
        return ' ; '.join(facts + hon)   # "which of the following statements is true?": the exam layer checks the options
    if ask is None or not facts:
        return None
    named = set(re.findall(E, ' ; '.join(facts)))
    if any(x not in named for x in re.findall(r'@\d+', ask)):   # asked about someone no sentence mentions: text lost in the scrape
        return None
    return ' ; '.join(facts + hon + [ask])
