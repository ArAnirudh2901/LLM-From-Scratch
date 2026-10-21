"""Exact reading of coded blood-relation questions ("A $ B means A is the father of B ... how is D related to A in
A $ B # C * D?"). Their definitions follow a fixed pattern, so code reads them instead of the model: every symbol
gets its relation and direction, the expression becomes one fact per adjacent pair (in text order), and the question
becomes "ask" or, for "which expression shows that P is Q's REL", a claim. Anything not fully understood returns
None and the model reads the question as before.

Works on prepare()'s tokens (letters and names are already placeholders such as @3).
"""
from __future__ import annotations

import re

from .kin import KIN

QUOTES = {"'", '"', '‘', '’', '“', '”', '`'}
VERB = r'(?:means|implies|denotes|represents|indicates|signifies|stands for|is coded as|shows)'
NOT_OP = {',', ';', '.', ':', '(', ')', '[', ']', 'and', 'or', 'if', 'then', 'is', 'the', 'of', 'a', 'an', 'to', 'in'}
REL = r'((?:paternal |maternal )?[a-z][a-z-]*)'
E = r'(@\d+)'
DEF_FWD = re.compile(rf'{E} (\S+) {E} {VERB} {E} is (?:the )?{REL} (?:of )?{E}')
DEF_MARRIED = re.compile(rf'{E} (\S+) {E} {VERB} {E} is married to {E}')
DEF_POSS = re.compile(rf"{E} (\S+) {E} {VERB} {E} is {E} 's {REL}")
ASK = [re.compile(p) for p in (   # each: how is the first person related to the second?
    rf'how is {E} (?:related|connected) (?:to|with) {E}', rf'(?:what|who) is {E} to {E}', rf'{E} is related to {E} as',
    rf'relation(?:ship)? of {E} (?:to|with) {E}', rf'relation(?:ship)? between {E} and {E}',
    rf'in what way is {E} related to {E}', rf'find the relation of {E} (?:with|to) {E}', rf'what relation does {E} have with {E}',
    rf'which of the following is the relation of {E} to {E}', rf'how are {E} and {E} related', rf"{E} is {E} 's \?")]
GENDER = re.compile(rf'{E} is (?:an? )?(female|male|woman|man|girl|boy|lady|gentleman)\b')
FEMALE = {'female', 'woman', 'girl', 'lady'}
_WV = r'(?:(?:would|will|can|does) )?(?:means?|shows?|indicates?|represents?|implies|imply|denotes?)'
WHICH = re.compile(rf'which (?:of the following|one|expression)[^?]*? {_WV} (?:that )?{E} is (?:the )?{REL} of {E}')
WHICH_POSS = re.compile(rf"which (?:of the following|one|expression)[^?]*? {_WV} (?:that )?{E} is {E} 's {REL}")


def _rel(word):
    w = word.strip()
    side = None
    for s in ('paternal', 'maternal'):
        if w.startswith(s + ' '):
            side, w = s, w[len(s) + 1:]
    return (side, w) if w in KIN else None


def definitions(text):
    """{symbol: (side, relation, flipped)}: flipped when "A op B" means "B is A's relation"; None if two
    definitions of one symbol disagree."""
    defs = {}
    for m in DEF_FWD.finditer(text):
        a, op, b, x, rel, y = m.groups()
        r = _rel(rel)
        if r is None or op in NOT_OP or {x, y} != {a, b} or a == b:
            continue
        d = (r[0], r[1], x == b)
        if defs.setdefault(op, d) != d:
            return None
    for m in DEF_POSS.finditer(text):   # "A op B means A is B's relation"
        a, op, b, x, y, rel = m.groups()
        r = _rel(rel)
        if r is None or op in NOT_OP or {x, y} != {a, b} or a == b:
            continue
        d = (r[0], r[1], x == b)
        if defs.setdefault(op, d) != d:
            return None
    for m in DEF_MARRIED.finditer(text):   # "A & B means A is married to B"
        a, op, b, x, y = m.groups()
        if op in NOT_OP or {x, y} != {a, b} or a == b:
            continue
        if defs.setdefault(op, (None, 'spouse', False)) != (None, 'spouse', False):
            return None
    return defs


def expressions(tokens, defs):
    """Maximal runs "@a op @b op @c ..." made only of defined symbols, outside the definitions themselves."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i].startswith('@') and i + 2 < len(tokens) and tokens[i + 1] in defs and tokens[i + 2].startswith('@'):
            j = i
            run = [tokens[i]]
            while j + 2 < len(tokens) and tokens[j + 1] in defs and tokens[j + 2].startswith('@'):
                run += [tokens[j + 1], tokens[j + 2]]
                j += 2
            nxt = ' '.join(tokens[j + 1:j + 3])
            if not re.match(VERB, nxt):   # a definition, not an expression
                out.append(run)
            i = j + 1
        else:
            i += 1
    return out


def facts_of(run, defs):
    fs = []
    for k in range(0, len(run) - 2, 2):
        a, op, b = run[k], run[k + 1], run[k + 2]
        side, rel, flipped = defs[op]
        x, y = (b, a) if flipped else (a, b)          # x is y's relation
        fs.append(f"{y} {(side + ' ') if side else ''}{rel} {x}")
    return fs


def read(tokens) -> str | None:
    """Facts for a coded question, or None when it is not one this reader fully understands."""
    toks = [t for t in tokens if t not in QUOTES]
    text = ' '.join(toks)
    defs = definitions(text)
    if not defs:
        return None
    runs = expressions(toks, defs)
    w, wp = WHICH.search(text), WHICH_POSS.search(text)
    if w or wp:   # "which expression shows that P is Q's REL": the options are read one by one
        p, rel, q = w.groups() if w else (wp.group(1), wp.group(3), wp.group(2))
        r = _rel(rel)
        return None if r is None else f"claim 1 ; {q} {(r[0] + ' ') if r[0] else ''}{r[1]} {p}"
    asked = set(re.findall(r'@\d+', ' '.join(re.findall(r'(?:how is|what is|relation|between|related)[^?.]*', text))))
    runs = [r for r in runs if len(set(r[::2]) & asked) >= 2] or runs   # options or page junk also look like expressions
    if len(runs) != 1:
        runs = sorted(runs, key=len)
        if len(runs) < 2 or len(runs[-1]) == len(runs[-2]):
            return None
        runs = runs[-1:]
    people = set(runs[0][::2])
    extra = []
    rel_of = re.search(rf"how is {E} 's {REL} related to {E}", text)        # "how is R's husband related to H?"
    of_rel = re.search(rf"how is {E} related to {E} 's {REL}", text)        # "how is H related to R's husband?"
    if rel_of and _rel(rel_of.group(2)):
        a, rel, x = rel_of.groups()
        side, w = _rel(rel)
        extra, y = [f"{a} {(side + ' ') if side else ''}{w} #1"], '#1'
        if a not in people or x not in people:
            return None
    elif of_rel and _rel(of_rel.group(3)):
        y, a, rel = of_rel.groups()
        side, w = _rel(rel)
        extra, x = [f"{a} {(side + ' ') if side else ''}{w} #1"], '#1'
        if a not in people or y not in people:
            return None
    else:
        ask = next((m for pat in ASK for m in [pat.search(text)] if m), None)
        if ask is None:
            return None
        y, x = ask.groups()
        if x not in people or y not in people:
            return None
    genders = [f"{m.group(1)} {'female' if m.group(2) in FEMALE else 'male'}" for m in GENDER.finditer(text)]
    last = None   # the one person of the previous clause, for "when P got the job, he called home"
    for sent in re.split(r' [.;,] ', text):   # "P lost his keys" / "P 's wife is a doctor" (a clause: no definitions, one person)
        ps = set(re.findall(r'@\d+', sent))
        if re.search(VERB, sent) or any(op in sent.split() for op in defs) or len(ps) > 1:
            last = None
            continue
        if not ps:
            if last is None:
                continue
            ps = {last}
        p = ps.pop()
        last = p
        sp = re.search(rf"{p} 's (wife|husband)\b", sent)
        pro = set(re.findall(r'\b(he|his|him|she|her)\b', sent))
        g = ('male' if sp.group(1) == 'wife' else 'female') if sp else \
            ('male' if pro <= {'he', 'his', 'him'} else 'female' if pro <= {'she', 'her'} else None) if pro else None
        if g and f'{p} {g}' not in genders:
            genders.append(f'{p} {g}')
    return ' ; '.join(facts_of(runs[0], defs) + extra + genders + [f'ask {x} {y}'])


def with_codes(read_model):
    """A reader that reads coded questions exactly and passes the rest to `read_model`."""
    def r(batch):
        dec = [read(t) for t in batch]
        need = [t for t, d in zip(batch, dec) if d is None]
        got = iter(read_model(need) if need else [])
        return [d if d is not None else next(got) for d in dec]
    return r
