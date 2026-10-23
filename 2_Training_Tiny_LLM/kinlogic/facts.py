"""The fact language the reader model writes, one statement per ";" in text order.

    @1 mother #1 ; #1 brother #2 ; #2 son @0 ; ask @1 @0

"x REL y" means y is x's REL. @k is a person (or placeholder) from the text, #k an unnamed person.
Facts
    x [not] [only] [paternal|maternal] REL y     family or other link
    x male | x female                             gender the text gives (pronoun, "a man", "Mrs.")
    x no-sibling | no-brother | no-sister | no-child | unmarried | married
    x opposite y | x same y                       genders differ / match
    x has N [side] REL                            exactly N such relatives ("has only daughters": has 0 son)
    members N | couples N | generations N | males N | females N | singles N   closed-family puzzle facts (singles: single parents)
    x gen N                                       x is in generation N of the family (1: the oldest)
Questions
    ask x y               how is y related to x?
    who x [side] REL      who is x's REL?
    count male | female | people | couples        how many in the family
    count x [side] REL    how many RELs does x have?
    gender x              is x male or female?
    pair male | female | couple | REL             which option is such a pair ("a pair of brothers")?
Groups
    part N     the facts after it are statement N (data sufficiency)
    claim N    the facts after it are claim N, to be checked, not assumed ("which of these is true?")
"""
from __future__ import annotations

import re

from .kin import KIN, NONKIN, ONLY, SIDED

ATTRS = ['no-sibling', 'no-brother', 'no-sister', 'no-child', 'unmarried', 'married']
COUNTABLE = ['male', 'female', 'people', 'couples']
SIDES = ['paternal', 'maternal']
PUZZLE = ['members', 'couples', 'generations', 'males', 'females', 'singles']
GROUPS = ['part', 'claim']
QUESTIONS = ['ask', 'who', 'count', 'gender', 'pair']
KEYWORDS = [';', *QUESTIONS, *PUZZLE, *GROUPS, 'not', 'only', 'has', 'male', 'female', 'opposite', 'same',
            'people', 'couple', *SIDES, *ATTRS]
RELS = [*KIN, *sorted(NONKIN)]
NUMBERS = [str(n) for n in range(0, 21)]
ENT_RE = re.compile(r'^[@#]\d+$')


def is_ent(t: str) -> bool:
    return bool(ENT_RE.match(t))


def is_named(t: str) -> bool:
    return t[0] == '@'


def _is_rel(t):
    return t in KIN or t in NONKIN


def _mods(tk):
    """ "not", "only", a side, in that order, each at most once."""
    m, i = {'neg': False, 'only': False, 'side': None}, 0
    if i < len(tk) and tk[i] == 'not':
        m['neg'], i = True, i + 1
    if i < len(tk) and tk[i] == 'only':
        m['only'], i = True, i + 1
    if i < len(tk) and tk[i] in SIDES:
        m['side'], i = tk[i], i + 1
    return m if i == len(tk) else None


def _side_ok(r, side):
    return not side or r in SIDED


def _sided_rel(tk, kin_only=False):
    """ "[side] REL" -> (side, rel) or None."""
    if len(tk) == 1 and (tk[0] in KIN or (not kin_only and tk[0] in NONKIN)):
        return None, tk[0]
    if len(tk) == 2 and tk[0] in SIDES and tk[1] in SIDED:
        return tk[0], tk[1]
    return None


def parse_stmt(tk: list[str]) -> dict | None:
    if not tk:
        return None
    a, n = tk[0], len(tk)
    b = tk[1] if n > 1 else None
    if a == 'ask':
        return {'t': 'ask', 'x': tk[1], 'y': tk[2]} if n == 3 and is_ent(b) and is_ent(tk[2]) and b != tk[2] else None
    if a == 'who':
        sr = _sided_rel(tk[2:]) if n >= 3 and is_ent(b) else None
        return {'t': 'who', 'x': b, 'r': sr[1], 'side': sr[0]} if sr else None
    if a == 'count':
        if n == 2 and b in COUNTABLE:
            return {'t': 'count', 'what': b}
        sr = _sided_rel(tk[2:], kin_only=True) if n >= 3 and is_ent(b) else None
        return {'t': 'count', 'what': 'rel', 'x': b, 'r': sr[1], 'side': sr[0]} if sr else None
    if a == 'gender':
        return {'t': 'gender?', 'x': b} if n == 2 and is_ent(b) else None
    if a == 'pair':
        return {'t': 'pair', 'what': b} if n == 2 and (b in ('male', 'female', 'couple') or b in KIN) else None
    if a in PUZZLE or a in GROUPS:
        return {'t': a, 'n': int(b)} if n == 2 and b.isdigit() else None
    if not is_ent(a) or b is None:
        return None
    if n == 2:
        if b in ('male', 'female'):
            return {'t': 'gender', 'x': a, 'g': 'M' if b == 'male' else 'F'}
        return {'t': 'attr', 'x': a, 'a': b} if b in ATTRS else None
    if b == 'has':
        sr = _sided_rel(tk[3:], kin_only=True) if n >= 4 and tk[2].isdigit() else None
        return {'t': 'has', 'x': a, 'n': int(tk[2]), 'r': sr[1], 'side': sr[0]} if sr else None
    if n == 3 and b == 'gen':   # "x gen 3": x belongs to the third generation (1: the oldest); exact readers only for now
        return {'t': 'gen', 'x': a, 'n': int(tk[2])} if tk[2].isdigit() and int(tk[2]) >= 1 else None
    if n == 3 and b in ('opposite', 'same'):
        return {'t': b, 'x': a, 'y': tk[2]} if is_ent(tk[2]) and tk[2] != a else None
    y, r = tk[-1], tk[-2]
    if not is_ent(y) or y == a or not _is_rel(r):
        return None
    m = _mods(tk[1:-2])
    if not m or not _side_ok(r, m['side']) or (m['only'] and r not in ONLY):
        return None
    if r in NONKIN and (m['only'] or m['side']):
        return None
    return {'t': 'rel', 'x': a, 'r': r, 'y': y, **m}


QUESTION_TYPES = {'ask', 'who', 'count', 'gender?', 'pair'}


def parse(text) -> tuple[list[dict], list[str]]:
    """Parses fact text (or a token list). Returns (statements, statements that did not parse).
    Facts after "part N" get part=N; after "claim N" get claim=N. Questions never belong to a group."""
    toks = text if isinstance(text, list) else text.split()
    stmts, bad, cur, group = [], [], [], {}

    def flush():
        nonlocal cur, group
        if not cur:
            return
        s = parse_stmt(cur)
        if s is None:
            bad.append(' '.join(cur))
        elif s['t'] in GROUPS:
            group = {s['t']: s['n']}
        elif s['t'] in QUESTION_TYPES:
            stmts.append(s)
        else:
            stmts.append({**s, **group})
        cur = []

    for t in toks:
        if t in (';', '<sep>'):
            flush()
        else:
            cur.append(t)
    flush()
    return stmts, bad


def format_stmt(s: dict) -> str:
    t = s['t']
    if t == 'rel':
        mods = [w for w in ('not' if s.get('neg') else None, 'only' if s.get('only') else None, s.get('side')) if w]
        return ' '.join([s['x'], *mods, s['r'], s['y']])
    if t == 'gender':
        return f"{s['x']} {'male' if s['g'] == 'M' else 'female'}"
    if t == 'attr':
        return f"{s['x']} {s['a']}"
    if t in ('opposite', 'same'):
        return f"{s['x']} {t} {s['y']}"
    if t == 'has':
        return ' '.join(w for w in (s['x'], 'has', str(s['n']), s.get('side'), s['r']) if w)
    if t == 'ask':
        return f"ask {s['x']} {s['y']}"
    if t == 'who':
        return ' '.join(w for w in ('who', s['x'], s.get('side'), s['r']) if w)
    if t == 'count':
        if s['what'] == 'rel':
            return ' '.join(w for w in ('count', s['x'], s.get('side'), s['r']) if w)
        return f"count {s['what']}"
    if t == 'gender?':
        return f"gender {s['x']}"
    if t == 'pair':
        return f"pair {s['what']}"
    return f"{t} {s['n']}"


def format(stmts: list[dict]) -> str:
    """Writes statements back as fact text."""
    out, group = [], {}
    for s in stmts:
        if s['t'] not in QUESTION_TYPES:
            g = {k: s[k] for k in GROUPS if s.get(k)}
            if g and g != group:
                k, v = next(iter(g.items()))
                out.append(f'{k} {v}')
            group = g or group
        out.append(format_stmt(s))
    return ' ; '.join(out)


# ---------- constrained decoding ----------

MAX_NEW = 16
_STARTERS = [*QUESTIONS, *PUZZLE, *GROUPS]
_AFTER_ENT = ['not', 'only', *SIDES, *RELS, 'male', 'female', *ATTRS, 'opposite', 'same', 'has']
_SIDED = sorted(SIDED)
_ONLY = sorted(ONLY)
_PAIR = ['male', 'female', 'couple', *KIN]


def vocabulary() -> list[str]:
    """Every token the fact language uses, apart from entities and the end marker."""
    return list(dict.fromkeys([*KEYWORDS, *RELS, *NUMBERS]))


def allowed_next(written: list[str], ents: list[str]) -> list[str]:
    """Tokens that may come next after `written` (fact tokens so far, ';' between statements).
    ents: the placeholders in the input. Unnamed people are introduced in order (#1, then #2...).
    '<eos>' may end the facts wherever a statement is complete."""
    out = _allowed(written, ents)
    return out + ['<eos>'] if ';' in out else out


def _allowed(written: list[str], ents: list[str]) -> list[str]:
    n_new = 0
    for t in written:
        if t[0] == '#':
            n_new = max(n_new, int(t[1:]))
    cur = []
    for t in written:
        cur = [] if t == ';' else cur + [t]
    people = ents + [f'#{k}' for k in range(1, n_new + 1)] + ([f'#{n_new + 1}'] if n_new < MAX_NEW else [])
    others = lambda x: [p for p in people if p != x]
    if not cur:
        return [*people, *_STARTERS, '<eos>']
    a, n = cur[0], len(cur)
    last = cur[-1]

    def sided(i, kin_only=False):   # "[side] REL" starting at cur[i]
        rels = [*KIN] if kin_only else RELS
        if n == i:
            return [*SIDES, *rels]
        if n == i + 1 and cur[i] in SIDES:
            return [w for w in _SIDED if not kin_only or w in KIN]
        return [';']

    if a == 'ask':
        return people if n == 1 else others(cur[1]) if n == 2 else [';']
    if a == 'who':
        return people if n == 1 else sided(2)
    if a == 'count':
        if n == 1:
            return [*COUNTABLE, *people]
        return [';'] if cur[1] in COUNTABLE else sided(2, kin_only=True)
    if a == 'gender':
        return people if n == 1 else [';']
    if a == 'pair':
        return _PAIR if n == 1 else [';']
    if a in PUZZLE or a in GROUPS:
        return NUMBERS[1:] if n == 1 else [';']
    # a statement about a person
    if n == 1:
        return _AFTER_ENT
    if cur[1] == 'has':
        return NUMBERS if n == 2 else sided(3, kin_only=True)
    if last in ('male', 'female') or last in ATTRS:
        return [';']
    if last in ('opposite', 'same'):
        return others(a)
    if last == 'not':
        return ['only', *SIDES, *RELS]
    if last == 'only':
        return [*SIDES, *_ONLY]
    if last in SIDES:
        return _SIDED
    if last in KIN or last in NONKIN:
        return others(a)
    return [';']
