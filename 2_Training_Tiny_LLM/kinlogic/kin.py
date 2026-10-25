"""Family relation words: what each means as moves through a family, and the name of a path found in one.

Moves: U to a parent, D to a child, S to the spouse, B to a sibling (same two parents, another person).
A step's gender is that of the person it reaches: 'M', 'F' or None (either).
"""
from __future__ import annotations

import re
from typing import NamedTuple


class Step(NamedTuple):
    m: str
    g: str | None = None


def U(g=None): return Step('U', g)
def D(g=None): return Step('D', g)
def S(g=None): return Step('S', g)
def B(g=None): return Step('B', g)


# Each word lists its meanings as step paths; most have one. Uncle, aunt, nephew and niece are blood relatives
# (exam convention); an uncle by marriage is written as two facts ("father's sister's husband").
KIN: dict[str, list[list[Step]]] = {
    'father': [[U('M')]], 'mother': [[U('F')]], 'parent': [[U()]],
    'son': [[D('M')]], 'daughter': [[D('F')]], 'child': [[D()]],
    'brother': [[B('M')]], 'sister': [[B('F')]], 'sibling': [[B()]],
    'husband': [[S('M')]], 'wife': [[S('F')]], 'spouse': [[S()]],
    'grandfather': [[U(), U('M')]], 'grandmother': [[U(), U('F')]], 'grandparent': [[U(), U()]],
    'grandson': [[D(), D('M')]], 'granddaughter': [[D(), D('F')]], 'grandchild': [[D(), D()]],
    'uncle': [[U(), B('M')]], 'aunt': [[U(), B('F')]],
    'nephew': [[B(), D('M')]], 'niece': [[B(), D('F')]],
    'cousin': [[U(), B(), D()]],
    'second-cousin': [[U(), U(), B(), D(), D()]],
    'great-uncle': [[U(), U(), B('M')]], 'great-aunt': [[U(), U(), B('F')]],
    'grand-nephew': [[B(), D(), D('M')]], 'grand-niece': [[B(), D(), D('F')]],
    'father-in-law': [[S(), U('M')]], 'mother-in-law': [[S(), U('F')]], 'parent-in-law': [[S(), U()]],
    'son-in-law': [[D('F'), S('M')]], 'daughter-in-law': [[D('M'), S('F')]],
    'brother-in-law': [[S(), B('M')], [B('F'), S('M')]],
    'sister-in-law': [[S(), B('F')], [B('M'), S('F')]],
}
for _k in range(1, 4):
    _great = 'great-' * _k
    _ups, _downs = [U()] * (_k + 1), [D()] * (_k + 1)
    KIN[f'{_great}grandfather'] = [[*_ups, U('M')]]
    KIN[f'{_great}grandmother'] = [[*_ups, U('F')]]
    KIN[f'{_great}grandparent'] = [[*_ups, U()]]
    KIN[f'{_great}grandson'] = [[*_downs, D('M')]]
    KIN[f'{_great}granddaughter'] = [[*_downs, D('F')]]
    KIN[f'{_great}grandchild'] = [[*_downs, D()]]

# Words that take "only" ("the only son of"): the person is the one such relative.
# A claim to check ("J is the aunt of M") also holds for relatives by marriage; facts read from a text stay blood.
BY_MARRIAGE = {'uncle': [[Step('U', None), Step('B', 'F'), Step('S', 'M')]],
               'aunt': [[Step('U', None), Step('B', 'M'), Step('S', 'F')]],
               'nephew': [[Step('S', None), Step('B', None), Step('D', 'M')]],
               'niece': [[Step('S', None), Step('B', None), Step('D', 'F')]]}


def claim_paths(word, side=None):
    extra = [[p[0]._replace(g={'paternal': 'M', 'maternal': 'F'}[side]) if side and p[0].m == 'U' else p[0], *p[1:]]
             for p in BY_MARRIAGE.get(word, [])]
    return (paths_of(word, side) or []) + extra


ONLY = set(KIN)   # "only son", "only grandson", "only daughter-in-law", "only maternal uncle", ...
# Words that take a side ("maternal uncle"): the first step goes to the father (paternal) or mother (maternal).
SIDED = {w for w, ps in KIN.items() if all(p[0].m == 'U' and len(p) > 1 for p in ps)}
# Links that are not family; reported when no family link exists.
NONKIN = {'friend', 'neighbour', 'colleague', 'classmate', 'teacher', 'student', 'boss', 'employee', 'roommate',
          'doctor', 'patient', 'coach', 'teammate', 'partner', 'acquaintance', 'stranger'}


def paths_of(word: str, side: str | None = None) -> list[list[Step]] | None:
    """A word's step paths with a side applied ("maternal" makes the first step reach the mother)."""
    paths = KIN.get(word)
    if paths is None or (side and word not in SIDED):
        return None
    if not side:
        return paths
    return [[U('M' if side == 'paternal' else 'F'), *p[1:]] for p in paths]


# ---------- naming a path found in a family ----------

def _by(g, m, f, n):
    return m if g == 'M' else f if g == 'F' else n


def _pre(prefix, word):
    """'great-' + 'grand-nephew or grand-niece' -> 'great-grand-nephew or great-grand-niece' (both alternatives)."""
    return ' or '.join(prefix + w for w in word.split(' or '))


ORDINAL = ['', '', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh']
REMOVED = {1: 'once', 2: 'twice', 3: 'thrice'}


def describe(moves) -> str:
    """Spells a shortest path out as words: "mother's brother's son". moves: [(m, g)] with m in U, D, S."""
    words, i = [], 0
    while i < len(moves):
        m, g = moves[i][0], moves[i][1]
        if m == 'U' and i + 1 < len(moves) and moves[i + 1][0] == 'D':   # up then straight down: a sibling
            words.append(_by(moves[i + 1][1], 'brother', 'sister', 'sibling'))
            i += 2
            continue
        words.append({'U': _by(g, 'father', 'mother', 'parent'), 'D': _by(g, 'son', 'daughter', 'child'),
                      'S': _by(g, 'husband', 'wife', 'spouse')}[m])
        i += 1
    return "'s ".join(words)


# Hindi kinship words, matched on the path signature ("UF U? DM": mother's brother).
HINDI = [(re.compile(p), w) for p, w in [
    (r'^UM$', 'pita'), (r'^UF$', 'maa'), (r'^DM$', 'beta'), (r'^DF$', 'beti'), (r'^SM$', 'pati'), (r'^SF$', 'patni'),
    (r'^U. DM$', 'bhai'), (r'^U. DF$', 'behen'),
    (r'^UM UM$', 'dada'), (r'^UM UF$', 'dadi'), (r'^UF UM$', 'nana'), (r'^UF UF$', 'nani'),
    (r'^UM U. UM$', 'pardada'), (r'^UM U. UF$', 'pardadi'), (r'^UF U. UM$', 'parnana'), (r'^UF U. UF$', 'parnani'),
    (r'^DM DM$', 'pota'), (r'^DM DF$', 'poti'), (r'^DF DM$', 'nati'), (r'^DF DF$', 'natin'),
    (r'^UM U. DM$', 'chacha or tau'), (r'^UM U. DF$', 'bua'), (r'^UF U. DM$', 'mama'), (r'^UF U. DF$', 'mausi'),
    (r'^UM U. DM SF$', 'chachi or tai'), (r'^UM U. DF SM$', 'phupha'), (r'^UF U. DM SF$', 'mami'),
    (r'^UF U. DF SM$', 'mausa'),
    (r'^UM U. DM D.$', 'chachera'), (r'^UM U. DF D.$', 'phuphera'), (r'^UF U. DM D.$', 'mamera'),
    (r'^UF U. DF D.$', 'mausera'),
    (r'^U. DM DM$', 'bhatija'), (r'^U. DM DF$', 'bhatiji'), (r'^U. DF DM$', 'bhanja'), (r'^U. DF DF$', 'bhanji'),
    (r'^S. UM$', 'sasur'), (r'^S. UF$', 'saas'), (r'^DF SM$', 'damaad'), (r'^DM SF$', 'bahu'),
    (r'^U. DM SF$', 'bhabhi'), (r'^U. DF SM$', 'jija'),
    (r'^SF U. DM$', 'saala'), (r'^SF U. DF$', 'saali'), (r'^SM U. DM$', 'devar or jeth'), (r'^SM U. DF$', 'nanad'),
]]


def name_moves(moves) -> dict:
    """Names the relative reached by `moves`, a shortest path from A to B ([(m, g)], g = gender reached).

    Returns {key, side, label, desc, hindi, note, gender}: key is the English word ("uncle"; 'path' when English
    has no single word), label adds the side ("maternal uncle"), desc spells the path out."""
    moves = [(m[0], m[1]) for m in moves]
    g_b = moves[-1][1] if moves else None
    sig = ' '.join(m + (g or '?') for m, g in moves)
    hindi = next((w for re_, w in HINDI if re_.match(sig)), None)
    i, n = 0, len(moves)
    lead = n > 0 and moves[0][0] == 'S'
    i += lead
    u = d = 0
    while i < n and moves[i][0] == 'U':
        u, i = u + 1, i + 1
    while i < n and moves[i][0] == 'D':
        d, i = d + 1, i + 1
    trail = i < n and moves[i][0] == 'S'
    i += trail
    desc = describe(moves)
    word = note = None
    sided = False
    gr = lambda k: 'great-' * max(0, k)
    if i == n:
        if not lead and not trail:
            if u == 0 and d == 0:
                word = 'self'
            elif d == 0:
                word = _by(g_b, 'father', 'mother', 'parent') if u == 1 else \
                    _pre(gr(u - 2), _by(g_b, 'grandfather', 'grandmother', 'grandparent'))
                sided = u >= 2
            elif u == 0:
                word = _by(g_b, 'son', 'daughter', 'child') if d == 1 else \
                    _pre(gr(d - 2), _by(g_b, 'grandson', 'granddaughter', 'grandchild'))
            elif u == 1 and d == 1:
                word = _by(g_b, 'brother', 'sister', 'sibling')
            elif d == 1:
                word = _by(g_b, 'uncle', 'aunt', 'uncle or aunt') if u == 2 else \
                    _pre(gr(u - 3), _by(g_b, 'great-uncle', 'great-aunt', 'great-uncle or great-aunt'))
                sided = True
            elif u == 1:
                word = _by(g_b, 'nephew', 'niece', 'nephew or niece') if d == 2 else \
                    _pre(gr(d - 3), _by(g_b, 'grand-nephew', 'grand-niece', 'grand-nephew or grand-niece'))
            elif u == d and u <= 12:
                word = 'cousin' if u == 2 else f'{ORDINAL[u - 1]}-cousin'
                sided = True
            elif u == 2 and d == 3:
                word = f"cousin's {_by(g_b, 'son', 'daughter', 'child')}"
            elif u == 3 and d == 2:
                word = f"{_by(moves[0][1], 'father', 'mother', 'parent')}'s cousin"
            elif 2 <= min(u, d) <= 12:   # cousin chart: degree from the nearer side, "removed" by the generation gap
                n, k = min(u, d) - 1, abs(u - d)
                word = f"{'cousin' if n == 1 else ORDINAL[n] + '-cousin'} {REMOVED.get(k, f'{k} times')} removed"
        elif lead and not trail:
            if u == 0 and d == 0:
                word = _by(g_b, 'husband', 'wife', 'spouse')
            elif u == 1 and d == 0:
                word = _by(g_b, 'father-in-law', 'mother-in-law', 'parent-in-law')
            elif u == 2 and d == 0:
                word = _by(g_b, 'grandfather-in-law', 'grandmother-in-law', 'grandparent-in-law')
            elif u == 1 and d == 1:
                word = _by(g_b, 'brother-in-law', 'sister-in-law', 'sibling-in-law')
            elif u == 1 and d == 2:
                word, note = _by(g_b, 'nephew', 'niece', 'nephew or niece'), 'by marriage'
        elif not lead and trail:
            if u == 0 and d == 1:
                word = _by(g_b, 'son-in-law', 'daughter-in-law', 'child-in-law')
            elif u == 0 and d == 2:
                word = _by(g_b, 'grandson-in-law', 'granddaughter-in-law', 'grandchild-in-law')
            elif u == 1 and d == 1:
                word = _by(g_b, 'brother-in-law', 'sister-in-law', 'sibling-in-law')
            elif u == 2 and d == 1:
                word, note, sided = _by(g_b, 'uncle', 'aunt', 'uncle or aunt'), 'by marriage', True
        elif u == 1 and d == 1:
            word = _by(g_b, 'co-brother', 'co-sister', 'co-sibling')
    side = _by(moves[0][1], 'paternal', 'maternal', None) if sided and not lead and u >= 2 else None
    base = word or desc
    label = ' '.join(x for x in (side, base) if x) + (f' ({note})' if note else '')
    if hindi and hindi.endswith('era'):
        hindi += _by(g_b, ' bhai', ' behen', ' bhai or behen')
    return {'key': word or 'path', 'base': base, 'side': side, 'label': label, 'desc': desc, 'hindi': hindi,
            'note': note, 'gender': g_b}
