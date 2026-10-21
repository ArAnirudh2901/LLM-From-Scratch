"""Exact family logic.

From the reader's facts, builds every family that fits them, answers the questions in each, and reports one
answer, answers that depend on something the text leaves open (usually a gender), "cannot be determined",
no family link, or a contradiction.

Conventions (standard for blood-relation exams):
  - a child's two parents are married to each other; one spouse each; spouses are a man and a woman
  - no step- or half-relations: a spouse's child is one's own child; siblings share both parents
  - different names are different people; an unnamed person is someone new unless the facts force otherwise
    ("the only son of my father" is the speaker himself)
  - nobody marries a sibling or an ancestor
  - when readings differ, the one with fewer people comes first
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .facts import is_named
from .kin import KIN, NONKIN, name_moves, paths_of

OPP = {'M': 'F', 'F': 'M'}
SLOT = {'M': 'fa', 'F': 'mo'}


class World:
    """One family under construction. People are ints; merged people share a union-find root."""
    __slots__ = ('uf', 'gen', 'fa', 'mo', 'sp', 'named', 'ent', 'frame', 'distinct', 'choices', 'nonkin')

    def __init__(self):
        self.uf, self.gen, self.fa, self.mo, self.sp = [], [], [], [], []
        self.named, self.ent, self.frame = [], [], []
        self.distinct, self.choices, self.nonkin = [], [], []

    def clone(self) -> World:
        w = World.__new__(World)
        for k in World.__slots__:
            setattr(w, k, getattr(self, k).copy())
        return w

    def add(self, ent: str | None = None, frame: bool = False) -> int:
        """A new person. ent: the fact-language name (@k, #k) or None; frame: a parent made only so that two
        siblings can share one."""
        i = len(self.uf)
        self.uf.append(i); self.gen.append(None); self.fa.append(-1); self.mo.append(-1); self.sp.append(-1)
        self.named.append(bool(ent) and is_named(ent)); self.ent.append(ent); self.frame.append(frame)
        return i

    def find(self, i: int) -> int:
        uf = self.uf
        while uf[i] != i:
            uf[i] = uf[uf[i]]
            i = uf[i]
        return i

    def g(self, i):
        return self.gen[self.find(i)]

    def rel(self, i, slot):
        v = getattr(self, slot)[self.find(i)]
        return -1 if v < 0 else self.find(v)

    def father(self, i): return self.rel(i, 'fa')
    def mother(self, i): return self.rel(i, 'mo')
    def spouse(self, i): return self.rel(i, 'sp')

    def roots(self):
        return [i for i, p in enumerate(self.uf) if p == i]

    def children(self, p):
        p = self.find(p)
        return [c for c in self.roots() if self.father(c) == p or self.mother(c) == p]

    def siblings(self, x):
        x = self.find(x)
        f, m = self.father(x), self.mother(x)
        if f < 0 and m < 0:
            return []
        return [c for c in self.roots() if c != x and ((f >= 0 and self.father(c) == f) or (m >= 0 and self.mother(c) == m))]

    # ----- changes; each returns False on a contradiction -----

    def set_g(self, i, g) -> bool:
        if not g:
            return True
        i = self.find(i)
        if self.gen[i]:
            return self.gen[i] == g
        self.gen[i] = g
        s = self.spouse(i)
        return s < 0 or self.set_g(s, OPP[g])

    def set_parent(self, c, p, slot) -> bool:
        c, p = self.find(c), self.find(p)
        if c == p or not self.set_g(p, 'M' if slot == 'fa' else 'F'):
            return False
        cur = self.rel(c, slot)
        if cur >= 0:
            return self.unify(cur, p)
        getattr(self, slot)[c] = p
        return True

    def marry(self, a, b) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.gen[a] and not self.set_g(b, OPP[self.gen[a]]):
            return False
        if self.gen[b] and not self.set_g(a, OPP[self.gen[b]]):
            return False
        sa = self.spouse(a)
        if sa >= 0 and sa != b and not self.unify(sa, b):
            return False
        a, b = self.find(a), self.find(b)
        sb = self.spouse(b)
        if sb >= 0 and sb != a and not self.unify(sb, a):
            return False
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.sp[a], self.sp[b] = b, a
        ga, gb = self.gen[a], self.gen[b]
        if ga and gb:
            return ga != gb
        return self.set_g(b, OPP[ga]) if ga else self.set_g(a, OPP[gb]) if gb else True

    def unify(self, a, b) -> bool:
        """Makes two people one. Fails for two different names or clashing facts."""
        a, b = self.find(a), self.find(b)
        if a == b:
            return True
        if self.named[a] and self.named[b]:
            return False
        if self.named[b] or (not self.named[a] and b < a):
            a, b = b, a
        gb = self.gen[b]
        if gb:
            if self.gen[a] and self.gen[a] != gb:
                return False
            self.gen[a] = gb
        self.uf[b] = a
        self.frame[a] = self.frame[a] and self.frame[b]
        pairs = []
        for slot in ('fa', 'mo', 'sp'):
            arr = getattr(self, slot)
            if arr[b] < 0:
                continue
            if arr[a] < 0:
                arr[a] = arr[b]
            else:
                pairs.append((arr[a], arr[b]))
        for x, y in pairs:
            if not self.unify(x, y):
                return False
        a = self.find(a)
        s = self.spouse(a)
        if s >= 0:
            if s == a:
                return False
            self.sp[s] = a
            ga, gs = self.gen[a], self.gen[s]
            if ga and gs:
                return ga != gs
            if ga or gs:   # spouses are a man and a woman
                return self.set_g(s, OPP[ga]) if ga else self.set_g(a, OPP[gs])
        return True

    def sibling(self, x, y) -> bool:
        """y becomes x's sibling: they share both parents (unnamed ones are made if none are known)."""
        x, y = self.find(x), self.find(y)
        if x == y:
            return False
        self.distinct.append((x, y))
        for slot, g in (('fa', 'M'), ('mo', 'F')):
            px, py = self.rel(x, slot), self.rel(y, slot)
            if px >= 0 and py >= 0:
                ok = self.unify(px, py)
            elif px >= 0:
                ok = self.set_parent(y, px, slot)
            elif py >= 0:
                ok = self.set_parent(x, py, slot)
            else:
                p = self.add(None, frame=True)
                self.gen[p] = g
                ok = self.set_parent(x, p, slot) and self.set_parent(y, p, slot)
            if not ok:
                return False
        return True

    def normalize(self) -> bool:
        """Keeps the conventions true: a child's parents are married; a parent's spouse is the other parent."""
        changed = True
        while changed:
            changed = False
            for c in self.roots():
                if self.uf[c] != c:
                    continue
                f, m = self.father(c), self.mother(c)
                if f >= 0 and m >= 0:
                    if self.spouse(f) != m:
                        if not self.marry(f, m):
                            return False
                        changed = True
                elif f >= 0 or m >= 0:
                    other = self.spouse(f if f >= 0 else m)
                    if other >= 0:
                        if not self.set_parent(c, other, 'mo' if f >= 0 else 'fa'):
                            return False
                        changed = True
        return True

    # ----- reading the family -----

    def is_ancestor(self, a, b) -> bool:
        """Is a an ancestor of b?"""
        a = self.find(a)
        seen, stack = set(), [self.find(b)]
        while stack:
            x = stack.pop()
            for p in (self.father(x), self.mother(x)):
                if p < 0 or p in seen:
                    continue
                if p == a:
                    return True
                seen.add(p)
                stack.append(p)
        return False

    def valid(self) -> bool:
        """Nobody is their own ancestor; spouses are not siblings or ancestors; siblings stayed two people."""
        for a, b in self.distinct:
            if self.find(a) == self.find(b):
                return False
        for x in self.roots():
            if self.is_ancestor(x, x):
                return False
            s = self.spouse(x)
            if s >= 0 and x < s:
                if self.is_ancestor(x, s) or self.is_ancestor(s, x):
                    return False
                fx, mx = self.father(x), self.mother(x)
                if (fx >= 0 and fx == self.father(s)) or (mx >= 0 and mx == self.mother(s)):
                    return False
        return True

    def follow(self, x, path):
        """People reached from x along one meaning of a word. Unknown genders do not match a gendered step."""
        cur = [self.find(x)]
        for st in path:
            nxt = []
            for c in cur:
                if st.m == 'U':
                    out = [self.father(c), self.mother(c)]
                elif st.m == 'D':
                    out = self.children(c)
                elif st.m == 'S':
                    out = [self.spouse(c)]
                else:
                    out = self.siblings(c)
                for o in out:
                    if o >= 0 and (not st.g or self.g(o) == st.g) and o not in nxt:
                        nxt.append(o)
            cur = nxt
        return cur

    def relatives(self, x, word, side=None):
        paths = paths_of(word, side)
        if not paths:
            return []
        x = self.find(x)
        out = []
        for p in paths:
            for r in self.follow(x, p):
                if r != x and r not in out:
                    out.append(r)
        return out

    def path(self, a, b):
        """Shortest path from a to b through parents, children and spouses; blood steps preferred on ties.
        Returns [(move, gender reached, person)], [] when a is b, None when unconnected."""
        a, b = self.find(a), self.find(b)
        if a == b:
            return []
        kids = {}
        for c in self.roots():
            for p in (self.father(c), self.mother(c)):
                if p >= 0:
                    kids.setdefault(p, []).append(c)
        best = {a: (0, -1, None)}
        heap = [(0, a)]
        while heap:
            cost, x = heapq.heappop(heap)
            if cost > best[x][0]:
                continue
            if x == b:
                break
            moves = [('U', self.father(x)), ('U', self.mother(x))] + [('D', c) for c in kids.get(x, [])] + [('S', self.spouse(x))]
            for m, y in moves:
                if y < 0:
                    continue
                c = cost + 100 + (m == 'S')
                if y not in best or c < best[y][0]:
                    best[y] = (c, x, m)
                    heapq.heappush(heap, (c, y))
        if b not in best:
            return None
        out, x = [], b
        while x != a:
            _, prev, m = best[x]
            out.append((m, self.g(x), x))
            x = prev
        return out[::-1]


# ---------- applying facts ----------

def _step(w: World, cur: int, st, to: int):
    """One step from cur to `to` (a person, or -1 for someone new). Returns [(world, reached)]."""
    out = []

    def fresh(choice):
        w2 = w.clone()
        if choice:
            w2.choices.append(choice)
        return w2, (to if to >= 0 else w2.add())

    if st.m == 'U':
        known = st.g or (w.g(to) if to >= 0 else None)
        for g in [known] if known else ['M', 'F']:
            choice = None if known else ('gender', w.find(to), g) if to >= 0 else ('side', w.find(cur), g)
            w2, t = fresh(choice)
            if w2.set_parent(cur, t, SLOT[g]) and w2.normalize():
                out.append((w2, w2.find(t)))
    elif st.m == 'D':
        gc = w.g(cur)
        for g in [gc] if gc else ['M', 'F']:
            w2, t = fresh(None if gc else ('gender', w.find(cur), g))
            if w2.set_g(cur, g) and w2.set_g(t, st.g) and w2.set_parent(t, cur, SLOT[g]) and w2.normalize():
                out.append((w2, w2.find(t)))
    else:
        w2, t = fresh(None)
        ok = w2.set_g(t, st.g) and (w2.marry(cur, t) if st.m == 'S' else w2.sibling(cur, t)) and w2.normalize()
        if ok:
            out.append((w2, w2.find(t)))
    return out


def _apply_path(w, x, path, y):
    states = [(w, x)]
    for k, st in enumerate(path):
        to = y if k == len(path) - 1 else -1
        states = [s for w0, cur in states for s in _step(w0, cur, st, to)]
    return [w1 for w1, _ in states]


def _apply_fact(w: World, f: dict, ids: dict):
    """The worlds one fact can lead to (empty: it contradicts)."""
    if f['t'] == 'gender':
        w2 = w.clone()
        return [w2] if w2.set_g(ids[f['x']], f['g']) and w2.normalize() else []
    if f['r'] in NONKIN:
        w2 = w.clone()
        w2.nonkin.append((ids[f['x']], f['r'], ids[f['y']]))
        return [w2]
    return [w2 for p in (paths_of(f['r'], f.get('side')) or []) for w2 in _apply_path(w, ids[f['x']], p, ids[f['y']])]


@dataclass
class _Run:
    ids: dict
    max_steps: int
    steps: int = 0
    cut: bool = False
    onlys: list = field(default_factory=list)
    attrs: list = field(default_factory=list)
    after: list = field(default_factory=list)
    negs: list = field(default_factory=list)
    links: list = field(default_factory=list)
    named: list = field(default_factory=list)
    members: int | None = None
    couples: int | None = None
    males: int | None = None
    females: int | None = None
    generations: int | None = None
    has: list = field(default_factory=list)
    extra: list = field(default_factory=list)

    def tick(self) -> bool:
        self.steps += 1
        if self.steps > self.max_steps:
            self.cut = True
        return not self.cut


def _search(w, facts, run: _Run):
    """Facts with one outcome are applied at once; then the fact with the fewest outcomes branches."""
    rest = facts
    while True:
        if not run.tick():
            return
        progressed, keep = False, []
        for f in rest:
            outs = _apply_fact(w, f, run.ids)
            if not outs:
                return
            if len(outs) == 1:
                w, progressed = outs[0], True
            else:
                keep.append((f, outs))
        rest = [f for f, _ in keep]
        if not rest:
            yield from _finish(w, run)
            return
        if not progressed:
            f, outs = min(keep, key=lambda k: len(k[1]))
            others = [g for g in rest if g is not f]
            for w2 in outs:
                yield from _search(w2, others, run)
            return


def _finish(w, run: _Run):
    """Constraints that need the whole family: "only", exact counts, closed families, gender totals and links,
    then the final checks (negations, attributes, counts, generations)."""
    ids = run.ids
    closed = run.members is not None

    def counted(w1):   # in a closed family, unnamed relatives become members first, then counts are checked
        if closed:
            for w2 in _closed_family(w1, run):
                yield from _has(w2, run.has, run, create=False)
        else:
            for w2 in _has(w1, run.has, run):
                yield from _closed_family(w2, run)

    for w2 in _identities(w, run):
        for w4 in counted(w2):
                for w5 in _gender_totals(w4, run):
                    for w6 in _gender_links(w5, run.links, run):
                        if not w6.valid():
                            continue
                        if not _negations(w6, run):
                            continue
                        if not all(_attr_holds(w6, ids[f['x']], f['a']) for f in run.after):
                            continue
                        if any(len(w6.relatives(ids[f['x']], f['r'], f.get('side'))) != f['n'] for f in run.has):
                            continue
                        if run.generations:
                            span, parts = _generations(w6, run)
                            # separate groups may sit on any levels relative to each other: only too many is wrong
                            if span > run.generations or (parts == 1 and span != run.generations):
                                continue
                        yield w6


def _negations(w, run: _Run) -> bool:
    """ "y is not x's REL": fails if y is; if y would be with the word's gender, y gets the other gender."""
    for f in run.negs:
        x, y = run.ids[f['x']], w.find(run.ids[f['y']])
        if f['r'] in NONKIN:
            continue
        if y in w.relatives(x, f['r'], f.get('side')):
            return False
        paths = paths_of(f['r'], f.get('side')) or []
        want = paths[0][-1].g if paths else None
        if want and not w.g(y):
            neutral = [[*q[:-1], q[-1]._replace(g=None)] for q in paths]
            if any(y in w.follow(x, q) for q in neutral) and not (w.set_g(y, OPP[want]) and w.normalize()):
                return False
    return True


def _attr_holds(w, x, a):
    if a == 'unmarried':
        return w.spouse(x) < 0
    if a == 'no-child':
        return not w.children(x)
    if a == 'married':
        return w.spouse(x) >= 0
    return True


def _identities(w, run: _Run):
    """ "only son", "no brother or sister": relatives that must be one person. Unknown genders branch."""
    for _ in range(200):
        if not run.tick():
            return
        v = _violation(w, run)
        if v is None:
            yield w
            return
        if v == 'fail':
            return
        kind, z, y, want = v
        if kind == 'merge':
            if not (w.unify(z, y) and w.normalize()):
                return
            continue
        for choices, apply in (([('gender', z, OPP[want])], lambda w2: w2.set_g(z, OPP[want])),
                               ([('gender', z, want), ('same', z, y)], lambda w2: w2.unify(z, y))):
            w2 = w.clone()
            w2.choices.extend(choices)
            if apply(w2) and w2.normalize():
                yield from _identities(w2, run)
        return


def _violation(w, run: _Run):
    ids = run.ids
    for f in run.onlys:                       # "y is x's only son": every other son of x is y
        x, y = w.find(ids[f['x']]), w.find(ids[f['y']])
        want = KIN[f['r']][0][0].g
        pool = w.children(x) if f['r'] in ('son', 'daughter', 'child') else w.siblings(x)
        for z in pool:
            if z == y:
                continue
            gz = w.g(z)
            if not want or gz == want:
                return 'fail' if w.named[z] and w.named[y] else ('merge', z, y, want)
            if not gz:
                return ('branch', z, y, want)
    for f in run.attrs:                       # "x has no brother or sister": anyone sharing x's parents is x
        x = w.find(ids[f['x']])
        want = {'no-brother': 'M', 'no-sister': 'F'}.get(f['a'])
        for z in w.siblings(x):
            gz = w.g(z)
            if not want or gz == want:
                return 'fail' if w.named[z] and w.named[x] else ('merge', z, x, want)
            if not gz:
                return ('branch', z, x, want)
    return None


def _gender_links(w, links, run: _Run):
    """ "U and X are of opposite gender": unknown genders branch."""
    ids = run.ids
    for f in links:
        x, y = ids[f['x']], ids[f['y']]
        gx, gy = w.g(x), w.g(y)
        same = f['t'] == 'same'
        if gx and gy:
            if same != (gx == gy):
                return
            continue
        if gx:
            options = [[(y, gx if same else OPP[gx])]]
        elif gy:
            options = [[(x, gy if same else OPP[gy])]]
        else:
            options = [[(x, 'M'), (y, 'M' if same else 'F')], [(x, 'F'), (y, 'F' if same else 'M')]]
        rest = [l for l in links if l is not f]
        for opt in options:
            w2 = w.clone()
            ok = True
            for n, g in opt:
                if not w.g(n):
                    w2.choices.append(('gender', w.find(n), g))
                ok = ok and w2.set_g(n, g)
            if ok and w2.normalize():
                yield from _gender_links(w2, rest, run)
        return
    yield w


def _has(w, has, run: _Run, create=True):
    """ "x has N RELs": relatives with an unknown gender may be counted in or out; missing ones are added as
    new people (a closed family later makes them members)."""
    if not has:
        yield w
        return
    f, rest = has[0], has[1:]
    x, n = run.ids[f['x']], f['n']
    definite = w.relatives(x, f['r'], f.get('side'))
    if len(definite) > n:
        return
    want = (paths_of(f['r'], f.get('side')) or [[None]])[0][-1].g
    maybe = []
    if want:   # same word without its last gender: candidates whose gender is still open
        neutral = [[*p[:-1], p[-1]._replace(g=None)] for p in paths_of(f['r'], f.get('side'))]
        for path in neutral:
            for z in w.follow(x, path):
                if z != w.find(x) and z not in definite and z not in maybe and not w.g(z):
                    maybe.append(z)
    need = n - len(definite)
    from itertools import combinations
    for k in range(min(need, len(maybe)), -1, -1):
        for chosen in combinations(maybe, k):
            if not run.tick():
                return
            w2 = w.clone()
            ok = all(w2.set_g(z, want if z in chosen else OPP[want]) for z in maybe) and w2.normalize()
            worlds = [w2] if ok else []
            if need > k and not create:
                continue
            for _ in range(need - k):   # still short: new relatives
                path = (paths_of(f['r'], f.get('side')) or [[]])[0]
                worlds = [w4 for w3 in worlds for w4 in _apply_path(w3, w3.find(x), path, -1)] if path else []
            for w3 in worlds:
                yield from _has(w3, rest, run, create)


def _gender_totals(w, run: _Run):
    """ "three males and three females": members whose gender is open are assigned to match."""
    if run.males is None and run.females is None:
        yield w
        return
    members = _members(w, run)
    males = sum(1 for m in members if w.g(m) == 'M')
    females = sum(1 for m in members if w.g(m) == 'F')
    open_ = [m for m in members if not w.g(m)]
    total = len(members)
    want_m = run.males if run.males is not None else total - run.females
    need = want_m - males
    if need < 0 or need > len(open_) or (run.females is not None and run.females - females != len(open_) - need):
        return
    from itertools import combinations
    for chosen in combinations(open_, need):
        if not run.tick():
            return
        w2 = w.clone()
        if all(w2.set_g(m, 'M' if m in chosen else 'F') for m in open_) and w2.normalize():
            for m in open_:
                w2.choices.append(('gender', m, 'M' if m in chosen else 'F'))
            yield w2


def _generations(w, run: _Run) -> tuple[int, int]:
    """(generations spanned by the largest connected group of members, number of groups); parents one level up,
    spouses on the same level."""
    members = set(_members(w, run))
    level, todo = {}, []
    span, parts = 0, 0
    for start in members:
        if start in level:
            continue
        level[start] = 0
        todo = [start]
        comp = [start]
        while todo:
            x = todo.pop()
            steps = [(w.father(x), -1), (w.mother(x), -1), (w.spouse(x), 0)] + [(c, 1) for c in w.children(x)]
            for y, d in steps:
                if y >= 0 and y not in level:
                    level[y] = level[x] + d
                    todo.append(y)
                    comp.append(y)
        levels = [level[m] for m in comp if m in members]
        span = max(span, max(levels) - min(levels) + 1)
        parts += 1
    return span, parts


def _members(w, run):
    out = []
    for r in [run.ids[e] for e in run.named] + run.extra:
        r = w.find(r)
        if r not in out:
            out.append(r)
    return out


def _closed_family(w, run: _Run):
    """Closed-family puzzles ("a family of six: A, B, C, D, E and F"): every unnamed person mentioned or implied
    is one of the members (parents made only for siblings may be outside); then the couple count."""
    if run.members is None:
        yield from _couples(w, run)
        return
    members = _members(w, run)
    anon = next((r for r in w.roots() if not w.named[r] and not w.frame[r] and r not in members), None)
    if anon is None:
        if len(members) <= run.members:
            yield from _couples(w, run)
        return
    for m in members:
        if not run.tick():
            return
        w2 = w.clone()
        w2.choices.append(('same', anon, m))
        if w2.unify(anon, m) and w2.normalize() and w2.valid():
            yield from _closed_family(w2, run)


def _couples(w, run: _Run):
    if run.couples is None:
        yield w
        return
    members = _members(w, run)
    have = sum(1 for m in members if w.spouse(m) in members) // 2
    if have > run.couples:
        return
    if have == run.couples:
        yield w
        return
    free = [m for m in members if w.spouse(m) < 0]
    for i in range(len(free)):
        for j in range(i + 1, len(free)):
            if not run.tick():
                return
            w2 = w.clone()
            w2.choices.append(('couple', free[i], free[j]))
            if w2.marry(free[i], free[j]) and w2.normalize() and w2.valid():
                yield from _couples(w2, run)


# ---------- answers ----------

NEUTRAL = {'father': 'parent', 'mother': 'parent', 'son': 'child', 'daughter': 'child', 'brother': 'sibling',
           'sister': 'sibling', 'husband': 'spouse', 'wife': 'spouse', 'uncle': 'uncle or aunt',
           'aunt': 'uncle or aunt', 'nephew': 'nephew or niece', 'niece': 'nephew or niece',
           'grandfather': 'grandparent', 'grandmother': 'grandparent', 'grandson': 'grandchild',
           'granddaughter': 'grandchild'}


def relation_in(w: World, a: int, b: int) -> dict:
    """How b is related to a in one family."""
    p = w.path(a, b)
    if p is None:
        a, b = w.find(a), w.find(b)
        for x, r, y in w.nonkin:
            if {w.find(x), w.find(y)} == {a, b}:
                return {'key': f'not family: {r}', 'base': r, 'side': None, 'nonkin': r}
        return {'key': 'none', 'base': 'none', 'side': None}
    n = name_moves([(m, g) for m, g, _ in p])
    base = n['desc'] if n['key'] == 'path' else n['key']
    return {**n, 'base': base, 'key': f"{n['side']} {base}" if n['side'] else base, 'moves': p}


def _conditions(w: World, label) -> list[str]:
    """The choices behind a family that people can read ("Dinabandhu is a man")."""
    out = []
    for kind, node, val in w.choices:
        n = w.find(node)
        if not w.named[n]:
            continue
        if kind == 'gender':
            c = f"{label(w.ent[n])} is {'a man' if val == 'M' else 'a woman'}"
        elif kind == 'side':
            c = f"on {label(w.ent[n])}'s {'father' if val == 'M' else 'mother'}'s side"
        else:
            continue
        if c not in out:
            out.append(c)
    return out


def _group(results):
    """One entry per distinct answer: the smallest family behind it first; conditions every such family shares."""
    by = {}
    for r in results:
        g = by.get(r['key'])
        if g is None:
            by[r['key']] = {**r, 'families': 1, 'all_conds': [set(r['conds'])]}
        else:
            g['families'] += 1
            g['all_conds'].append(set(r['conds']))
            if r['size'] < g['size']:
                g.update({k: v for k, v in r.items() if k != 'conds'}, conds=r['conds'])
    # fewer people first, then the closer relation (fewer steps, blood before marriage)
    cost = lambda g: sum(100 + (m == 'S') for m, _, _ in g.get('moves') or [])
    out = sorted(by.values(), key=lambda g: (g.get('against', 0), g['size'], cost(g), -g['families']))
    for g in out:
        shared = set.intersection(*g.pop('all_conds'))
        g['conds'] = [c for c in g['conds'] if c in shared]
    return out


def _verdict(answers):
    """'unique', 'side' (paternal or maternal not given), 'gender' (only the asked person's gender is open),
    'conditional' (each answer has a stated
    condition), 'ambiguous', 'none' (no link), 'contradiction' (no family fits)."""
    if not answers:
        return {'status': 'contradiction', 'answers': []}
    common = set.intersection(*(set(a['conds']) for a in answers))
    for a in answers:   # a condition every answer shares (or a forced one) explains nothing
        a['conds'] = [c for c in a['conds'] if c not in common]
    if len(answers) == 1:
        return {'status': 'none' if answers[0]['key'] == 'none' else 'unique', 'answers': answers}
    if answers[0].get('against', 0) == 0 and all(a.get('against', 0) > 0 for a in answers[1:]):
        return {'status': 'assumed', 'answers': answers}   # names decide: the other readings go against them
    if len({a.get('base') for a in answers}) == 1 and all(a.get('side') for a in answers):
        return {'status': 'side', 'answers': answers, 'neutral': answers[0]['base']}   # side not given
    if all(a.get('base') in NEUTRAL for a in answers):
        neutral = {(a.get('side'), NEUTRAL[a['base']]) for a in answers}
        if len(neutral) == 1:
            side, word = next(iter(neutral))
            return {'status': 'gender', 'answers': answers, 'neutral': f'{side} {word}' if side else word}
    return {'status': 'conditional' if all(a['conds'] for a in answers) else 'ambiguous', 'answers': answers}


def _assume(w: World, priors: dict, ids: dict):
    """Applies name genders where the family leaves a gender open and they fit. Returns (world, assumed, against):
    assumed: people given their name's gender; against: people whose settled gender differs from their name's."""
    w2, assumed, against = w, [], 0
    for e, g in priors.items():
        if e not in ids:
            continue
        cur = w2.g(ids[e])
        if cur:
            against += cur != g
            continue
        trial = w2.clone()
        if trial.set_g(ids[e], g) and trial.normalize() and trial.valid():
            w2 = trial
            assumed.append(e)
    return w2, assumed, against


def solve(stmts: list[dict], label=lambda e: e, asks=(), parts=None, people=(), priors=None, max_steps: int = 20000,
          max_families: int = 2000) -> dict:
    """Solves parsed fact statements (facts.parse).

    label(ent) -> display name used in conditions; asks: extra (x, y) pairs to answer (people the user picked);
    parts: the data-sufficiency statements to use (None: all); people: every placeholder in the text (a puzzle's
    members include those no fact mentions); priors: {placeholder: 'M' | 'F'} genders suggested by names, used only
    where the facts leave a gender open (readings that agree with them come first).
    Returns {ok, complete, families, questions: [{q, status, answers, neutral?}], claims: {n: verdict}, worlds, ids}."""
    if parts is not None:
        stmts = [s for s in stmts if not s.get('part') or s['part'] in parts]
    premise = [s for s in stmts if not s.get('claim')]
    claims = {}
    for s in stmts:
        if s.get('claim'):
            claims.setdefault(s['claim'], []).append(s)
    w0 = World()
    ents = []
    for s in premise:
        for k in ('x', 'y'):
            if s.get(k) and s[k] not in ents:
                ents.append(s[k])
    for group in claims.values():   # people a claim names; its unnamed people are bound when checking
        for s in group:
            for k in ('x', 'y'):
                if s.get(k) and s[k][0] == '@' and s[k] not in ents:
                    ents.append(s[k])
    for x, y in asks:
        for e in (x, y):
            if e not in ents:
                ents.append(e)
    for e in people:
        if e not in ents:
            ents.append(e)
    ids = {e: w0.add(e) for e in ents}
    run = _Run(ids=ids, max_steps=max_steps, named=[e for e in ents if is_named(e)])
    members = next((s['n'] for s in premise if s['t'] == 'members'), None)
    for _ in range((members or 0) - len(run.named)):   # members the text never names
        run.extra.append(w0.add())
    facts, questions = [], []
    for s in premise:
        t = s['t']
        if t in ('ask', 'who', 'count', 'gender?', 'pair'):
            questions.append(s)
        elif t == 'rel':
            if s.get('neg'):
                run.negs.append(s)
            else:
                facts.append(s)
                if s.get('only'):
                    run.onlys.append(s)
        elif t == 'gender':
            facts.append(s)
        elif t == 'attr':
            (run.after if s['a'] in ('unmarried', 'no-child', 'married') else run.attrs).append(s)
        elif t in ('same', 'opposite'):
            run.links.append(s)
        elif t == 'has':
            run.has.append(s)
        elif t in ('members', 'couples', 'males', 'females', 'generations'):
            setattr(run, t, s['n'])
    questions += [{'t': 'ask', 'x': x, 'y': y, 'picked': True} for x, y in asks]
    worlds = []
    for w in _search(w0, facts, run):
        worlds.append(w)
        if len(worlds) >= max_families:
            run.cut = True
            break
    out = {'ok': bool(worlds), 'complete': not run.cut, 'families': len(worlds), 'questions': [],
           'worlds': worlds, 'ids': ids, 'claims': {}}
    for k, group in claims.items():
        held = [claim_holds(w, group, ids) for w in worlds]
        out['claims'][k] = ('contradiction' if not held else 'true' if all(v is True for v in held)
                            else 'false' if all(v is False for v in held) else 'possible')
    assumed_worlds = [_assume(w, priors or {}, ids) for w in worlds]
    for q in questions:
        if q['t'] == 'pair':   # answered per option (pair_holds)
            out['questions'].append({'q': q, 'status': 'options', 'answers': []})
            continue
        results = []
        for w0, (w, assumed, against) in zip(worlds, assumed_worlds):
            base = {'conds': _conditions(w0, label), 'size': len(w0.roots()), 'world': w, 'assumed': assumed,
                    'against': against}
            if q['t'] == 'ask':
                results.append({**relation_in(w, ids[q['x']], ids[q['y']]), **base})
            elif q['t'] == 'who':
                found = w.relatives(ids[q['x']], q['r'], q.get('side'))
                named = lambda ns: sorted(w.ent[n] for n in ns if w.ent[n] and is_named(w.ent[n]))
                names = named(found)
                key = ','.join(names) if names else 'unnamed' if found else 'none'
                results.append({'key': key, 'names': names, **base})
                maybe = [n for n in _open_gender_relatives(w, ids[q['x']], q['r'], q.get('side')) if n not in found]
                if named(maybe):   # people who are that relative if their open gender is the word's
                    more = named(found + maybe)
                    results.append({'key': ','.join(more), 'names': more, **base,
                                    'conds': base['conds'] + [f"{label(w.ent[n])}'s gender is not given" for n in maybe if w.ent[n]]})
            elif q['t'] == 'gender?':
                g = w.g(ids[q['x']])
                results.append({'key': {'M': 'male', 'F': 'female'}.get(g, 'unknown'), **base})
            else:
                lo, hi = _count(w, q, run)
                for n in range(lo, hi + 1):   # members whose gender is open could count either way
                    results.append({'key': str(n), 'n': n, **base})
        out['questions'].append({'q': q, **_verdict(_group(results))})
    return out


def _count(w, q, run) -> tuple[int, int]:
    """(fewest, most) the family allows."""
    if q['what'] == 'rel':
        x = run.ids[q['x']]
        n = len(w.relatives(x, q['r'], q.get('side')))
        return n, n + len(_open_gender_relatives(w, x, q['r'], q.get('side')))
    members = _members(w, run)
    if q['what'] == 'people':
        return len(members), len(members)
    if q['what'] == 'couples':
        n = sum(1 for m in members if w.spouse(m) in members) // 2
        return n, n
    g = 'M' if q['what'] == 'male' else 'F'
    n = sum(1 for m in members if w.g(m) == g)
    return n, n + sum(1 for m in members if not w.g(m))


def claim_holds(w: World, group: list[dict], ids: dict):
    """Does one claim (its facts) hold in a family? True, False, or None when it depends on a gender the family
    leaves open. Unnamed people the claim introduces may be anyone fitting."""
    facts = [f for f in group if f['t'] in ('rel', 'gender', 'attr', 'has', 'opposite', 'same')]
    bound = {e: w.find(i) for e, i in ids.items()}

    def rel_value(x, y, f):
        if f['r'] in NONKIN:
            return any({w.find(a), w.find(c)} == {x, y} and r == f['r'] for a, r, c in w.nonkin)
        rel = w.relatives(x, f['r'], f.get('side'))
        if y in rel:
            return True if not f.get('only') or rel == [y] else False
        paths = paths_of(f['r'], f.get('side')) or []
        neutral = [[st._replace(g=None) for st in q] for q in paths]
        reach = any(y in w.follow(x, q) for q in neutral)
        return None if reach and not w.g(y) else False

    def check(f, b):
        x, y = b.get(f.get('x')), b.get(f.get('y')) if f.get('y') else None
        t = f['t']
        if t == 'rel':
            v = rel_value(x, y, f)
            return v if v is None or not f.get('neg') else not v
        if t == 'gender':
            g = w.g(x)
            return None if not g else g == f['g']
        if t == 'attr':
            if f['a'] in ('no-sibling', 'no-brother', 'no-sister'):
                want = {'no-brother': 'M', 'no-sister': 'F'}.get(f['a'])
                sibs = w.siblings(x)
                if any(not want or w.g(z) == want for z in sibs):
                    return False
                return None if any(not w.g(z) for z in sibs) else True
            return _attr_holds(w, x, f['a'])
        if t == 'has':
            return len(w.relatives(x, f['r'], f.get('side'))) == f['n']
        gx, gy = w.g(x), w.g(y)
        return None if not (gx and gy) else (gx == gy) == (t == 'same')

    def value(i, b):
        if i == len(facts):
            return True
        f = facts[i]
        free = [k for k in ('x', 'y') if f.get(k) and f[k] not in b]
        if not free:
            v = check(f, b)
            if v is False:
                return False
            rest = value(i + 1, b)
            return None if v is None and rest is not False else rest
        best = False
        for cand in w.roots():   # bind an unnamed person of the claim to someone in the family
            v = value(i, {**b, f[free[0]]: cand})
            if v:
                return True
            if v is None:
                best = None
        return best

    return value(0, bound)


def pair_holds(w: World, what: str, x: int, y: int) -> bool:
    """Is (x, y) "a pair of females" / "a couple" / "a pair of brothers"?"""
    x, y = w.find(x), w.find(y)
    if x == y:
        return False
    if what in ('male', 'female'):
        g = 'M' if what == 'male' else 'F'
        return w.g(x) == g and w.g(y) == g
    if what == 'couple':
        return w.spouse(x) == y
    return y in w.relatives(x, what) and x in w.relatives(y, what) if what in ('brother', 'sister', 'sibling', 'cousin') \
        else y in w.relatives(x, what) or x in w.relatives(y, what)


def _open_gender_relatives(w: World, x, word, side=None):
    """Relatives reached by the word's steps when the last person's gender is still open."""
    paths = paths_of(word, side) or []
    out = []
    for q in paths:
        if not q[-1].g:
            continue
        for n in w.follow(x, [*q[:-1], q[-1]._replace(g=None)]):
            if not w.g(n) and n != w.find(x) and n not in out:
                out.append(n)
    return out


def alternatives(result: dict, q_index: int = 0, limit_worlds: int = 6) -> list[dict]:
    """Readings beyond the standard one, for when the expected answer is not among the options: an unnamed person
    (said in the text, or implied, like the parent an "uncle" goes through) turns out to be someone already there
    ("the daughter of my grandfather" = "my mother"). Returns extra answers (relation_in dicts), most likely first."""
    q = result['questions'][q_index]['q']
    if q['t'] != 'ask':
        return []
    ids = result['ids']
    out, seen = [], set(a['key'] for a in result['questions'][q_index]['answers'])
    for w in result['worlds'][:limit_worlds]:
        roots = w.roots()
        for e in roots:
            if w.named[e]:
                continue
            for f in roots:
                if f == e:
                    continue
                w2 = w.clone()
                if not (w2.unify(e, f) and w2.normalize() and w2.valid()):
                    continue
                r = relation_in(w2, ids[q['x']], ids[q['y']])
                if r['key'] not in seen:
                    seen.add(r['key'])
                    out.append({**r, 'merged': (w.ent[e] or 'an unnamed person', w.ent[f] or 'an unnamed person'),
                                'world': w2, 'size': len(w2.roots())})
    return sorted(out, key=lambda a: a['size'])
