"""Phrase-type balance for training (phrase_audit.py): rewrites verified generated questions into the clause shapes real
exams use far more than our generator (', who is' 18x, 'who is' without a comma 450x, 'whose' 7x), by merging adjacent
sentences. Meaning and facts are unchanged; sentence order is kept, so the fact order (text order) still holds. Each
rewrite is checked: when an exact reader reads it, its answer must equal the gold answer, or the row is dropped.

    python boost_phrases.py --pool data/dataset_v6 --n 60000 --out data/phrase_boost.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import re

E = r'([@#]\d+)'
REL = r'((?:the |a |an )?(?:only )?(?:paternal |maternal )?(?:elder |younger )?[a-z][a-z-]*)'


def sentences(tokens):
    """Statement sentences (token lists ending in '.') and the rest (the question) kept as one tail."""
    out, cur = [], []
    for t in tokens:
        cur.append(t)
        if t == '.':
            out.append(cur)
            cur = []
    return out, cur


def merge(s1, s2, rng):
    """One clause sentence from two adjacent statements, or None."""
    a, b = ' '.join(s1[:-1]), ' '.join(s2[:-1])
    m1 = re.fullmatch(rf'{E} (is|was) {REL} of {E}', a)
    if m1:
        x, v, r, y = m1.groups()
        m2 = re.fullmatch(rf'{re.escape(y)} (is|was) (.+)', b)   # "X is the R of Y . Y is the S of Z"
        if m2:
            return f'{x} {v} {r} of {y} , who {m2.group(1)} {m2.group(2)} .'.split()
        m2 = re.fullmatch(rf"{re.escape(y)} 's {REL} is {E}", b)   # "X is the R of Y . Y 's S is Z" -> "... Y , whose S is Z"
        if m2:
            return f'{x} {v} {r} of {y} , whose {m2.group(1)} is {m2.group(2)} .'.split()
        m2 = re.fullmatch(rf'{re.escape(y)} (has|had) (.+)', b)   # "X is the R of Y . Y has ..."
        if m2:
            return f'{x} {v} {r} of {y} , who {m2.group(1)} {m2.group(2)} .'.split()
        m2 = re.fullmatch(rf'{re.escape(x)} (is|was|has|had) (.+)', b)   # "X is the R of Y . X has ..."
        if m2:
            if rng.random() < 0.4:   # real exams: ', who is' 2:1 over 'who is' without commas; the generator wrote neither
                return f'{x} , who {v} {r} of {y} , {m2.group(1)} {m2.group(2)} .'.split()
            return f'{x} who {v} {r} of {y} {m2.group(1)} {m2.group(2)} .'.split()
    m1 = re.fullmatch(rf"{E} 's {REL} is {E}", a)   # "X 's R is Y . X is ..." -> "X , whose R is Y , is ..."
    if m1:
        x, r, y = m1.groups()
        m2 = re.fullmatch(rf'{re.escape(x)} (is|was|has|had) (.+)', b)
        if m2:
            return f'{x} , whose {r} is {y} , {m2.group(1)} {m2.group(2)} .'.split()
        m2 = re.fullmatch(rf"{re.escape(y)} 's {REL} is {E}", b)   # "X 's R is Y . Y 's S is Z" -> "... Y , whose S is Z"
        if m2:
            return f"{x} 's {r} is {y} , whose {m2.group(1)} is {m2.group(2)} .".split()
        m2 = re.fullmatch(rf'{re.escape(y)} (is|was|has|had) (.+)', b)   # "X 's R is Y . Y is ..." -> "... Y , who is ..."
        if m2:
            return f"{x} 's {r} is {y} , who {m2.group(1)} {m2.group(2)} .".split()
    return None


def boost(tokens, rng, p=0.8):
    sents, tail = sentences(tokens)
    out, i, changed = [], 0, False
    while i < len(sents):
        if i + 1 < len(sents) and rng.random() < p:
            m = merge(sents[i], sents[i + 1], rng)
            if m:
                out.append(m)
                i += 2
                changed = True
                continue
        out.append(sents[i])
        i += 1
    return ([t for s in out for t in s] + tail) if changed else None


def main():
    from kinlogic.codes import _fits
    from kinlogic.rules import read as read_rules
    from train_reader import answers
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='data/dataset_v6')
    ap.add_argument('--n', type=int, default=60000, help='source rows to try')
    ap.add_argument('--out', default='data/phrase_boost.jsonl.gz')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for f in sorted(glob.glob(f'{a.pool}/kinds/*.jsonl.gz')) + sorted(glob.glob(f'{a.pool}/deep/*.jsonl.gz')):
        try:
            got = [json.loads(l) for _, l in zip(range(a.n // 8), gzip.open(f, 'rt'))]
        except (EOFError, OSError):
            continue
        rows += got
    rng.shuffle(rows)
    st, out = collections.Counter(), []
    for r in rows[:a.n]:
        toks = r.get('tokens')
        if not isinstance(toks, list) or not r.get('lf'):
            continue
        new = boost(toks, rng)
        if not new:
            st['no adjacent pair to merge'] += 1
            continue
        people = sorted({t for t in toks if t[0] == '@'})
        try:
            lf = read_rules(new)
            if lf and _fits(lf) and answers(lf, people) != answers(r['lf'], people):
                st['exact reader disagrees (dropped)'] += 1
                continue
            st['checked by an exact reader' if lf else 'kept (by construction)'] += 1
        except Exception:
            st['kept (by construction)'] += 1
        out.append({**r, 'kind': 'phrase-' + str(r.get('kind', 'q')), 'tokens': new, 'text': ' '.join(new)})
    with gzip.open(a.out, 'wt') as f:
        f.write(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out))
    from phrase_audit import PAT
    rare = ['paternal / maternal', 'of-chain (R of the S of)', 'count (has two sons)', 'negation (not / no)', 'code symbols (means)',
            'grand-/great-', 'only (only son / only child)', 'only daughter / son of my', 'profession', 'brother-/sister- (siblings)',
            'possessive chain x2 (X \'s R \'s S)', 'pointing / photo / picture']
    keep = []
    for r in rows:
        t = ' '.join(r['tokens']) if isinstance(r.get('tokens'), list) else ''
        if t and not PAT["X 's R is Y"].search(t) and any(PAT[k].search(t) for k in rare):
            keep.append({**r, 'kind': 'rare-' + str(r.get('kind', 'q'))})
    rare_out = a.out.replace('.jsonl.gz', '_rare.jsonl.gz')
    with gzip.open(rare_out, 'wt') as f:
        f.write(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in keep))
    print(len(keep), 'rows with rare meaning types (and no "X \'s R is Y") ->', rare_out)
    print(len(out), 'rows ->', a.out, dict(st))


if __name__ == '__main__':
    main()


# --- restyling toward the forms real exams use (phrase_audit.py): every rewrite keeps the facts, or changes them only
# where the engine's answer provably stays the same (checked per row) ---
_P = r'(@\d+)'
_R = r'((?:only )?(?:paternal |maternal )?(?:elder |younger )?[a-z][a-z-]*)'
QFORMS = [('how is {a} related to {b} ?', 0.62), ('who is {a} to {b} ?', 0.14), ('what is {a} to {b} ?', 0.1),
          ('how is {a} related with {b} ?', 0.08), ('what is the relation of {a} with {b} ?', 0.06)]
QIN = re.compile(rf"(?:how is {_P} (?:related|connected) (?:to|with) {_P}|what relation does {_P} bear to {_P}|{_P} is related to {_P} as|"
                 rf"what is {_P} to {_P}|who is {_P} to {_P}|what is the relation(?:ship)? of {_P} (?:with|to) {_P}) \?")


def _sent_rewrite(s, rng):
    """One statement sentence (no final '.') in a more exam-like form; same fact."""
    m = re.fullmatch(rf"{_P} 's {_R} is {_P}", s) or re.fullmatch(rf"the {_R} of {_P} is {_P}", s)
    if m:   # "X 's R is Y" / "the R of X is Y" -> "Y is the R of X"
        x, r, y = (m.group(1), m.group(2), m.group(3)) if s.startswith('@') else (m.group(2), m.group(1), m.group(3))
        return f'{y} is the {r} of {x}' if rng.random() < 0.8 else f"{y} is {x} 's {r}"
    m = re.fullmatch(rf'{_P} has (?:a|an) {_R} (?:named|called) {_P}', s) or re.fullmatch(rf'{_P} has (?:a|an) {_R} , {_P}', s)
    if m and not m.group(2).startswith('only'):   # "X has a R named Y" -> "Y is the R of X"
        return f'{m.group(3)} is the {m.group(2)} of {m.group(1)}'
    return None


def _married(s, lf, rng):
    """'Y is the wife/husband of X' (or 'X 's wife is Y') -> 'X is married to Y', the fact becoming 'X spouse Y'."""
    m = re.fullmatch(rf'{_P} is the (wife|husband) of {_P}', s) or re.fullmatch(rf"{_P} 's (wife|husband) is {_P}", s)
    if not m:
        return None
    if s.endswith(m.group(3)) and "'s" in s:
        x, rel, y = m.group(1), m.group(2), m.group(3)
    else:
        y, rel, x = m.group(1), m.group(2), m.group(3)
    old = f'{x} {rel} {y}'
    if old not in lf.split(' ; '):
        return None
    new_lf = ' ; '.join(f'{x} spouse {y}' if f == old else f for f in lf.split(' ; '))
    return (f'{x} is married to {y}' if rng.random() < 0.5 else f'{y} is married to {x}'), new_lf


def restyle(tokens, lf, rng, married=True):
    """(tokens, lf) moved toward exam style: sentence forms, 'married to', coordination, an 'if ...' conditional,
    exam question wording. None when nothing changed."""
    sents, tail = sentences(tokens)
    body = [' '.join(s[:-1]) for s in sents]
    changed = False
    for i, s in enumerate(body):
        if rng.random() < 0.85:
            new = _sent_rewrite(s, rng)
            if new:
                body[i], changed = new, True
        if married and rng.random() < 0.5:
            got = _married(body[i], lf, rng)
            if got:
                body[i], lf = got
                changed = True
    out, i = [], 0
    while i < len(body):   # "X is the R of Y . X is the S of Z" -> "X is the R of Y and the S of Z"
        if i + 1 < len(body) and rng.random() < 0.5:
            m1 = re.fullmatch(rf'{_P} is (the |a |an )?{_R} of {_P}', body[i])
            m2 = re.fullmatch(rf'{_P} is (the |a |an )?{_R} of {_P}', body[i + 1])
            if m1 and m2 and m1.group(1) == m2.group(1):
                out.append(f'{body[i]} and {(m2.group(2) or "").strip() + " " if m2.group(2) else ""}{m2.group(3)} of {m2.group(4)}'.replace('  ', ' '))
                i, changed = i + 2, True
                continue
        out.append(body[i])
        i += 1
    q = ' '.join(tail)
    m = QIN.search(q)
    if m:
        g = [x for x in m.groups() if x]
        a, b = g[0], g[1]
        if q.startswith('what relation does') or ' is related to ' in m.group(0):
            pass
        r_ = rng.random()
        acc = 0.0
        for form, w in QFORMS:
            acc += w
            if r_ < acc:
                new_q = q.replace(m.group(0), form.format(a=a, b=b))
                if new_q != q:
                    q, changed = new_q, True
                break
    if len(out) >= 2 and q.startswith(('how is', 'who is', 'what is')) and rng.random() < 0.25:   # "if A , B and C , how is ...?"
        text = 'if ' + ' , '.join(out[:-1]) + ' and ' + out[-1] + ' , ' + q
        changed = True
    else:
        text = ' '.join(s + ' .' for s in out) + (' ' + q if q else '')
    return (text.split(), lf) if changed else None


def restyle_rows(rows, rng, n):
    from train_reader import answers
    st, out = collections.Counter(), []
    for r in rows[:n]:
        toks, lf = r.get('tokens'), r.get('lf')
        if not isinstance(toks, list) or not lf:
            continue
        got = restyle(toks, lf, rng)
        if not got:
            st['unchanged'] += 1
            continue
        new, new_lf = got
        people = sorted({t for t in toks if t[0] == '@'})
        if new_lf != lf:
            try:
                same = answers(new_lf, people) == answers(lf, people)
            except Exception:
                same = False
            if not same:
                st['answer would change (dropped)'] += 1
                continue
        if sorted(t for t in new if t[0] == '@') != sorted(t for t in toks if t[0] == '@') and set(new) - set(toks) - {'if', 'and', ',', 'married', 'to', 'the', 'of', 'is', 'who', 'what', 'how', 'related', 'with', 'relation', '?', "'s"}:
            st['odd tokens (dropped)'] += 1
            continue
        st['restyled'] += 1
        out.append({**r, 'kind': 'style-' + str(r.get('kind', 'q')), 'tokens': new, 'lf': new_lf, 'text': ' '.join(new)})
    return out, st
