"""Splits the real questions into fair and broken items and tags each with a category, so accuracy is reported on what
can be answered from the page and per kind of question (the ship target is measured on the fair items).

A question is broken when its scraped text holds a worked solution, several questions run together, a code expression
uses symbols that are never defined, or there is no question to answer. Rules are conservative: an item is flagged
only on a clear sign, and `--show` prints flagged items for spot checks.

    python -m eval.real_clean --results runs/v10-r3-reworded/real_k8.json [--show broken]
"""
from __future__ import annotations

import argparse
import collections
import json
import re

from eval.run_real import as_text, load_real

SOLUTION = re.compile(r'###|\b(?:solution|explanation|hence|therefore|thus)\b|\bi\.\s?e\.|\bans(?:wer)?\s*[:.\-]\s*\(?[a-e1-5]\b', re.I)
MULTI = re.compile(r'\b(?:q\s?\d+\s*[).:]|question\s*:?\s*\d+\b|\d+\s*\.\s*(?:how|who|what|which)\b)', re.I)
DEFINES = re.compile(r"(?:^|[\s'‘(])([A-Za-z])\s*([^\w\s'’,.()\-?])\s*([A-Za-z])\s*['’]?\s*(?:means|implies|denotes|indicates|stands for)", re.I)
EXPR_SYM = re.compile(r'\b[A-Z]\s*([^\w\s,.?()\'"’‘:;\-])\s*(?=[A-Z]\b)')
ASK = re.compile(r'\?|\bhow\b|\bwho\b|\bwhich\b|\bwhat\b|\brelat', re.I)


def _norm(t):
    return t.replace('×', 'x').replace('–', '-').replace('—', '-').replace('â€“', '-')


def broken(r) -> str | None:
    q = _norm(r['question'])
    if SOLUTION.search(q):
        return 'solution text'
    if len(MULTI.findall(q)) >= 1 and len(re.findall(r'\?', q)) >= 2:
        return 'several questions'
    used = set(EXPR_SYM.findall(q + ' ' + _norm(' '.join(t for _, t in r['options']))))
    if used and re.search(r'\bmeans?\b|\bimplies\b|\bdenotes\b', q, re.I):
        defined = {m[1] for m in DEFINES.findall(q)}
        defined |= set(re.findall(r"['‘]?\s*[A-Za-z]\s*([^\w\s'’,.()\-?])\s*[A-Za-z]\s*['’]?\s*(?:means|implies|denotes)", q))
        if defined and used - defined:
            return 'undefined code symbol'
    from kinlogic.codes import definitions, QUOTES
    from kinlogic.prepare import prepare
    if re.search(r'\bmeans?\b', q, re.I) and EXPR_SYM.search(q):
        toks = [t for t in prepare(q)['tokens'] if t not in QUOTES]
        if definitions(' '.join(toks)) is None:   # one symbol defined twice with different meanings: a symbol lost in scraping
            return 'conflicting code definitions'
    if not r['options'] and not ASK.search(q):
        return 'no question'
    if re.search(r'\bURL Source:|\bMarkdown Content:|\bPublished Time:', q):
        return 'page text, not a question'
    opts = [_norm(t) for _, t in r['options']]
    place = r'\b(?:north|south|east|west|to the (?:left|right)|(?:immediate )?(?:left|right) of|facing|km|metres?)\b'
    if opts and sum(bool(re.search(place, o, re.I)) for o in opts) * 2 > len(opts):
        return 'other topic (seating or direction)'   # options are places, not relations
    key = str(r.get('answer') or '').strip().lower()
    if r['options'] and len(key) == 1 and key not in {str(lab).strip().lower() for lab, _ in r['options']}:
        return 'key not among the options'   # options lost in scraping
    return None


def category(r) -> str:
    q = r['question']
    if re.search(r'\bmeans?\b|\bimplies\b|\bdenotes\b', q, re.I) and EXPR_SYM.search(q):
        return 'coded'
    if re.search(r'statement\s*(?:i|1)\b|\bstatements?\s*:', q, re.I):
        return 'sufficiency'
    if re.search(r'\b(?:members?|persons|people)\b.{0,60}\b(?:family|generations?)\b|\bmarried couples?\b|\bgenerations?\b', q, re.I):
        return 'puzzle'
    if re.search(r'\bpointing\b|\bintroduc|\bsaid\b|\bsays\b|\btold\b|\bpicture\b|\bphotograph\b', q, re.I):
        return 'speech'
    return 'statements' if len(q) < 400 else 'long text'


def report(results=None):
    rows = load_real()
    miss, oks = set(), None
    if results:
        res = json.load(open(results))
        oks = res['score'].get('ok')   # per question, in load_real order (exact)
        miss = {m['q'] for m in res['misses']}
    by = collections.defaultdict(lambda: [0, 0])
    flags = collections.Counter()
    for i, r in enumerate(rows):
        b = broken(r)
        flags[b or 'fair'] += 1
        ok = oks[i] if oks else r['question'][:300] not in miss   # older files: matched by text (approximate)
        for k in ('all', 'fair' if not b else 'broken', f'cat:{category(r)}' if not b else None):
            if k:
                by[k][0] += ok
                by[k][1] += 1
    return rows, flags, {k: f'{a}/{n} = {a / n:.1%}' for k, (a, n) in sorted(by.items())}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=None, help='run_real --out file (misses) to score')
    ap.add_argument('--show', default=None, help="print flagged items of this reason (or 'all')")
    a = ap.parse_args()
    rows, flags, scores = report(a.results)
    print(json.dumps({'flags': dict(flags), 'accuracy': scores}, indent=1))
    if a.show:
        for r in rows:
            b = broken(r)
            if b and (a.show == 'all' or a.show == b):
                print(f'[{b}] {re.sub(chr(10), " ", as_text(r))[:260]}')
