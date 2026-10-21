"""Checks every authored question against the engine with its gold facts: the facts must parse, the gold facts'
names must all be found by prepare(), and the engine's option must be the expected one.

    python -m eval.check_authored
"""
import collections
import re

from eval.authored_set import A, split
from eval.exam_set import gold_facts
from kinlogic.exam import answer
from kinlogic.facts import parse
from kinlogic.prepare import prepare


def option_reads(it, p):
    """Gold facts for each option of a "which expression" item, placed the way the pipeline phrases each option."""
    stmts, _ = parse(gold_facts(it, p['ents']))
    c = next(s for s in stmts if s.get('claim') and s['t'] == 'rel')
    names = {e['id']: e['text'] for e in p['ents']}
    cut = re.search(r'which (?:of the following|one)', p['question'], re.I)
    defs = p['question'][:cut.start()] if cut else p['question']
    out = []
    for lab, o in p['options']:
        pp = prepare(f"{defs} In the expression '{o}', how is {names.get(c['y'], c['y'])} related to {names.get(c['x'], c['x'])}?")
        out.append(gold_facts({'facts': it['reads'][lab]}, pp['ents']))
    return out


def check(items=A, verbose=True):
    bad, by = [], collections.Counter()
    for it in items:
        p = prepare(it['text'])
        gold = gold_facts(it, p['ents'])
        _, errs = parse(gold)
        leftovers = [t for t in gold.split() if t[0].isupper() or t.startswith('~')]
        reads = option_reads(it, p) if it.get('reads') else None
        r = answer(it['text'], lambda batch, g=gold: reads if reads and len(batch) == len(reads) and len(batch) > 1 else [g for _ in batch])
        ok = not errs and not leftovers and r['option'] == it['expect']
        by[(split(it), ok)] += 1
        if not ok:
            bad.append(it['id'])
            if verbose:
                print(f"BAD {it['id']}: got {r['option']} ({r.get('answer')}) want {it['expect']} | parse errors {errs} "
                      f"| unplaced {leftovers} | {gold}")
    n = len(items)
    print(f"{n - len(bad)}/{n} verified; splits: " + ', '.join(f'{s} {sum(v for (sp, _), v in by.items() if sp == s)}' for s in ('train', 'dev', 'test')))
    return bad


if __name__ == '__main__':
    check()
