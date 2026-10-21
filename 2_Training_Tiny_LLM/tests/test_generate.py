"""Generated questions must be true to their family: their facts never contradict, and the engine's answers
include the family's real answer (unless the text leaves it open)."""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kinlogic import generate as gen  # noqa: E402
from kinlogic.engine import NEUTRAL, solve  # noqa: E402
from kinlogic.facts import parse  # noqa: E402

KINDS = [k for k in dir(gen) if k.startswith('q_')]
OPEN = {'child', 'sibling', 'grandchild', 'parent', 'spouse', 'child-in-law', 'uncle or aunt', 'nephew or niece',
        'grandparent', 'sibling-in-law', 'parent-in-law', 'great-grandchild', 'great-grandparent', 'co-sibling',
        'grand-nephew or grand-niece', 'great-uncle or great-aunt'}


NEUTRAL_WORDS = [('grandfather', 'grandparent'), ('grandmother', 'grandparent'), ('granddaughter', 'grandchild'),
                 ('grandson', 'grandchild'), ('father', 'parent'), ('mother', 'parent'), ('daughter', 'child'),
                 ('son', 'child'), ('brother', 'sibling'), ('sister', 'sibling'), ('husband', 'spouse'), ('wife', 'spouse'),
                 ('uncle', 'uncle or aunt'), ('aunt', 'uncle or aunt'), ('nephew', 'nephew or niece'), ('niece', 'nephew or niece')]


def neutral(word):
    import re
    table = dict(NEUTRAL_WORDS)
    return re.sub(r'[a-z]+', lambda m: table.get(m.group(0), m.group(0)), word)


def agrees(truth, v):
    """The family's answer is among the engine's, possibly with the gender or side the text leaves open."""
    strip = lambda w: w.split(' ', 1)[1] if w.startswith(('paternal ', 'maternal ')) else w
    tb = strip(truth)
    for a in v['answers']:
        b = a.get('base') or ''
        import re
        open_ = re.search(r'(child|parent|sibling|spouse)\b| or ', b)
        if truth == a['key'] or tb == b or (open_ and (neutral(tb) == neutral(b) or tb in b.split(' or '))):
            return True
    return v['status'] in ('gender', 'side')


@pytest.mark.parametrize('kind', KINDS)
def test_generated_questions_are_true(kind):
    """Facts never contradict; answers miss the family's only when the text is too ambiguous (at most 3%)."""
    rng = random.Random(11)
    made, misses = 0, []
    for _ in range(400):
        ex = getattr(gen, kind)(rng)
        if ex is None:
            continue
        made += 1
        stmts, bad = parse(ex['lf'])
        assert not bad, (ex['text'], ex['lf'], bad)
        r = solve(stmts, people=[e['id'] for e in ex['ents']])
        assert r['ok'] or not r['complete'], ('contradiction', ex['text'], ex['lf'])
        t = ex.get('truth')
        for q in r['questions']:
            keys = [a['key'] for a in q['answers']]
            if q['q']['t'] == 'ask' and isinstance(t, str):
                ok = agrees(t, q)
            elif q['q']['t'] == 'who' and isinstance(t, dict):
                want = sorted(k for k, v in ex['names'].items() if v in t['who'])
                ok = not want or any(a.get('names') == want for a in q['answers'])
            elif q['q']['t'] == 'count' and isinstance(t, dict):
                ok = str(t['count']) in keys
            elif q['q']['t'] == 'gender?' and isinstance(t, dict):
                ok = t['gender'] in keys or 'unknown' in keys
            else:
                continue
            if not ok:
                misses.append((ex['text'][:200], ex['lf'], t, keys[:3]))
    assert made > 100
    assert len(misses) <= 0.03 * made, misses[:5]
