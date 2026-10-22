import gzip
import json
import random

from kinlogic import deep, reword_local as RL
from kinlogic.prepare import prepare


def _facts_json(ex):
    """The extraction a perfect reader would give for ex's own text (names as written)."""
    p = prepare(ex['text'])
    name = {e['id']: e['text'] for e in p['ents']}
    facts, genders, ask = [], [], {}
    for part in ex['lf'].split(' ; '):
        t = part.split()
        if t[0] == 'ask':
            ask = {'how_is': name[t[2]], 'related_to': name[t[1]]}
        elif len(t) == 2:
            genders.append({'person': name[t[0]], 'gender': t[1]})
        else:
            facts.append({'person': name[t[2]], 'is_the': t[1], 'of': name[t[0]]})
    return json.dumps({'facts': facts, 'genders': genders, 'ask': ask})


class Fake:
    """Rewrite = the question itself; reading = its true facts."""
    kind = 'fake'

    def __init__(self, model):
        self.by_text = {}

    def chat(self, prompts, temperature, max_tokens):
        out = []
        for p in prompts:
            q = p.split('Question: ', 1)[1]
            out.append(q if temperature else self.by_text[q])
        return out


def test_pipeline_keeps_faithful_rewordings(tmp_path, monkeypatch):
    rng = random.Random(2)
    rows = [x for x in (deep.make(rng, 'statements', d) for d in (2, 3, 4) for _ in range(4)) if x]
    for r in rows:
        r['names'] = {e['id']: e['text'] for e in prepare(r['text'])['ents']}
    fake = Fake('x')
    fake.by_text = {r['text']: _facts_json(r) for r in rows}
    monkeypatch.setattr(RL, 'Backend', lambda model: fake)
    monkeypatch.setattr(RL, 'sample_sources', lambda src, n, seed=0: rows)
    out = tmp_path / 'out.jsonl.gz'
    monkeypatch.setattr('sys.argv', ['x', '--out', str(out), '--n', '100', '--chunk', '5', '--workers', '1'])
    RL.main()
    kept = [json.loads(l) for l in gzip.open(out, 'rt')]
    assert len(kept) >= len(rows) - 1 and all(k['kind'].startswith('reworded-') for k in kept)
    assert RL.done_sources(str(out)) == {k['source'] for k in kept}
