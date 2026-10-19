from __future__ import annotations
import json, random, re, sys

import torch

sys.path.insert(0, '.')
import fast_qa
from eval_pipeline import build
from model_gpt import GPT
from name_tagger import TagVocab, find_names
from qa_data import Vocab, load_splits, unknown_eval_sets

# Reference outputs from the Python models, for web/test/test_node.mjs to compare against the WASM engine.
dv = torch.device('cpu')
ck = torch.load('runs/clutrr-qa-v8/best.pt', map_location=dv)
m = GPT(**ck['config']).eval(); m.load_state_dict(ck['model']); v = Vocab(ck['vocab'])
tg = torch.load('runs/name-tagger-v4/tagger.pt', map_location=dv)
t = GPT(**tg['config']).eval(); t.load_state_dict(tg['model']); tv = TagVocab(tg['vocab'])
_, _, val, test = load_splits(other_tests=True)
rng = random.Random(0)
clean_val, clean_test = [e for e in val if not e['seen']], [e for e in test if not e['seen']]
q = lambda e: {'story': e['story'], 'a': e['a'], 'b': e['b'], 'rel': e['rel'], 'k': e['k']}

out = {}
# 1. raw forward: last-token logits of three story prefixes
seqs = [fast_qa.view_tokens(v, e, view) for e, view in ((clean_val[0], 0), (clean_val[1], 2), (clean_test[5], 3))]
with torch.no_grad():
    out['forward'] = [{'ids': s, 'logits': m(torch.tensor([s]))[0][0, -1].tolist()} for s in seqs]
# 2. single-view search on 60 questions
us = unknown_eval_sets(val, 20, seed=5)
exs = rng.sample(clean_val, 30) + us['unknown: two families'][:15] + us['unknown: friend/stranger'][:15]
eng = fast_qa.SlotEngine(m)
preds, chains, info = fast_qa.predict(eng, v, exs, views=1)
out['search'] = [{**q(e), 'pred': p, 'chain': c, 'share': s, 'p_start': ps, 'p_link': pl}
                 for e, p, c, s, ps, pl in zip(exs, preds, chains, info['share'].tolist(), info['p_start'].tolist(), info['p_link'].tolist())]
# 3. tagger on plain texts (incl. places and one long text)
items = [x for k, xs in build(val, 6, 31).items() for x in xs]
texts = [s for s, *_ in items] + [' '.join(f'Name{i} went to the park with his sister Kalo{i}.' for i in range(60))]
out['tagger'] = [{'text': s, 'names': find_names(t, tv, s, dv)} for s in texts]
# 4. labelled sets for the speed / accuracy sweep
sets = unknown_eval_sets(val, 150, seed=2)
out['sweep'] = {'val answerable': [q(e) for e in rng.sample(clean_val, 300)],
                'test answerable': [q(e) for e in rng.sample(clean_test, 200)],
                'val distractors': [q(e) for e in sets['answerable + distractors']],
                'friend/stranger': [q(e) for e in sets['unknown: friend/stranger']],
                'two families': [q(e) for e in sets['unknown: two families']]}
json.dump(out, open('web/test/golden.json', 'w'))
print('wrote web/test/golden.json', {k: len(x) for k, x in out.items()})
