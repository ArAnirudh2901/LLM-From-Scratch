"""Evaluates a reader checkpoint end to end (reader -> engine -> option):

    python -m eval.evaluate --ckpt runs/reader-v9/best.pt --sets exam,clutrr,riddle,generated

exam      hand-written exam questions (eval/exam_set.py): right option
clutrr    CLUTRR test stories (2-10 steps): engine's first answer vs the label, and vs the label-error-free ceiling
riddle    RiddleBench blood relations (CC0, hard puzzles; downloaded): right option
generated fresh generated questions by kind: exact facts and same answer
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import urllib.request

import torch

from kinlogic.engine import solve
from kinlogic.exam import answer
from kinlogic.facts import parse
from model_gpt import GPT  # noqa: F401
from model_reader import make_model
from reader import Vocab, decode, generate
from train import get_device

RIDDLE_URL = 'https://huggingface.co/datasets/ai4bharat/RiddleBench/resolve/main/data/train-00000-of-00001.parquet'


def load(ckpt, device):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = make_model(ck['config']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, Vocab(ck['vocab'])


def first(lf, people=()):
    st, _ = parse(lf)
    q = solve(st, people=people, max_steps=5000)['questions']
    if not q or not q[-1]['answers']:
        return None, None
    a = q[-1]['answers'][0]
    return a.get('base') or a['key'], q[-1]


def eval_exam(read):
    from eval.exam_set import Q, split
    score = {'dev': [0, 0], 'test': [0, 0]}
    misses = []
    for item in Q:
        r = answer(item['text'], read)
        ok = r['option'] == item['expect']
        s = score[split(item)]
        s[0] += ok; s[1] += 1
        if not ok:
            misses.append((item['id'], split(item), r['option'], item['expect'], r['facts']))
    right = score['dev'][0] + score['test'][0]
    return {'exam': f'{right}/{len(Q)}', 'exam_acc': right / len(Q), 'exam_dev_acc': score['dev'][0] / score['dev'][1],
            'exam_test_acc': score['test'][0] / score['test'][1], 'exam_misses': misses[:20]}


def eval_authored(read):
    """Hand-written questions (eval/authored_set.py): right option on the dev and the locked test split."""
    from eval.authored_set import A, split
    from eval.check_authored import option_reads   # noqa: F401  (expression items read their options with the model)
    score = {'dev': [0, 0], 'test': [0, 0]}
    for it in A:
        sp = split(it)
        if sp not in score:
            continue
        ok = answer(it['text'], read)['option'] == it['expect']
        score[sp][0] += ok
        score[sp][1] += 1
    return {f'authored_{k}_acc': a / max(1, b) for k, (a, b) in score.items()} | {'authored_n': {k: b for k, (_, b) in score.items()}}


def eval_items(read, items, key):
    """Accuracy and per-kind counts on a gold-verified item list (fresh_set, gender_set)."""
    by = {}
    for it in items:
        ok = answer(it['text'], read)['option'] == it['expect']
        a = by.setdefault(it['kind'], [0, 0])
        a[0] += ok
        a[1] += 1
    n = sum(a for a, _ in by.values())
    return {f'{key}_acc': round(n / len(items), 4), f'{key}_by_kind': {k: f'{a}/{b}' for k, (a, b) in sorted(by.items())}}


def eval_clutrr(read, path='data/clutrr_facts/test.jsonl.gz', n=None):
    rows = [json.loads(l) for l in gzip.open(path, 'rt')][:n]
    preds = read([r['tokens'] for r in rows])
    hit = ceil = 0
    by_k = {}
    def right(lf):   # gender-open answers count when their neutral word is the label; no answer never counts
        got, v = first(lf)
        return got == r['answer'] or bool(v and v.get('neutral') == r['answer'])

    for r, p in zip(rows, preds):
        ok = right(p)
        hit += ok
        ceil += right(r['lf'])
        k = by_k.setdefault(r['k'], [0, 0])
        k[0] += ok; k[1] += 1
    return {'clutrr_acc': hit / len(rows), 'clutrr_ceiling': ceil / len(rows),
            'clutrr_by_steps': {k: round(a / b, 3) for k, (a, b) in sorted(by_k.items())}}


def eval_riddle(read, cache='data/riddlebench.parquet'):
    import pandas as pd
    if not os.path.exists(cache):
        urllib.request.urlretrieve(RIDDLE_URL, cache)
    df = pd.read_parquet(cache)
    df = df[df.type == 'blood relations']
    right = answered = 0
    for row in df.itertuples():
        try:
            r = answer(row.question, read)
        except Exception:
            continue
        answered += r['option'] is not None
        right += (r['option'] or '').lower() == row.answer.strip().lower()
    return {'riddle_acc': right / len(df), 'riddle_answered': answered / len(df), 'riddle_n': len(df)}


def eval_deep(model, vocab, device, root='data/dataset'):
    """The locked half of the deep-chain test set (build_dataset.py): answer accuracy overall, by depth and by format."""
    from train_reader import evaluate, load_pool
    rows = load_pool(root)['test'] if os.path.exists(f'{root}/test/deep.jsonl.gz') else []
    if not rows:
        return {'deep_acc': None}
    from kinlogic.codes import read as read_code
    from train_reader import answers
    r, preds = evaluate(model, vocab, device, rows)
    ok, hits = {}, 0
    for x, p in zip(rows, preds):   # the pipeline: coded questions are read exactly, the rest by the model
        people = sorted({t for t in x['tokens'] if t[0] == '@'})
        hit = answers(read_code(x['tokens']) or p, people) == answers(x['lf'], people)
        hits += hit
        for k in (('depth', x['depth']), ('format', x['kind'])):
            a = ok.setdefault(k, [0, 0])
            a[0] += hit
            a[1] += 1
    return {'deep_acc': round(hits / len(rows), 4), 'deep_model_acc': r['answer'], 'deep_n': len(rows),
            'deep_by_depth': {d: round(ok[('depth', d)][0] / ok[('depth', d)][1], 3) for d in sorted(d for t, d in ok if t == 'depth')},
            'deep_by_format': {f: round(a / b, 3) for (t, f), (a, b) in ok.items() if t == 'format'}}


def eval_unseen(model, vocab, device, path='data/unseen_test.jsonl.gz'):
    """Frozen test no run trains on or is picked with (eval/unseen_set.py): new seed, names outside every generator
    list, wording not in any training file. Pipeline accuracy: coded questions read exactly, the rest by the model."""
    from kinlogic.codes import read as read_code
    from train_reader import answers, evaluate, load_jsonl
    if not os.path.exists(path):
        return {'unseen_acc': None}
    rows = load_jsonl(path)
    r, preds = evaluate(model, vocab, device, rows)
    by = {}
    for x, p in zip(rows, preds):
        people = sorted({t for t in x['tokens'] if t[0] == '@'})
        a = by.setdefault(x['kind'], [0, 0])
        a[0] += answers(read_code(x['tokens']) or p, people) == answers(x['lf'], people)
        a[1] += 1
    hits = sum(a for a, _ in by.values())
    return {'unseen_acc': round(hits / len(rows), 4), 'unseen_model_acc': r['answer'], 'unseen_n': len(rows),
            'unseen_by_kind': {k: f'{a}/{b}' for k, (a, b) in sorted(by.items())}}


def eval_reworded(model, vocab, device, path='data/reworded_dev.jsonl.gz'):
    """Held-out verified rewrites (kinlogic/reword_merge.py): wording no generator template wrote."""
    from train_reader import evaluate, load_jsonl
    if not os.path.exists(path):
        return {'reworded_answer': None}
    r, _ = evaluate(model, vocab, device, load_jsonl(path))
    return {'reworded_exact': r['exact'], 'reworded_answer': r['answer'], 'reworded_kinds': r['kinds']}


def eval_generated(model, vocab, device, n=1000, seed=4242):
    from train_reader import dev_set, evaluate
    r, _ = evaluate(model, vocab, device, dev_set(n, seed))
    return {'generated_exact': r['exact'], 'generated_answer': r['answer'], 'generated_kinds': r['kinds']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/reader-v9/best.pt')
    ap.add_argument('--sets', default='exam,authored,fresh,gender,deep,unseen,reworded,clutrr,riddle,generated')
    ap.add_argument('--out', default=None, help='write the results as JSON here')
    ap.add_argument('--beam', type=int, default=8, help='readings tried per exam question (best first); 8 beat 4 by 4 exam questions, 16 adds little')
    a = ap.parse_args()
    device = get_device()
    model, vocab = load(a.ckpt, device)
    read = lambda batch: decode(model, vocab, batch, device)
    beam = lambda batch: generate(model, vocab, batch, device, k=a.beam)   # several readings, the engine picks
    res = {}
    for s in a.sets.split(','):
        from eval.fresh_set import F as FRESH
        from eval.gender_set import G as GENDER
        res.update({'exam': lambda: eval_exam(beam), 'authored': lambda: eval_authored(beam),
                    'fresh': lambda: eval_items(beam, FRESH, 'fresh'), 'gender': lambda: eval_items(beam, GENDER, 'gender'), 'clutrr': lambda: eval_clutrr(read), 'riddle': lambda: eval_riddle(beam),
                    'generated': lambda: eval_generated(model, vocab, device), 'deep': lambda: eval_deep(model, vocab, device),
                    'reworded': lambda: eval_reworded(model, vocab, device),
                    'unseen': lambda: eval_unseen(model, vocab, device)}[s]())
        print(json.dumps({k: v for k, v in res.items() if not k.endswith('misses')}, indent=1), flush=True)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == '__main__':
    main()
