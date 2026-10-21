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
from model_gpt import GPT
from reader import Vocab, decode
from train import get_device

RIDDLE_URL = 'https://huggingface.co/datasets/ai4bharat/RiddleBench/resolve/main/data/train-00000-of-00001.parquet'


def load(ckpt, device):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = GPT(**ck['config']).to(device)
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
    from eval.exam_set import Q
    right, misses = 0, []
    for item in Q:
        r = answer(item['text'], read)
        ok = r['option'] == item['expect']
        right += ok
        if not ok:
            misses.append((item['id'], r['option'], item['expect'], r['facts']))
    return {'exam': f'{right}/{len(Q)}', 'exam_acc': right / len(Q), 'exam_misses': misses[:15]}


def eval_clutrr(read, path='data/clutrr_facts/test.jsonl.gz', n=None):
    rows = [json.loads(l) for l in gzip.open(path, 'rt')][:n]
    preds = read([r['tokens'] for r in rows])
    hit = ceil = 0
    by_k = {}
    for r, p in zip(rows, preds):
        got, v = first(p)
        gold, gv = first(r['lf'])
        keys = [r['answer'], (gv or {}).get('neutral')]
        ok = got in keys or (v and v.get('neutral') == r['answer'])
        hit += bool(ok)
        ceil += gold == r['answer'] or (gv and gv.get('neutral') == r['answer'])
        k = by_k.setdefault(r['k'], [0, 0])
        k[0] += bool(ok); k[1] += 1
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


def eval_generated(model, vocab, device, n=1000, seed=4242):
    from train_reader import dev_set, evaluate
    r, _ = evaluate(model, vocab, device, dev_set(n, seed))
    return {'generated_exact': r['exact'], 'generated_answer': r['answer'], 'generated_kinds': r['kinds']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/reader-v9/best.pt')
    ap.add_argument('--sets', default='exam,clutrr,riddle,generated')
    ap.add_argument('--out', default=None, help='write the results as JSON here')
    a = ap.parse_args()
    device = get_device()
    model, vocab = load(a.ckpt, device)
    read = lambda batch: decode(model, vocab, batch, device)
    res = {}
    for s in a.sets.split(','):
        res.update({'exam': lambda: eval_exam(read), 'clutrr': lambda: eval_clutrr(read), 'riddle': lambda: eval_riddle(read),
                    'generated': lambda: eval_generated(model, vocab, device)}[s]())
        print(json.dumps({k: v for k, v in res.items() if not k.endswith('misses')}, indent=1), flush=True)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == '__main__':
    main()
