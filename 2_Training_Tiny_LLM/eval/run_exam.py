"""Scores a reader on the hand-written exam set: python -m eval.run_exam [checkpoint|gold]"""
import sys

from eval.exam_set import Q, gold_facts
from kinlogic.exam import answer
from kinlogic.prepare import prepare


def gold_reader(item):
    p = prepare(item['text'])
    gold = gold_facts(item, p['ents'])
    return lambda batch: [gold for _ in batch]


def run(make_reader, verbose=True):
    right, rows = 0, []
    for item in Q:
        r = answer(item['text'], make_reader(item))
        ok = r['option'] == item['expect']
        right += ok
        rows.append((item['id'], ok, r['option'], item['expect'], r['answer']))
        if verbose and not ok:
            print(f"MISS {item['id']}: got {r['option']} ({r['answer']}) want {item['expect']} | {r['facts']}")
    print(f'{right}/{len(Q)} correct')
    return rows


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == 'gold':
        run(gold_reader)
    else:
        import torch
        from model_gpt import GPT
        from reader import Vocab, decode
        from train import get_device
        dev = get_device()
        ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
        model = GPT(**ck['config']).to(dev)
        model.load_state_dict(ck['model'])
        vocab = Vocab(ck['vocab'])
        run(lambda item: (lambda batch: decode(model, vocab, batch, dev)))
