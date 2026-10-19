from __future__ import annotations
import argparse, random, re
from collections import Counter

import torch

from ask_qa import bracket
from model_gpt import GPT
from name_tagger import NON_PERSON_TEMPLATES, REAL_PLACES, TagVocab, bracket_names, find_names, invented_name
from qa_data import (EVAL_TEMPLATES, UNKNOWN, Vocab, add_distractors, beam_predict, linked_in_text, load_splits,
                     people, prefix_fits, unanswerable)
from train import get_device

# End-to-end check of ask_qa's behaviour on held-out stories as plain text: every name replaced by an invented
# one (never in training), names found by the tagger, then the QA model. Outcomes per question type:
# right / wrong / unknown ("the story doesn't say") / refused (a query name is not a person) / not sure.


ORGS = {'Google', 'Microsoft', 'Walmart', 'Starbucks', 'Harvard', 'Stanford', 'Toyota', 'Nike', 'Netflix', 'Boeing',
        'Oxford', 'Yale'}


def to_plain(ex: dict, rng) -> tuple[str, str, str]:
    """Story without brackets, every person renamed to a new invented name; returns story, a, b."""
    new = {}
    for n in dict.fromkeys(people(ex['story']) + [ex['a'], ex['b']]):
        m = invented_name(rng)
        while m in new.values() or len(m) < 3:
            m = invented_name(rng)
        new[n] = m
    return re.sub(r'\[([^\]]+)\]', lambda g: new[g.group(1)], ex['story']), new[ex['a']], new[ex['b']]


def with_place(ex: dict, rng) -> tuple[str, str, str]:
    """Plain story plus a real place/company (never in tagger training); the question asks about the place."""
    story, a, _ = to_plain(ex, rng)
    place = rng.choice(REAL_PLACES)
    if rng.random() < 0.5:
        story += ' ' + rng.choice(NON_PERSON_TEMPLATES).format(X=a, P=place)
    else:
        sents = re.split(r'(?<=[.!?])\s+', story)
        j = rng.randrange(len(sents))
        end = sents[j][-1] if sents[j][-1:] in '.!?' else ''
        prep = rng.choice(['at', 'from'] if place in ORGS else ['in', 'from', 'near'])
        sents[j] = (sents[j][:-1] if end else sents[j]) + f" {prep} {place}{end or '.'}"
        story = ' '.join(sents)
    return story, a, place


def build(split: list[dict], n: int, seed: int) -> dict:
    rng = random.Random(seed)
    clean = [e for e in split if not e.get('seen')]
    def make(f):
        out = []
        while len(out) < n:
            e, o = rng.choice(clean), rng.choice(clean)
            x = f(e, o) if o is not e else None
            if x is not None:
                out.append(x)
        return out
    return {
        'answerable': make(lambda e, o: (*to_plain(e, rng), e['rel'])),
        'answerable + distractors': make(lambda e, o: (lambda d: d and (*to_plain(d, rng), d['rel']))(
            add_distractors(e, rng, EVAL_TEMPLATES, o))),
        'unknown: friend/stranger': make(lambda e, o: (lambda d: d and (*to_plain(d, rng), UNKNOWN))(
            unanswerable(e, rng, EVAL_TEMPLATES, kind='friend'))),
        'unknown: two families': make(lambda e, o: (lambda d: d and (*to_plain(d, rng), UNKNOWN))(
            unanswerable(e, rng, EVAL_TEMPLATES, o, kind='two'))),
        'place/company question': make(lambda e, o: (*with_place(e, rng), 'refuse')),
        'person not in story': make(lambda e, o: (lambda d: d and (*to_plain(d, rng), 'refuse'))(
            unanswerable(e, rng, EVAL_TEMPLATES, kind='absent'))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='runs/clutrr-qa-v8/best.pt')
    p.add_argument('--tagger', default='runs/name-tagger-v4/tagger.pt')
    p.add_argument('--n', type=int, default=200)          # questions per type and split
    p.add_argument('--views', type=int, default=4)
    p.add_argument('--unk_threshold', type=float, default=0.5)
    p.add_argument('--min_conf', type=float, default=0.7)
    p.add_argument('--splits', nargs='*', default=['val', 'test'])
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = get_device(args.cpu)
    ck = torch.load(args.ckpt, map_location=device)
    model = GPT(**ck['config']).to(device).eval(); model.load_state_dict(ck['model'])
    vocab = Vocab(ck['vocab'])
    tg = torch.load(args.tagger, map_location=device)
    tagger = GPT(**tg['config']).to(device).eval(); tagger.load_state_dict(tg['model'])
    tvocab = TagVocab(tg['vocab'])
    _, _, val, test = load_splits(other_tests=True)
    rows = {}

    for split, data, seed in [x for x in (('val', val, 21), ('test', test, 22)) if x[0] in args.splits]:
        for kind, items in build(data, args.n, seed).items():
            recs, todo = [], []   # (gold, prediction or 'refused', agreement, linked in text)
            for story, a, b, gold in items:
                found = find_names(tagger, tvocab, story, device)
                if a not in found or b not in found:
                    recs.append((gold, 'refused', 0.0, True))
                    continue
                ex = {'story': bracket_names(story, found), 'a': a, 'b': b, 'rel': gold}
                if not prefix_fits(vocab, ex, ck['config']['block_size']):
                    recs.append((gold, 'refused', 0.0, True))   # story too long for the model
                    continue
                todo.append(ex)
            for i in range(0, len(todo), 64):
                chunk = todo[i:i + 64]
                preds, _, info = beam_predict(model, vocab, chunk, device, views=args.views,
                                              block_size=ck['config']['block_size'], unk_threshold=args.unk_threshold)
                recs += [(e['rel'], pr, sh, linked_in_text(e['story'], e['a'], e['b']))
                         for e, pr, sh in zip(chunk, preds, info['share'].tolist())]
                if device.type == 'mps':
                    torch.mps.empty_cache()
            rows[(split, kind)] = recs
            for rule in (False, True):
                print(f"{split:4s} {kind:26s} {'+rule' if rule else 'model'} " + summary(recs, rule, args.min_conf), flush=True)
    torch.save(rows, f"runs/logs/eval_pipeline_rows_{'_'.join(args.splits)}.pt")


def summary(recs, rule: bool, min_conf: float) -> str:
    """Shares of: confident answer right / wrong, 'Not sure' with the best guess right / wrong, 'unknown', refused."""
    c = Counter()
    for gold, pr, sh, linked in recs:
        if pr == 'refused':
            c['refused'] += 1
        elif pr == UNKNOWN or (rule and not linked):
            c['unknown'] += 1
        else:
            c[('sure ' if sh >= min_conf else 'not sure, guess ') + ('right' if pr == gold else 'wrong')] += 1
    n = len(recs)
    return f'n={n} | ' + ' | '.join(f'{k} {c[k] / n:.3f}' for k in (
        'sure right', 'sure wrong', 'not sure, guess right', 'not sure, guess wrong', 'unknown', 'refused'))

if __name__ == '__main__':
    main()
