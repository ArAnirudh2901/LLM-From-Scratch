from __future__ import annotations
import argparse, re

import torch

from model_gpt import GPT
from name_tagger import TagVocab, bracket_names, find_names
from qa_data import RELATIONS, UNKNOWN, Vocab, anonymize, beam_predict, linked_in_text, predict, prefix_fits
from train import get_device
from verify_model import check


def bracket(story: str, names: list[str]) -> str:
    """Wrap bare mentions of the given names in [brackets] so they become entity tokens. Any name works."""
    for n in names:
        story = re.sub(rf'(?<!\[)\b{re.escape(n)}\b(?!\])', f'[{n}]', story)
    return story

# Example (names need no brackets; the tagger finds them, including names never seen in training):
#   python ask_qa.py --story "Ashley's daughter, Lillian, asked her mom to read her a story. \
#       Nicholas's sister Lillian asked him for some help." --query Ashley Nicholas


def answer(model, vocab, cot: bool, block_size: int, story: str, people: list[str], a: str, b: str, device,
           beam: int = 8, views: int = 4, unk_threshold: float = 0.5, min_conf: float = 0.7,
           link_check: bool = True) -> tuple[str, list[str]]:
    """Answer sentence and chain steps (real names) for "how is b related to a?". Refuses when a or b is not a
    person in the story; says so when the story gives no link or the model's chains disagree."""
    lookup = {p.lower(): p for p in people}
    missing = [n for n in (a, b) if n.lower() not in lookup]
    if missing:
        return (f"Can't answer: {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not a person in the story. "
                "I only answer how two people in the story are related."), []
    a, b = lookup[a.lower()], lookup[b.lower()]
    if a == b:
        return f'{a} and {b} are the same person.', []
    if link_check and not linked_in_text(story, a, b):
        return f"The story doesn't say how {b} is related to {a} (they are never connected in the text).", []
    ex = {'story': story, 'a': a, 'b': b, 'rel': RELATIONS[0]}
    if not prefix_fits(vocab, ex, block_size):
        return f"Can't answer: the story is too long for this model (about {block_size - 40} words at most).", []
    if cot and beam:
        (rel,), (chain,), info = beam_predict(model, vocab, [ex], device, beam=beam, views=views,
                                              block_size=block_size, unk_threshold=unk_threshold)
        share, p_unk = info['share'].item(), info['p_unknown'].item()
    else:
        (rel,), (chain,) = predict(model, vocab, [ex], device, cot, block_size)
        share, p_unk = 1.0, 0.0
    if rel == UNKNOWN:
        return f"The story doesn't say how {b} is related to {a} (p_unknown {p_unk:.2f}).", []
    names = {tok: name for name, tok in anonymize(ex)[1].items()}
    steps = [' '.join(names.get(w, w) for w in s.split()) for s in filter(None, chain.split('<sep>'))]
    if share < min_conf:
        return f"Not sure. Best guess: {b} is {a}'s {rel} (agreement {share:.2f}, p_unknown {p_unk:.2f}).", steps
    return f"{b} is {a}'s {rel} (agreement {share:.2f}, p_unknown {p_unk:.2f}).", steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='runs/clutrr-qa-v8/best.pt')
    p.add_argument('--story', required=True, help='plain text; names in [brackets] are optional')
    p.add_argument('--names', nargs='*', default=[], help='extra names to treat as people')
    p.add_argument('--tagger', default='runs/name-tagger-v4/tagger.pt')
    p.add_argument('--query', nargs=2, required=True, metavar=('A', 'B'), help="answers: B is A's ___")
    p.add_argument('--beam', type=int, default=4)    # beam 4 + 1 view: same accuracy as 8 x 4 (web/test/calib.json)
    p.add_argument('--views', type=int, default=1)
    p.add_argument('--unk_threshold', type=float, default=0.5)   # say "doesn't say" when p_unknown >= this
    p.add_argument('--min_conf', type=float, default=0.7)        # below this vote agreement: "Not sure"
    p.add_argument('--no_link_check', action='store_true')   # skip the rule: people never connected in the text
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = get_device(args.cpu)
    for path in (args.ckpt, args.tagger):   # refuse damaged model files before loading them
        check(path)
    ck = torch.load(args.ckpt, map_location=device)
    model = GPT(**ck['config']).to(device).eval()
    model.load_state_dict(ck['model'])
    vocab = Vocab(ck['vocab'])

    story = args.story
    if '[' in story:
        people = re.findall(r'\[([^\]]+)\]', story)
    else:   # no brackets: the name tagger finds the people
        tg = torch.load(args.tagger, map_location=device)
        tagger = GPT(**tg['config']).to(device).eval()
        tagger.load_state_dict(tg['model'])
        people = find_names(tagger, TagVocab(tg['vocab']), story, device)
        print('names found:', ', '.join(people) or '(none)')
        story = bracket_names(story, people)
    story = bracket(story, args.names)
    people = list(dict.fromkeys(people + args.names))

    text, steps = answer(model, vocab, ck.get('cot', False), ck['config']['block_size'], story, people,
                         *args.query, device, args.beam, args.views, args.unk_threshold, args.min_conf,
                         not args.no_link_check)
    for s in steps:
        print('  ' + s)
    print(text)


if __name__ == '__main__':
    main()
