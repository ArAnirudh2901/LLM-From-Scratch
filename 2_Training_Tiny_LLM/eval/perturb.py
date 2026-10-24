"""Altered copies of real exam questions with the same answer: a check that a score is not tied to the surface the
pipeline was tuned on (evaluation only, never training text). Each copy
  - shuffles the options (the key moves with its option; 'none of these', 'cannot be determined', 'both' stay put),
  - renames people consistently: letters to letters unused in the question, first names to same-gender names that no
    generator list has (eval.unseen_set.held_out_names); names whose gender is open and A/I/O/U/V/X keep their spelling,
  - swaps code symbols consistently ("A $ B means ..." -> "A # B means ..."),
  - sometimes adds a page prefix ("Question 12:") or drops the closing '?'.

    python -m eval.perturb --split dev --k 3        # exact readers only (no model): original vs altered
    python -m eval.run_real --ckpt ... --split dev --perturb 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

KEEP_LETTERS = set('AIOUVX')   # articles, the pronoun I, Roman numerals, the x code symbol
NEW_LETTERS = [c for c in 'BCDEFGHJKLMNPQRSTWYZ']
SYMBOLS = ['$', '#', '%', '&', '*', '+', '÷', '×', '=', '!', '^', '<', '>', '©', '£', '★']
VERB = re.compile(r'\b(?:means|implies|denotes|represents|indicates|signifies|stands for|is coded as|shows)\b', re.I)
FIXED = re.compile(r'none of|cannot|can\'t|not (?:be )?(?:possible|determined)|data (?:is )?(?:in)?adequate|all of|both|neither|either', re.I)
NAMES_CACHE = 'data/real_questions/perturb_names.json'


COMMON = {   # everyday names as exam pages use them; those in a generator list or of unclear gender drop out below
    'M': 'Aarav Vihaan Advait Kabir Ishaan Reyansh Ayaan Dhruv Siddharth Pranav Nikhil Varun Kunal Tarun Gaurav Yash Manish '
         'Abhishek Vivek Ankit Sachin Deepak Sunil Anil Mahesh Suresh Naresh Dinesh Mukesh Rakesh Lokesh Hitesh Jatin Lalit '
         'Mohit Sumit Pankaj Neeraj Sanjay Ajay Akash Prakash Kailash Girish Satish Harish Jagdish Bharat Chetan Darshan Gautam '
         'Hemant Jayant Keshav Madhav Nakul Omkar Parth Raghav Sameer Tushar Uday Vinay Yogesh Irfan Imran Farhan Salman Arif '
         'Asif Joseph Thomas Mathew Abraham Gurpreet Venkatesh Srinivas Murali Karthik Senthil Balaji Prabhu Ganesh Shankar',
    'F': 'Aanya Diya Saanvi Ananya Ishita Kavya Meera Nisha Pooja Priya Riya Sneha Tanvi Asha Usha Lata Geeta Seema Reena '
         'Meena Neha Anjali Bhavna Charu Deepa Divya Esha Gauri Hema Isha Jaya Kirti Lavanya Madhuri Nandini Pallavi Rashmi '
         'Shalini Sunita Swati Trisha Uma Vandana Yamini Zoya Ayesha Fatima Sana Mary Elizabeth Lakshmi Padma Revathi Kamala '
         'Sujatha Bhavani Radhika Shruti Aishwarya Komal Preeti Sonal Rupa Mamta'}


def held_out() -> dict:
    """{'M': [...], 'F': [...]}: everyday names plus clear-gender names outside every generator list (cached); only
    names whose gender our name lookup is sure of, so a renamed question keeps its answer."""
    if os.path.exists(NAMES_CACHE):
        return json.load(open(NAMES_CACHE))
    from eval.unseen_set import held_out_names
    from kinlogic.names import INDIAN_F, INDIAN_M, WORLD_F, WORLD_M, name_gender
    used = {n.lower() for n in INDIAN_M + INDIAN_F + WORLD_M + WORLD_F}
    odd = re.compile(r'aa|ee|oo|mh|md|mj|jj|ndr$|dr$|[^aeiouyhnlrsmtdk]$')   # transliteration noise ('Jithemdr')
    n = held_out_names(per=400, seed=7)
    out = {}
    for g, k in (('M', 0), ('F', 1)):
        cand = [x for x in COMMON[g].split() if x.lower() not in used]
        cand += [x for x in n['indian'][k] + n['world'][k] if not odd.search(x.lower())]
        out[g] = [x for x, got in zip(cand, name_gender(cand)) if got == g]
    json.dump(out, open(NAMES_CACHE, 'w'))
    return out


def _sub(text: str, mapping: dict, word: bool) -> str:
    """All keys at once (no chained renames); word=True: whole words only ('Ravi', "Ravi's", not 'Ravindra')."""
    if not text or not mapping:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pat = '|'.join(re.escape(k) for k in keys)
    pat = rf'(?<![A-Za-z])(?:{pat})(?![A-Za-z])' if word else pat
    return re.sub(pat, lambda m: mapping[m.group(0)], text)


def _texts(r) -> list[str]:
    return [r['question']] + [t for _, t in r['options']] + [r.get('answer_text') or '']


def variant(r: dict, rng: random.Random, names: dict) -> dict:
    from kinlogic.names import name_gender
    from kinlogic.prepare import prepare
    from eval.run_real import as_text
    v = json.loads(json.dumps(r))
    full = ' '.join(_texts(r))
    ents = prepare(as_text(r))['ents']
    present = set(re.findall(r'(?<![A-Za-z])[A-Z](?![A-Za-z])', full))
    letters = sorted(c for c in present if c not in KEEP_LETTERS)
    if re.search(r'[A-Z][xX×][A-Z]|[A-Z][^\sA-Za-z][A-Z]|(?<![A-Za-z])[A-Z]{2}(?![A-Za-z])', full):
        letters = []   # glued letters ('PxB', '+R×Q', 'PQ'): renaming could miss some, so keep them all
    free = [c for c in NEW_LETTERS if c not in present]
    rng.shuffle(free)
    mapping = dict(zip(letters, free))   # letters beyond the free ones keep their spelling
    people = sorted({e['text'] for e in ents if e['kind'] == 'name' and e['text'][:1].isupper()})
    used = {w.lower() for w in re.findall(r'[A-Za-z]+', full)}
    for n, g in zip(people, name_gender(people) if people else []):
        pool = [x for x in names.get(g or '', []) if x.lower() not in used]
        if pool:
            new = rng.choice(pool)
            used.add(new.lower())
            mapping[n] = new
    ops = {}
    if VERB.search(full):
        cand = set(re.findall(r'(?<![A-Za-z])[A-Z]\s*([^\w\s.,;:\'"()\[\]?/\-@‘’“”`–—])\s*[A-Z](?![A-Za-z])', full))
        if re.search(r'\b[A-Z] [xX] [A-Z]\b', full):
            cand.discard('×')   # the letter x stands for × here: both must change together, so neither does
        spare = [s for s in SYMBOLS if s not in full and not (s == '×' and re.search(r'\b[A-Z] [xX] [A-Z]\b', full))]   # letter x already means ×
        rng.shuffle(spare)
        ops = dict(zip(sorted(cand), spare))

    def alter(t):
        return _sub(_sub(t, mapping, True), ops, False)

    v['question'] = alter(r['question'])
    if r.get('answer_text'):
        v['answer_text'] = alter(r['answer_text'])
    opts = [[l, alter(t)] for l, t in r['options']]
    if opts and r.get('answer') in [l for l, _ in opts] and not any(re.search(r'\([a-e]\)|\b[a-e] and [a-e]\b', t) for _, t in opts):
        move = [i for i, (_, t) in enumerate(opts) if not FIXED.search(t)]
        texts = [opts[i][1] for i in move]
        key = next(t for l, t in opts if l == r['answer'])
        rng.shuffle(texts)
        for i, t in zip(move, texts):
            opts[i][1] = t
        v['answer'] = next(l for l, t in opts if t == key)
    v['options'] = opts
    if rng.random() < 0.3:
        v['question'] = rng.choice(['Question {}: ', 'Q{}. ', '{}. ']).format(rng.randint(1, 60)) + v['question']
    if v['question'].rstrip().endswith('?') and rng.random() < 0.2:
        v['question'] = v['question'].rstrip()[:-1].rstrip()
    v['variant_of'] = r['question']
    return v


def variants(rows: list[dict], k: int, seed: int = 0) -> list[tuple[int, dict]]:
    """(index of the original, altered copy) for k copies of every row."""
    names = held_out()
    out = []
    for i, r in enumerate(rows):
        rng = random.Random(f'{seed}:{r["question"]}')
        out += [(i, variant(r, rng, names)) for _ in range(k)]
    return out


def compare(rows, oks_orig, pairs, oks_var, split_of, broken) -> dict:
    """Accuracy of originals vs altered copies, and how many originals stay right in every copy (fair items)."""
    import collections
    per = collections.defaultdict(list)
    for (i, _), ok in zip(pairs, oks_var):
        per[i].append(ok)
    rep = {}
    for s in sorted({split_of(r) for r in rows}):
        idx = [i for i, r in enumerate(rows) if split_of(r) == s and not broken(r)]
        if not idx:
            continue
        n = len(idx)
        o = sum(oks_orig[i] for i in idx)
        vv = sum(sum(per[i]) for i in idx) / max(1, sum(len(per[i]) for i in idx))
        robust = sum(oks_orig[i] and all(per[i]) for i in idx)
        flips = sum(oks_orig[i] and not all(per[i]) for i in idx)
        rep[f'{s} fair'] = {'n': n, 'original': f'{o / n:.1%}', 'altered (mean)': f'{vv:.1%}', 'right in all copies': f'{robust / n:.1%}',
                            'right, then wrong after altering': flips}
    return rep


def main():
    from eval.real_clean import broken
    from eval.run_real import load_real, run, split_of
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--show', type=int, default=0, help='print this many altered dev questions that flipped (dev only)')
    a = ap.parse_args()
    rows = [r for r in load_real() if split_of(r) == a.split]
    none = lambda b: [''] * len(b)   # exact readers only
    s0, _ = run(none, rows)
    pairs = variants(rows, a.k, a.seed)
    s1, _ = run(none, [v for _, v in pairs])
    print(json.dumps(compare(rows, s0['ok'], pairs, s1['ok'], split_of, broken), indent=1))
    if a.show and a.split == 'dev':
        from eval.run_real import as_text
        shown = 0
        for (i, v), ok in zip(pairs, s1['ok']):
            if s0['ok'][i] and not ok and shown < a.show:
                shown += 1
                print('\nORIGINAL:', as_text(rows[i])[:400], '| key', rows[i].get('answer'), '\nALTERED: ', as_text(v)[:400], '| key', v.get('answer'))


if __name__ == '__main__':
    main()
