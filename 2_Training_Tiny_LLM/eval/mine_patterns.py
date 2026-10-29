"""Exam-pattern mining: what real blood-relation questions look like, counted, next to what the generator writes.

Runs where the private real questions are (Kaggle or the owner's Mac). It reads the dev split only (never 'test' or
the locked2 questions) and writes counts and masked question skeletons, never question text, so the report can be
committed and used to steer generation: pattern mining is allowed, real text never reaches training (eval/sites.py,
eval/leak.py). Skeletons replace people with @, relation words with R and numbers with #, and are kept only when at
least --min_skeleton dev questions share them.

    python -m eval.mine_patterns --out runs/patterns_dev.json
    python -m eval.mine_patterns --trace runs/v10-r8p-final/trace_dev.jsonl --out runs/patterns_dev.json
    python -m eval.mine_patterns --gen_files data/mix_r8.jsonl.gz data/balanced_mix_v1.jsonl.gz --out ...

For every feature (opening, question form, construction, kin word, coded symbols, length, people, options, answer)
the report gives its share of real dev questions, its share of the generated sample, their ratio, and with --trace
the model's accuracy on the dev questions that have it and how many of them it missed. 'gaps' lists features common
in real questions (>= 1%) that the generator writes at under half that rate; 'misses' ranks features by missed dev
questions. Generate the gaps first, weighted by real share, and check the result on dev, never on the locked test.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from kinlogic.generate import HINDI, WORDS
from kinlogic.prepare import prepare

KIN = sorted({w for forms in WORDS.values() for w in forms if w} | set(HINDI.values()) |
             {'co-brother', 'co-sister', 'spouse', 'sibling', 'parent', 'child', 'grandchild', 'in-law',
              'nephew', 'niece', 'uncle', 'aunt', 'cousin', 'father-in-law', 'mother-in-law', 'son-in-law',
              'daughter-in-law', 'brother-in-law', 'sister-in-law', 'mausa', 'chachi', 'mami', 'devar', 'jeth', 'nanad',
              'saala', 'saali', 'bhai', 'behen', 'beta', 'beti', 'pita', 'mata', 'husband', 'wife'}, key=len, reverse=True)
KIN_RE = re.compile(r"\b(?:great-)*(?:" + '|'.join(map(re.escape, KIN)) + r")s?\b")
PROFESSIONS = r'doctor|engineer|teacher|lawyer|nurse|pilot|architect|dancer|singer|farmer|businessman|professor|clerk|banker|actor|player'

OPENINGS = {
    'pointing / showing': r'\b(pointing (?:to|at|towards)|showing|looking at|while showing)\b',
    'photograph / portrait': r'\b(photograph|photo|portrait|picture)\b',
    'introducing': r'\bintroduc',
    'direct speech (quotes)': r'["“”]',
    'reported speech (said / told)': r'\b(said|says|told|tells|asked|replied|remarked)\b',
    'coded symbols ("A + B means")': r"\b[a-z]\s*[^\w\s,.?!'\"()]{1,2}\s*[a-z]\b.*\bmeans?\b|\bmeans?\b.*\b[a-z]\s*[^\w\s,.?!'\"()]{1,2}\s*[a-z]\b",
    'coded letters ("A $ B" / "P # Q")': r'\b[a-z]\s*[$#@%&*]\s*[a-z]\b',
    'family of N / members': r'\b(family of|members?)\b',
    'couples / generations': r'\b(couples?|generations?)\b',
    'directions / study the information': r'\b(directions?|study the following|read the following|read the information)\b',
    'question set (1-5)': r'\(\s*\d\s*[-–to]+\s*\d\s*\)',
    'statements I / II (sufficiency)': r'\bstatements?\s*(?:i\b|1\b|:)|\bi\.\s.*\bii\.\s',
    'which is true / conclusions': r'\b(definitely true|is true|are true|is false|conclusions?|follows?)\b',
    'cannot be determined in options': r'\b(cannot be determined|can\'t be determined|data inadequate|none of these)\b',
    'story / narrative (past tense, 60+ words)': r'',   # set from length and tense below
}
QUESTION_FORMS = {
    'how is X related to Y': r'\bhow is\b.*\brelated to\b',
    'how are X and Y related': r'\bhow are\b.*\brelated\b',
    'X is related to Y as': r'\brelated to\b.*\bas\b',
    'what is the relation(ship)': r'\bwhat is the relation',
    "how is X's R related": r"\bhow is \w+'s\b",
    'who is the R of': r'\bwho is (?:the )?\w+(?:-in-law)? of\b',
    'whose R is': r'\bwhose\b',
    'X is the what of Y': r'\bis the what\b|\bwhat of\b|\bis \w+\'s what\b',
    'how many': r'\bhow many\b',
    'which of the following': r'\bwhich of the following\b',
    'gender of': r'\b(gender|male or female|is a man|is a woman)\b',
    'which pair / who among': r'\b(which pair|who among|which (?:two|one))\b',
    'no question mark': r'',   # set below
}
CONSTRUCTIONS = {
    "possessive chain (X's R's R)": r"\w+'s \w+(?:-in-law)?'s\b",
    'of chain (the R of the R of)': r'\b(?:the )?\w+(?:-in-law)? of (?:the |his |her |my )?\w+(?:-in-law)? of\b',
    'relative clause (who / whose / whom)': r'\b(who|whose|whom)\b(?!.*\?$)',
    'pronoun he / she / his / her': r'\b(he|she|his|her|him)\b',
    'first person (my / I / me)': r'\b(my|i|me|mine)\b',
    'second person (your / you)': r'\b(your|you)\b',
    'only': r'\bonly\b',
    'negation (not / no / neither)': r"\b(not|no|neither|nor|isn't|doesn't)\b",
    'married / unmarried / bachelor': r'\b(married|unmarried|bachelor|spinster|wedding)\b',
    'elder / younger / twin': r'\b(elder|younger|eldest|youngest|older|twin)\b',
    'step / half / adopted / late': r'\b(step\w*|half-\w+|adopted|late)\b',
    'paternal / maternal side': r'\b(paternal|maternal)\b',
    'great- / grand- relations': r'\b(great-|grand)',
    'professions': rf'\b({PROFESSIONS})s?\b',
    'Hindi kin words': r'\b(' + '|'.join(sorted(set(HINDI.values()) | {'mausa', 'chachi', 'mami', 'devar', 'jeth', 'nanad', 'saala', 'saali'})) + r')\b',
    'respectively / coordination': r'\b(respectively|and the \w+ of)\b',
    'single letters as names (3+)': r'',   # set below
    'all lower case': r'',
}
SYMBOLS = '+-−×÷*@#$%&!^=/<>~'


def answer_text(r) -> str:
    if r.get('answer_text'):
        return str(r['answer_text'])
    for label, text in r.get('options') or []:
        if str(label).lower() == str(r.get('answer') or '').lower():
            return text
    t = r.get('truth')
    return t if isinstance(t, str) else ''


def text_of(r) -> str:
    if 'question' in r:
        opts = ' '.join(f'({l}) {t}' for l, t in r.get('options') or [])
        return f"{r['question']} {opts}".strip()
    return r.get('text') or ' '.join(r.get('tokens') or [])


def band(x, edges) -> str:
    for e in edges:
        if x <= e:
            return f'<= {e}'
    return f'> {edges[-1]}'


def skeleton(text: str) -> str | None:
    """The question sentence with people as @, relation words as R and numbers as #."""
    try:
        toks = prepare(text)['tokens']
    except Exception:
        return None
    if '?' not in toks:
        return None
    end = toks.index('?')
    start = max([i + 1 for i, t in enumerate(toks[:end]) if t in ('.', '!', ':', ';')] or [0])
    while start < end and toks[start] in ('"', "'", '\u201c', '\u201d', ',', '-'):   # the quote a sentence closed
        start += 1
    out = []
    for t in toks[start:end + 1]:
        t = t.lower()
        out.append('@' if re.fullmatch(r'[@#]\d+', t) else '#' if re.fullmatch(r'\d+', t) else 'R' if KIN_RE.fullmatch(t) else t)
    return ' '.join(out) if len(out) <= 25 else None


def features(r) -> dict:
    """{family: set of features} for one question (real row or generated row)."""
    full = text_of(r)
    raw = full[:full.rfind('?') + 1] if '?' in full else full   # the question body: options would count their kin words
    t = raw.lower()
    words = len(t.split())
    f = collections.defaultdict(set)
    for name, rx in OPENINGS.items():
        if rx and re.search(rx, t):
            f['opening'].add(name)
    if words >= 60 and re.search(r'\b(was|were|went|visited|had|met|came)\b', t):
        f['opening'].add('story / narrative (past tense, 60+ words)')
    qpart = t[:t.rfind('?') + 1] if '?' in t else t
    for name, rx in QUESTION_FORMS.items():
        if rx and re.search(rx, qpart):
            f['question form'].add(name)
    if '?' not in t:
        f['question form'].add('no question mark')
    for name, rx in CONSTRUCTIONS.items():
        if rx and re.search(rx, t):
            f['construction'].add(name)
    if len(set(re.findall(r'\b[A-Z]\b', raw))) >= 3:
        f['construction'].add('single letters as names (3+)')
    if raw == t and re.search(r'[a-z]', t):
        f['construction'].add('all lower case')
    for w in set(KIN_RE.findall(t)):
        f['kin word'].add(re.sub(r's$', '', w) if not w.endswith('ss') else w)
    syms = {c for c in raw if c in SYMBOLS}
    if syms and re.search(r'\bmeans?\b', t):
        f['coded'].add(f'{len(syms)} symbols')
        ops = max((len(re.findall(r'[' + re.escape(SYMBOLS) + r']', m)) for m in re.findall(r'(?:\b[A-Z]\s*[^\w\s,.?!]\s*)+\b[A-Z]\b', raw)), default=0)
        f['coded'].add(f'expression with {ops} operators' if ops else 'no expression')
    f['length (words)'].add(band(words, (20, 40, 80, 120, 170, 240, 320)))
    try:
        people = len({e['id'] for e in prepare(raw)['ents']})
    except Exception:
        people = 0
    f['people'].add(band(people, (2, 3, 4, 5, 6, 8, 12, 20, 32)))
    opts = r.get('options') or re.findall(r'(?:^|\s)\(?[a-eA-E1-5][.)]\s', full[len(raw):])
    f['options'].add(band(len(opts), (0, 3, 4, 5)) if opts else 'none (open question)')
    ans = re.sub(r'\s+', ' ', answer_text(r).lower()).strip()
    if ans:
        f['answer'].add(ans if len(ans) <= 40 else ans[:40])
    return f


def _profile(rows):
    counts = collections.defaultdict(collections.Counter)
    per_row = []
    for r in rows:
        fs = features(r)
        per_row.append(fs)
        for fam, names in fs.items():
            counts[fam].update(names)
    return counts, per_row


def mine(real_rows, gen_rows, trace=None, min_skeleton: int = 3) -> dict:
    """The report (counts and masked skeletons only). trace: {text: ok} for the real rows (eval.trace records)."""
    rc, real_feats = _profile(real_rows)
    gc, _ = _profile(gen_rows)
    nr, ng = max(len(real_rows), 1), max(len(gen_rows), 1)
    acc = collections.defaultdict(collections.Counter)
    if trace:
        for r, fs in zip(real_rows, real_feats):
            ok = trace.get(text_of(r))
            if ok is None:
                continue
            for fam, names in fs.items():
                for n in names:
                    acc[(fam, n)]['n'] += 1
                    acc[(fam, n)]['ok'] += bool(ok)
    report = {}
    for fam in sorted(set(rc) | set(gc)):
        if fam == 'answer':   # answers: real ones only, plus the generator's share of the same answers
            names = [n for n, _ in rc[fam].most_common(60)]
        else:
            names = sorted(set(rc[fam]) | set(gc[fam]), key=lambda n: -rc[fam][n])
        report[fam] = {}
        for n in names:
            e = {'real': round(rc[fam][n] / nr, 4), 'real_n': rc[fam][n], 'gen': round(gc[fam][n] / ng, 4)}
            e['gen/real'] = round(e['gen'] / e['real'], 2) if e['real'] else None
            a = acc.get((fam, n))
            if a and a['n']:
                e['model_acc'], e['missed'] = round(a['ok'] / a['n'], 3), a['n'] - a['ok']
            report[fam][n] = e
    sk_real = collections.Counter(s for s in map(skeleton, map(text_of, real_rows)) if s)
    sk_gen = collections.Counter(s for s in map(skeleton, map(text_of, gen_rows)) if s)
    skeletons = [{'skeleton': s, 'real': c, 'gen': sk_gen[s]} for s, c in sk_real.most_common() if c >= min_skeleton]
    gaps = sorted(({'family': fam, 'feature': n, **e} for fam, d in report.items() for n, e in d.items()
                   if fam != 'answer' and e['real'] >= 0.01 and (e['gen'] < 0.5 * e['real'])),
                  key=lambda g: -(g['real'] - g['gen']))
    misses = sorted(({'family': fam, 'feature': n, **e} for fam, d in report.items() for n, e in d.items() if e.get('missed')),
                    key=lambda g: -g['missed'])
    return {'real_questions': len(real_rows), 'generated_rows': len(gen_rows), 'features': report,
            'skeletons': skeletons, 'gaps': gaps, 'misses': misses[:40]}


def generated_sample(n: int, seed: int, files=(), limit: int = 20000):
    if files:
        from eval.balance_report import rows_of
        rows = []
        for f in files:
            rows += list(rows_of(f, limit))
        random.Random(seed).shuffle(rows)
        return rows[:n] if n else rows
    from kinlogic.generate import sample
    rng, out = random.Random(seed), []
    while len(out) < n:
        ex = sample(rng)
        if ex:
            out.append(ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='data/real_questions/*.jsonl')
    ap.add_argument('--trace', default=None, help='eval.trace JSONL of a dev run: adds model accuracy per feature')
    ap.add_argument('--gen_n', type=int, default=6000, help='generated questions to compare against (0: all of --gen_files)')
    ap.add_argument('--gen_files', nargs='*', default=[], help='training files to compare against instead of fresh generator output')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--min_skeleton', type=int, default=3)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from eval.locked import hashes
    from eval.run_real import load_real, split_of
    from eval.verify_keys import qhash
    locked = hashes()
    real = [r for r in load_real(a.pattern) if split_of(r) == 'dev' and qhash(r) not in locked]
    trace = None
    if a.trace:
        trace = {}
        for line in open(a.trace):
            rec = json.loads(line)
            if rec.get('split', 'dev') == 'dev':
                trace[rec['text']] = rec.get('ok')
    rep = mine(real, generated_sample(a.gen_n, a.seed, a.gen_files), trace, a.min_skeleton)
    print(f"{rep['real_questions']} real dev questions vs {rep['generated_rows']} generated")
    print('\nGaps (common in real questions, under half as common in generated ones):')
    for g in rep['gaps'][:30]:
        print(f"  {g['family']:15s} {g['feature']:45s} real {g['real']:6.1%}  gen {g['gen']:6.1%}"
              + (f"  model {g['model_acc']:.0%}" if 'model_acc' in g else ''))
    if rep['misses']:
        print('\nMost missed dev questions by feature:')
        for g in rep['misses'][:20]:
            print(f"  {g['family']:15s} {g['feature']:45s} missed {g['missed']:4d}  accuracy {g['model_acc']:.0%}  gen {g['gen']:6.1%}")
    print(f"\n{len(rep['skeletons'])} question skeletons shared by >= {a.min_skeleton} dev questions; top 10:")
    for s in rep['skeletons'][:10]:
        print(f"  {s['real']:4d} real {s['gen']:5d} gen  {s['skeleton']}")
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rep, f, indent=1)


if __name__ == '__main__':
    main()
