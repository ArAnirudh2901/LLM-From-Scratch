"""Focus rows for factored reading: the whole question, then <focus> and one of its sentences -> only the facts that
sentence states (unnamed people numbered from #1 within it; nothing for a sentence that states no fact). The reader then sees every sentence in context but
writes a short reading, and exam.with_split_search(focus=True) joins the sentence readings under the engine.

    python build_focus.py --n 40000 --out data/focus_v1.jsonl.gz

Generated questions only, each solved by the engine first (build_dataset.engine_ok); never real questions. Each row
keeps its source questions ('src': the question and any padding questions) for train_reader's leak guard."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import random
import re
from multiprocessing import Pool

KINDS = {'puzzle': 35, 'statements': 25, 'speech': 15, 'nested': 10, 'story': 8, 'identity': 4, 'introduce': 3}
GROUPS = re.compile(r'(^|; )(claim|part) \d')   # group markers belong to the whole question
P_LONG, LONG = 0.35, (160, 460)   # real focus inputs reach 443 tokens, generated questions ~150: pad some with other families
N_ENT = 32


def focus(tokens, s: int, e: int) -> list[str]:
    from kinlogic.exam import focus_input
    return focus_input(tokens, tokens[s:e])


def local(lf: str) -> str:
    ren = {}
    return re.sub(r'#\d+', lambda m: ren.setdefault(m.group(0), f'#{len(ren) + 1}'), lf)


def filler(rng, used: set, want: int) -> tuple[list[str], list[list[str]]]:
    """Statements about other families (people renumbered past `used`), about `want` tokens: long context whose
    facts the focus row never asks for. Names and letters only, so 'the man' stays unambiguous. Returns the tokens
    and the questions they came from."""
    from kinlogic import generate as gen
    from kinlogic.exam import _sentences
    out, src, nxt = [], [], max((int(t[1:]) for t in used), default=-1) + 1
    for _ in range(60):
        if len(out) >= want:
            break
        ex = rng.choice([gen.q_statements, gen.q_nested, gen.q_story, gen.q_story, gen.q_story])(rng)   # stories: most words per person
        if not ex or any(e['kind'] not in ('name', 'letter') for e in ex['ents']) or '"' in ex['tokens']:
            continue
        body = [t for s in _sentences(list(ex['tokens'])) if s[-1] == '.' for t in s]
        ids = sorted({t for t in body if re.fullmatch(r'@\d+', t)}, key=lambda t: int(t[1:]))
        if not body:   # only the question
            continue
        if nxt + len(ids) > N_ENT:
            break
        ren = {t: f'@{nxt + i}' for i, t in enumerate(ids)}
        nxt += len(ids)
        out += [ren.get(t, t) for t in body]
        src.append(list(ex['tokens']))
    return out, src


def rows_of(ex, rng, per_q: int, p_long: float = 0.0) -> list[dict]:
    units = [u for u in ex.get('units') or [] if not GROUPS.search(u[2])]   # a fact-free sentence teaches an empty reading
    stmts = collections.Counter(a for s in ex['lf'].split(' ; ') for a in set(re.findall(r'#\d+', s)))
    own = lambda lf: collections.Counter(a for s in lf.split(' ; ') for a in set(re.findall(r'#\d+', s)))
    units = [u for u in units if all(stmts[a] == n for a, n in own(u[2]).items())]   # an unnamed person other sentences also name
    rng.shuffle(units)
    rows = []
    for s, e, lf in units[:per_q]:
        toks, src = list(ex['tokens']), [list(ex['tokens'])]
        if rng.random() < p_long:
            pre, more = filler(rng, {t for t in toks if re.fullmatch(r'@\d+', t)}, rng.randint(*LONG) - len(toks))
            toks, s, e, src = pre + toks, s + len(pre), e + len(pre), src + more
        rows.append({'tokens': focus(toks, s, e), 'lf': local(lf), 'kind': 'focus-' + ex['kind'], 'src': src})
    return rows


def _job(args):
    kind, n, seed, per_q = args
    from build_dataset import engine_ok
    from kinlogic import generate as gen
    gen.UNITS = True
    rng = random.Random(seed)
    out, tried = [], 0
    while len(out) < n and tried < n * 20:
        tried += 1
        ex = gen.KINDS[kind](rng)
        if ex and ex.get('units') and engine_ok(ex):
            out += rows_of(ex, rng, per_q, P_LONG)
    return kind, out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40000, help='focus rows in total')
    ap.add_argument('--per_q', type=int, default=3, help='at most this many sentences of one question')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--out', default='data/focus_v1.jsonl.gz')
    a = ap.parse_args()
    tot = sum(KINDS.values())
    jobs, chunk = [], 500
    for kind, w in KINDS.items():
        need = round(a.n * w / tot)
        jobs += [(kind, min(chunk, need - i), a.seed * 100003 + len(jobs) + i, a.per_q) for i in range(0, need, chunk)]
    rows, by = [], collections.Counter()
    with Pool(a.workers) as p:
        for kind, got in p.imap(_job, jobs):   # job order: the same file every run
            rows += got
            by[kind] += len(got)
    random.Random(a.seed).shuffle(rows)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with gzip.open(a.out + '.tmp', 'wt') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(a.out + '.tmp', a.out)
    man = {'rows': len(rows), 'by_kind': dict(by), 'per_q': a.per_q, 'seed': a.seed}
    json.dump(man, open(re.sub(r'(\.jsonl)?(\.gz)?$', '', a.out) + '_manifest.json', 'w'), indent=1)
    print(json.dumps(man))


if __name__ == '__main__':
    main()
