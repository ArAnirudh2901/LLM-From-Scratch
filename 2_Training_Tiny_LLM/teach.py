"""Teacher-labelled training rows (program-aided distillation; learning from executions, Wang et al. 2021): a frontier
model writes the family facts of answer-keyed questions (kinlogic.reword EXTRACT_BATCH), our engine answers from them,
and a row is kept only when that answer is the keyed option. Two independent signals must agree, so wrong keys and
wrong readings both drop out. Only the facts become training data (no teacher weights). Never run on evaluation
questions (eval/run_real.py rows are refused by wording).

    GEMINI_API_KEY=... python teach.py --questions data/symgen/part-k1.jsonl.gz --out data/teach_k1.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import re
import time


def main():
    from kinlogic.exam import _answer_one
    from kinlogic.kin import KIN
    from kinlogic.prepare import prepare
    from kinlogic.reword import EXTRACT_BATCH, _numbered, call_rotating, to_facts
    MODELS = os.environ.get('TEACH_MODELS', 'gemini-3.5-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash,gemini-3-flash-preview').split(',')
    from kinlogic.symgen import as_text, real_keys
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--pause', type=float, default=6.0)
    a = ap.parse_args()
    qs = [json.loads(l) for p in a.questions for f in glob.glob(p) for l in gzip.open(f, 'rt')]
    blocked = real_keys()
    done = set()
    if os.path.exists(a.out):
        try:
            done = {r['text'] for r in map(json.loads, gzip.open(a.out, 'rt'))}
        except (EOFError, OSError):
            pass
    st = collections.Counter()
    todo = [q for q in qs if as_text(q) not in done]
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        texts = [as_text(q) for q in chunk]
        preps = [prepare(t) for t in texts]
        keep = [j for j, p in enumerate(preps) if hashlib.sha1(' '.join(p['tokens']).encode()).hexdigest() not in blocked]
        st['evaluation wording (refused)'] += len(chunk) - len(keep)
        if not keep:
            continue
        try:
            raw = call_rotating(EXTRACT_BATCH.format(n=len(keep), words=', '.join(sorted(KIN)), questions=_numbered([texts[j] for j in keep])),
                                MODELS, temperature=0.0, json_out=True)
            readings = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))['readings']
        except Exception as e:
            st['call failed'] += 1
            print('call failed', type(e).__name__, str(e)[:80], flush=True)
            time.sleep(60)
            continue
        if len(readings) != len(keep):
            st['batch mismatch'] += len(keep)
            continue
        rows = []
        for j, reading in zip(keep, readings):
            q, p = chunk[j], preps[j]
            try:
                lf = to_facts(reading, p['ents'])
                res = _answer_one(p, lf, lambda b: [''] * len(b), None) if lf else {}
            except Exception:
                lf, res = None, {}
            if lf and res.get('option') == 'abcde'[q['answer']]:
                rows.append({'kind': 'teacher-' + q.get('style', 'q'), 'text': texts[j], 'tokens': p['tokens'], 'lf': lf,
                             'truth': q['options'][q['answer']]})
                st['kept'] += 1
            else:
                st['no agreement' if lf else 'unreadable'] += 1
        if rows:
            with gzip.open(a.out, 'at') as f:   # one gzip block per batch: resumable
                f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        print(f'{min(i + a.batch, len(todo))}/{len(todo)} | {dict(st)}', flush=True)
        time.sleep(a.pause)
    print('done', dict(st))


if __name__ == '__main__':
    main()
