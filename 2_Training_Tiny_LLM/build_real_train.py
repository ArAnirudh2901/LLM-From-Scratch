"""New real exam questions for TRAINING only (RL pool and STaR rows; never evaluation). Three stages:

    python build_real_train.py clean                     # raw/*.jsonl -> clean.jsonl + report.json (no network)
    GEMINI_API_KEY=... python build_real_train.py keys   # independent key check, resumable -> keys.json
    python build_real_train.py export                    # -> pool.jsonl (train_rlvr --pool), keyed.jsonl.gz (star.py)

clean: same text clean-up and validity rules as the evaluation set (eval.real_clean.broken), then every question is
dropped whose wording an evaluation set or our training data already has: exact text, the masked wording (people
replaced by placeholders, so a renamed copy is caught) and near copies (masked 3-gram Jaccard >= 0.8, e.g. one word
changed). Exact readers (never wrong where they read) check the published key where they apply.
keys: like eval.verify_keys. Keyed: a screen solve agreeing with the key confirms it; otherwise a separate solve and
review must agree on another option to correct it, else ambiguous (dropped). Unkeyed: two separate solves must agree.
export: only confirmed / corrected / agreed keys. keyed.jsonl.gz is the symgen format, so star.py turns it into
training rows {kind, text, tokens, lf, truth} (prepared tokens + engine-checked facts), the format of every row the
reader trains on.
"""
from __future__ import annotations

import collections
import glob
import gzip
import hashlib
import html
import json
import os
import re
import sys
import unicodedata

D = 'data/real_train2'
OK = ('confirmed-exact', 'confirmed', 'corrected', 'agreed')
FIGURE = re.compile(r'\b(?:figure|diagram|image|graph)\b|\b(?:tree|chart)\s+(?:shown|given)\b', re.I)
MAX_TOKENS = 400   # block 512 leaves room for the facts


def clean_text(t) -> str:
    from kinlogic.prepare import CITATION
    t = str(t or '')
    for _ in range(3):   # double-escaped pages ('&amp;#x27;')
        u = html.unescape(t)
        if u == t:
            break
        t = u
    t = unicodedata.normalize('NFC', t).replace(' ', ' ').replace('​', '').replace('﻿', '')
    t = re.sub(r'</?(?:b|i|u|em|strong|br|span|sup|sub|p|div|li)\b[^<>]*>', ' ', t, flags=re.I)   # tags only: '<' and '>' are code symbols too
    t = CITATION.sub(' ', t)   # "[SSC CGL 2019]"
    return re.sub(r'\s+', ' ', t).strip()


def clean_row(r) -> dict:
    q = clean_text(r['question'])
    q = re.sub(r'^(?:Q(?:uestion|ues)?\s*(?:no\.?)?\s*)?\d{1,3}\s*[.):\]]\s+', '', q, flags=re.I)   # "Q12." page numbering
    q = re.sub(r'\s*Choose an answer\s*$', '', q, flags=re.I)
    opts = [[l.lower(), re.sub(r'^\s*\(?[A-Ea-e1-5][).]\s*', '', clean_text(t)).rstrip(' .;')] for l, t in r.get('options') or []]
    key = str(r.get('answer') or '').strip().lower() or None
    return {'question': q, 'options': opts, 'answer': key, 'site': r.get('site', ''), 'url': r.get('url', '')}


def masked(text) -> list[str]:
    from kinlogic.prepare import prepare
    return prepare(text)['tokens']


def mkey(tokens) -> str:
    return hashlib.sha1(' '.join(tokens).encode()).hexdigest()


def grams(tokens) -> set:
    return {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}


def norm(t) -> str:
    """Text key; code symbols stay ('X + Z - W' differs from 'X - Z + W')."""
    return re.sub(r'[\s\'"‘’“”.,;:?!()]+', '', str(t).lower())


def _items_text(items):
    for it in items:
        if isinstance(it, dict):
            t = next((it[k] for k in ('text', 'question', 'q') if isinstance(it.get(k), str)), None)
        else:
            t = max((x for x in it if isinstance(x, str)), key=len, default=None) if isinstance(it, (tuple, list)) else None
        if t:
            yield t


def evaluation_texts() -> list[str]:
    """Every question any evaluation set holds (real + its altered/reworded copies, fresh online, hand-written sets)."""
    from eval.run_real import as_text, load_real
    out = [as_text(r) for r in load_real()]
    for f in glob.glob('data/fresh_online/*.jsonl'):
        for line in open(f):
            r = json.loads(line)
            out.append(r['question'] + ' ' + ' '.join(f'({l}) {t}' for l, t in r.get('options') or []))
    import importlib
    for mod, name in (('eval.exam_set', 'Q'), ('eval.fresh_set', 'F'), ('eval.gender_set', 'G'), ('eval.authored_set', 'A')):
        try:
            out += list(_items_text(getattr(importlib.import_module(mod), name)))
        except Exception as e:
            print('could not read', mod, type(e).__name__)
    return out


def training_keys() -> set:
    """Masked wordings the reader has trained on (verified pool, rewrites, STaR/teacher/correction rows)."""
    seen = set()
    files = glob.glob('data/dataset_v6/kinds/*.jsonl.gz') + glob.glob('data/dataset_v6/deep/*.jsonl.gz') + glob.glob('data/dataset_v6/*.jsonl.gz')
    files += [f for f in glob.glob('data/*.jsonl.gz') if 'dev' not in f and 'unseen' not in f]
    for f in files:
        try:
            for line in gzip.open(f, 'rt'):
                t = json.loads(line).get('tokens')
                if isinstance(t, list):
                    seen.add(mkey(t)[:16])
        except (EOFError, OSError):
            pass   # a truncated part file: rows up to the break were read
    return seen


def clean():
    from eval.real_clean import broken, category
    from eval.run_real import as_text
    from eval.sites import KIN
    from kinlogic.codes import _fits, read as read_code
    from kinlogic.exam import _answer_one
    from kinlogic.prepare import prepare
    from kinlogic.rules import read as read_rules
    from kinlogic.speech import read as read_speech
    raw = [json.loads(l) for f in sorted(glob.glob(f'{D}/raw/*.jsonl')) for l in open(f)]
    why = collections.Counter()
    ev = evaluation_texts()
    ev_text = {norm(t) for t in ev}
    ev_tok = [masked(t) for t in ev]
    ev_key = {mkey(t) for t in ev_tok}
    ev_grams = [(g, len(g)) for g in map(grams, ev_tok) if g]
    seen_train = training_keys()
    print(f'raw {len(raw)} | evaluation questions {len(ev)} | training wordings {len(seen_train)}', flush=True)
    out, own_text, own_key = [], set(), set()
    for r0 in raw:
        r = clean_row(r0)
        q, opts, key = r['question'], r['options'], r['answer']
        texts = [t for _, t in opts]
        if not 4 <= len(opts) <= 5:
            why['not 4-5 options'] += 1; continue
        if any(not t for t in texts) or len({norm(t) for t in texts}) < len(texts):
            why['empty or repeated option'] += 1; continue
        if key is not None and key not in [l for l, _ in opts]:
            why['key not among the options'] += 1; continue
        if not KIN.search(q + ' ' + ' '.join(texts)) and not re.search(r'\bmeans\b', q, re.I):
            why['not a blood-relation question'] += 1; continue
        if FIGURE.search(q):
            why['needs a figure'] += 1; continue
        b = broken(r)
        if b:
            why[b] += 1; continue
        try:
            p = prepare(as_text(r))
        except Exception:
            why['text preparation failed'] += 1; continue
        if len(p['options']) != len(opts):
            why['options not separable from the text'] += 1; continue
        if not p['ents']:
            why['no people found'] += 1; continue
        if len(p['tokens']) > MAX_TOKENS:
            why['too long'] += 1; continue
        tk = norm(q + ' '.join(texts))
        k = mkey(p['tokens'])
        if norm(as_text(r)) in ev_text or k in ev_key:
            why['same wording as an evaluation question'] += 1; continue
        g = grams(p['tokens'])
        if g and any(len(g & eg) / (len(g) + n - len(g & eg)) >= 0.8 for eg, n in ev_grams if 0.6 < n / len(g) < 1.67):
            why['near copy of an evaluation question'] += 1; continue
        if k[:16] in seen_train:
            why['already in our training data'] += 1; continue
        if tk in own_text or k in own_key:
            why['repeat within the new data'] += 1; continue
        own_text.add(tk)
        own_key.add(k)
        r['id'] = k[:16]
        r['category'] = category(r)
        r['tokens'] = len(p['tokens'])
        exact = None
        try:
            exact = read_code(p['tokens'], p['options'], p['ents']) or _fits(read_rules(p['tokens']) or read_speech(p['tokens']), p['ents'])
        except Exception:
            pass
        r['exact'] = bool(exact)
        if exact:
            try:
                got = _answer_one(p, exact, lambda bt: [''] * len(bt), None).get('option')
            except Exception:
                got = None
            r['exact_option'] = got
            if key and got == key:
                r['key_status'] = 'confirmed-exact'
        out.append(r)
        why['kept'] += 1
    with open(f'{D}/clean.jsonl', 'w') as f:
        f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    rep = {'raw': len(raw), 'by_reason': dict(why.most_common()),
           'kept_by_site': dict(collections.Counter(r['site'] for r in out)),
           'kept_by_category': dict(collections.Counter(r['category'] for r in out)),
           'keyed': sum(1 for r in out if r['answer']), 'key_confirmed_by_exact_readers': sum(1 for r in out if r.get('key_status')),
           'exact_reader_disagrees_with_key': sum(1 for r in out if r['exact'] and r['answer'] and r.get('exact_option') != r['answer'])}
    json.dump(rep, open(f'{D}/report.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1))


def keys():
    from eval.run_real import as_text
    from eval.verify_keys import CONVENTIONS, MODELS, REVIEW, REVIEW_MODELS, SCREEN, SOLVE, gemini, letter
    rows = [json.loads(l) for l in open(f'{D}/clean.jsonl')]
    out = f'{D}/keys.json'
    done = json.load(open(out)) if os.path.exists(out) else {}
    save = lambda: json.dump(done, open(out, 'w'), indent=0)
    todo = [r for r in rows if r['id'] not in done and r.get('key_status') != 'confirmed-exact']
    print(f'{len(rows)} clean, {len(todo)} to check', flush=True)
    for i in range(0, len(todo), 8):   # 1: screen
        chunk = todo[i:i + 8]
        got = gemini(SCREEN.format(conv=CONVENTIONS, n=len(chunk), qs='\n\n'.join(f'Question {j + 1}: {as_text(r)}' for j, r in enumerate(chunk))), MODELS).get('answers', [])
        if len(got) != len(chunk):
            continue
        for r, g in zip(chunk, got):
            done[r['id']] = {'status': 'confirmed' if r['answer'] and letter(g) == r['answer'] else 'disputed', 'screen': letter(g)}
        save()
        print(f'screen {min(i + 8, len(todo))}/{len(todo)}', flush=True)
    valid = lambda r, x: x in 'abcde'[:len(r['options'])] and x != ''
    second = [r for r in rows if not r['answer'] and done.get(r['id'], {}).get('status') == 'disputed']
    for i in range(0, len(second), 8):   # 2a unkeyed: a second batched solve (review models first, other batches) must agree
        chunk = second[i:i + 8]
        got = gemini(SCREEN.format(conv=CONVENTIONS, n=len(chunk), qs='\n\n'.join(f'Question {j + 1}: {as_text(r)}' for j, r in enumerate(chunk[::-1]))), REVIEW_MODELS).get('answers', [])
        if len(got) != len(chunk):
            continue
        for r, g in zip(chunk[::-1], got):
            v = done[r['id']]
            ok = letter(g) == v['screen'] and valid(r, letter(g))
            v.update(status='agreed' if ok else 'ambiguous', key=letter(g) if ok else None, second=letter(g))
        save()
        print(f'second solve {min(i + 8, len(second))}/{len(second)}', flush=True)
    for r in rows:   # 2b keyed: a separate solve and review where the screen disagrees with the key
        v = done.get(r['id'])
        if not v or v['status'] != 'disputed' or not r['answer']:
            continue
        s = gemini(SOLVE.format(conv=CONVENTIONS, q=as_text(r)), REVIEW_MODELS)
        a, amb = letter(s.get('answer')), bool(s.get('ambiguous'))
        if True:
            rv = gemini(REVIEW.format(conv=CONVENTIONS, key=r['answer'], q=as_text(r)), REVIEW_MODELS)
            b = letter(rv.get('answer'))
            if rv.get('key_correct') is True and not rv.get('ambiguous'):
                v.update(status='confirmed')
            elif a == b and a != r['answer'] and valid(r, a) and not (amb or rv.get('ambiguous')):
                v.update(status='corrected', key=a)
            else:
                v.update(status='ambiguous')
        save()
    print('done', dict(collections.Counter(v['status'] for v in done.values())))


def export():
    rows = [json.loads(l) for l in open(f'{D}/clean.jsonl')]
    ks = json.load(open(f'{D}/keys.json')) if os.path.exists(f'{D}/keys.json') else {}
    pool, st = [], collections.Counter()
    for r in rows:
        v = ks.get(r['id'], {})
        status = r.get('key_status') or v.get('status', 'unchecked')
        key = v.get('key') if status in ('corrected', 'agreed') else r['answer']
        if status in ('corrected', 'agreed') and r.get('exact_option') and r['exact_option'] == key:
            status = 'confirmed-exact'
        if r.get('exact') and r.get('exact_option') and r['exact_option'] != key and status != 'confirmed-exact':
            status = 'ambiguous'   # an exact reading disagrees with every key we have
        st[status] += 1
        if status in OK and key:
            pool.append({**r, 'answer': key, 'key_status': status})
    with open(f'{D}/pool.jsonl', 'w') as f:
        f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pool))
    with gzip.open(f'{D}/keyed.jsonl.gz', 'wt') as f:   # symgen format: star.py input
        f.write(''.join(json.dumps({'question': r['question'], 'options': [t for _, t in r['options']], 'answer': 'abcde'.index(r['answer']),
                                    'style': r['category'], 'source': r['site'], 'id': r['id']}, ensure_ascii=False) + '\n' for r in pool))
    print(json.dumps({'by_key_status': dict(st), 'pool': len(pool), 'needs the model (no exact reading)': sum(1 for r in pool if not r['exact']),
                      'by_category': dict(collections.Counter(r['category'] for r in pool))}, indent=1))


if __name__ == '__main__':
    {'clean': clean, 'keys': keys, 'export': export}[sys.argv[1]]()
