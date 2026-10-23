"""Real training questions from crawled pages whose layouts our parsers miss: a fast model copies each page's
blood-relation questions (text, options, keyed answer) out of the page text, nothing else. Only pages of the training
crawl's domains are read (never the evaluation sites' pages); every question is checked (4-5 options, the key is one
of them, it is about family relations) and dropped when its wording matches an evaluation question. Output rows use
the symgen format, so teach.py and star.py can label them.

    GEMINI_API_KEY=... python llm_extract.py --domains data/real_train_urls.txt --out data/real_train/llm_questions.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import time
import urllib.parse
from pathlib import Path

PROMPT = """Below is the text of a web page with blood-relation (family relation) reasoning questions.
Copy every complete multiple-choice question from it exactly as written, with its options and the answer the page gives.
Skip anything incomplete (missing options or missing answer). Do not solve anything yourself; only copy.
Return JSON only: {{"questions": [{{"question": "...", "options": ["...", "..."], "answer": "the correct option's letter as printed, e.g. c"}}]}}

Page:
{page}"""


def page_url(html: str) -> str | None:
    m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)', html, re.I) or \
        re.search(r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)', html, re.I)
    return m.group(1) if m else None


def main():
    from eval.sites import CACHE, html_lines
    from kinlogic.reword import call
    from eval.run_real import load_real
    norm = lambda t: re.sub(r'[^a-z0-9]+', '', str(t).lower())
    _key = lambda t: norm(t)
    real = [norm(r['question']) for r in load_real()]
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', required=True, help='file of start URLs; their domains are the ones read')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=os.environ.get('EXTRACT_MODEL', 'gemini-3.1-flash-lite'))
    ap.add_argument('--pause', type=float, default=4.0)
    a = ap.parse_args()
    doms = {urllib.parse.urlparse(u).netloc for u in open(a.domains).read().split()}
    blocked, seen, done_pages = set(real) | {x[:80] for x in real}, set(), set()   # evaluation wording, whole or its start
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        for r in map(json.loads, gzip.open(out, 'rt')):
            seen.add(_key(r['question']))
            done_pages.add(r.get('page'))
    pages = []
    for f in sorted(CACHE.glob('*.html')):
        html = f.read_text(errors='ignore')
        u = page_url(html)
        if u and urllib.parse.urlparse(u).netloc in doms and u not in done_pages and re.search(r'blood|relation|family', html, re.I):
            pages.append((u, html))
    print(f'{len(pages)} training pages to read', flush=True)
    kept = 0
    for i, (u, html) in enumerate(pages):
        text = '\n'.join(html_lines(html))[:14000]
        try:
            raw = call(PROMPT.format(page=text), 'gemini', temperature=0.0, json_out=True, model=a.model)
            qs = json.loads(re.search(r'\{.*\}', raw, re.S).group(0)).get('questions', [])
        except Exception as e:
            print(f'{u}: call failed {type(e).__name__}', flush=True)
            time.sleep(30)
            continue
        rows = []
        for q in qs:
            opts = [re.sub(r'^\(?[a-eA-E1-5][).:]\s*', '', str(o)).strip() for o in (q.get('options') or [])]
            key = str(q.get('answer', '')).strip().lower().strip('().')
            if not (4 <= len(opts) <= 5) or len(key) != 1 or key not in 'abcde'[:len(opts)]:
                continue
            text_q = str(q.get('question', '')).strip()
            if not re.search(r'\b(?:father|mother|son|daughter|brother|sister|husband|wife|uncle|aunt|nephew|niece|grand\w*|cousin|related|relation|means)\b', text_q, re.I):
                continue
            k = _key(text_q)
            if k in seen or k in blocked or k[:80] in blocked:
                continue
            seen.add(k)
            rows.append({'question': text_q, 'options': opts, 'answer': 'abcde'.index(key), 'style': 'real', 'page': u})
        if rows:
            with gzip.open(out, 'at') as fh:
                fh.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        kept += len(rows)
        print(f'{i + 1}/{len(pages)} {urllib.parse.urlparse(u).netloc}: +{len(rows)} (total {kept})', flush=True)
        time.sleep(a.pause)


if __name__ == '__main__':
    main()
