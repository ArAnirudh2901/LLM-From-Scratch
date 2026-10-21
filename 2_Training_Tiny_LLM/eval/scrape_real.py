"""Collects real blood-relation exam questions (question, options, answer key) for EVALUATION and for finding
phrasings the generator lacks. Never used as training text: the questions belong to their sites.

Only sites whose robots.txt allows general crawlers and does not block AI agents; one request every 2 seconds.
Output (git-ignored): data/real_questions/<site>.jsonl  {site, url, question, options: [[label, text]], answer}

    python -m eval.scrape_real [--max-pages 60]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path

from bs4 import BeautifulSoup

UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
OUT = Path('data/real_questions')
DELAY = 2.0


class Polite:
    def __init__(self):
        self.robots, self.last = {}, 0.0

    def allowed(self, url):
        host = urllib.parse.urlsplit(url)
        base = f'{host.scheme}://{host.netloc}'
        if base not in self.robots:
            rp = urllib.robotparser.RobotFileParser(base + '/robots.txt')
            try:
                rp.read()
            except Exception:
                rp = None
            self.robots[base] = rp
        rp = self.robots[base]
        return rp is None or (rp.can_fetch('*', url) and rp.can_fetch('ClaudeBot', url) and rp.can_fetch('anthropic-ai', url))

    def get(self, url):
        if not self.allowed(url):
            print('robots.txt disallows', url)
            return None
        return self.fetch(url)

    def fetch(self, url):
        wait = DELAY - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            print('failed', url, e)
            return None


def clean(t):
    return re.sub(r'\s+', ' ', t).strip()


def indiabix(html, url):
    s = BeautifulSoup(html, 'lxml')
    out = []
    for c in s.select('.bix-div-container'):
        q = c.select_one('.bix-td-qtxt')
        opts = []
        for row in c.select('.bix-opt-row'):
            lab = row.select_one('.bix-td-option span')
            val = row.select_one('.bix-td-option-val')
            if lab and val:
                m = re.search(r'letter-([a-e])', ' '.join(lab.get('class', [])))
                opts.append([m.group(1) if m else '?', clean(val.get_text(' '))])
        ans = c.select_one('.bix-ans-option span[class*=letter]')
        m = re.search(r'letter-([a-e])', ' '.join(ans.get('class', []))) if ans else None
        if q and opts and m:
            out.append({'question': clean(q.get_text(' ')), 'options': opts, 'answer': m.group(1), 'url': url})
    links = {urllib.parse.urljoin(url, a['href']) for a in s.find_all('a', href=True)
             if re.search(r'/blood-relation-test/\d{6}$', a['href'])}
    return out, links


def examveda(html, url):
    s = BeautifulSoup(html, 'lxml')
    out = []
    for art in s.select('article.question-box'):
        q = art.select_one('.question-text')
        opts = []
        for i, lab in enumerate(art.select('.question-option')):
            opts.append(['abcde'[i], clean(lab.get_text(' '))])
        ca = art.get('data-correct-answer')
        if q and opts and ca and ca.isdigit() and 1 <= int(ca) <= len(opts):
            out.append({'question': clean(q.get_text(' ')), 'options': opts, 'answer': 'abcde'[int(ca) - 1], 'url': url})
    links = {urllib.parse.urljoin(url, a['href']) for a in s.find_all('a', href=True)
             if 'practice-mcq-question-on-blood-relation' in a['href'] and re.search(r'[?&](page|section)=\d+', a['href'])}
    return out, links


SITES = {
    'indiabix': ('https://www.indiabix.com/verbal-reasoning/blood-relation-test/', indiabix),
    'examveda': ('https://www.examveda.com/competitive-reasoning/practice-mcq-question-on-blood-relation/', examveda),
}


def crawl(name, start, parse, http, max_pages):
    seen, todo, rows = set(), [start], []
    while todo and len(seen) < max_pages:
        url = todo.pop(0)
        if url in seen:
            continue
        seen.add(url)
        html = http.get(url)
        if not html:
            continue
        got, links = parse(html, url)
        rows += got
        todo += sorted(l for l in links if l not in seen)
    uniq = {}
    for r in rows:
        uniq.setdefault(r['question'].lower(), {**r, 'site': name})
    return list(uniq.values()), len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', default=','.join(SITES))
    ap.add_argument('--max-pages', type=int, default=60)
    ap.add_argument('--blogs', default=None, help='file of article URLs to read as well')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    http = Polite()
    if a.blogs:
        rows = blog_pages(http, [u.strip() for u in open(a.blogs) if u.strip()])
        uniq = {}
        for r in rows:
            uniq.setdefault(r['question'].lower(), {**r, 'site': urllib.parse.urlsplit(r['url']).netloc})
        with open(OUT / 'blogs.jsonl', 'w') as f:
            for r in uniq.values():
                f.write(json.dumps(r) + '\n')
        print(f'blogs: {len(uniq)} questions')
        return
    for name in a.sites.split(','):
        start, parse = SITES[name]
        rows, pages = crawl(name, start, parse, http, a.max_pages)
        with open(OUT / f'{name}.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        print(f'{name}: {len(rows)} questions from {pages} pages')



# ---------- prep-blog articles (read through Jina Reader as text) ----------

Q_START = re.compile(r'^\s*(?:#+\s*)?(?:\*\*)?\s*(?:Q(?:uestion)?\s*\.?\s*)?(\d{1,2})\s*[.):]\s*(?:\*\*)?\s*(.*)$', re.I)
ANS_LABEL = re.compile(r'(?:correct\s+)?(?:answer|ans)\b\.?\s*(?:is)?\s*[:\-–]?\s*(?:\*\*)?\s*(?:option\s*)?\(?([a-eA-E1-5])\)?(?=[\s.):*]|$)', re.I)
ANS_TEXT = re.compile(r'(?:answer|ans)\b\.?\s*[:\-–]\s*(?:\*\*)?\s*(.+?)\**\s*$', re.I)
KEY_LINE = re.compile(r'^\s*(?:\*\*)?\s*(?:Q\.?\s*)?(\d{1,2})\s*[.):\-]?\s*(?:\*\*)?\s*(?:ans(?:wer)?\s*[:.\-]?\s*)?\(?([a-eA-E])\)?\s*(?:\*\*)?\s*$', re.I)
RELWORDS = re.compile(r'\b(father|mother|son|daughter|brother|sister|husband|wife|uncle|aunt|nephew|niece|grand\w+|'
                      r'cousin|in-law|married|sibling|child|parent|means)\b', re.I)


def blog(text, url):
    """Numbered questions with options, answers inline ("Answer: B") or in a key ("1. (b)"); questions of a set
    without their own facts inherit the set's description."""
    from kinlogic.prepare import split_options
    lines = text.splitlines()
    blocks, cur = [], None
    for ln in lines:
        m = Q_START.match(ln)
        if m and not KEY_LINE.match(ln) and len(m.group(2)) > 8:
            cur = {'n': int(m.group(1)), 'lines': [m.group(2)]}
            blocks.append(cur)
        elif cur is not None:
            cur['lines'].append(ln)
    keys = {}
    for ln in lines:
        k = KEY_LINE.match(ln)
        if k:
            keys.setdefault(int(k.group(1)), k.group(2).lower())
    out, head = [], ''
    for b in blocks:
        body, ans, ans_text, in_body = [], None, None, True
        for ln in b['lines']:
            if re.search(r'\b(answer|ans)\b', ln, re.I):
                m = ANS_LABEL.search(ln)
                if m:
                    ans = m.group(1).lower()
                else:
                    t = ANS_TEXT.search(ln)
                    ans_text = clean(t.group(1)).rstrip('.') if t else ans_text
                break
            if re.match(r'^\s*(\*\*)?(solution|explanation|sol)\b', ln, re.I):
                in_body = False   # keep looking for the answer line after the solution
            if in_body:
                body.append(ln)
        txt = clean(' '.join(body).replace('**', ''))
        q, opts = split_options(txt)
        if not opts:
            opts = []
        ans = ans or keys.get(b['n'])
        if ans and ans.isdigit():
            ans = 'abcde'[int(ans) - 1]
        sentences = re.split(r'(?<=[.!?])\s+', q)
        if len(RELWORDS.findall(q)) >= 3 and len(sentences) >= 3:
            head = ' '.join(sentences[:-1])   # a set's description, shared by its next questions
        elif len(RELWORDS.findall(q)) <= 1 and head:
            q = head + ' ' + q
        if (ans and opts) or ans_text:
            out.append({'question': q, 'options': [[l, t] for l, t in opts], 'answer': ans if opts else None,
                        'answer_text': ans_text, 'url': url})
    return out


def blog_pages(http, urls):
    rows = []
    for url in urls:
        if not http.allowed(url):   # the site's own rules; Jina Reader only renders the page to text
            print('robots.txt disallows', url)
            continue
        text = http.fetch('https://r.jina.ai/' + url)
        if text:
            got = blog(text, url)
            print(f'{len(got):3d} questions  {url}')
            rows += got
    return rows


if __name__ == '__main__':
    main()
