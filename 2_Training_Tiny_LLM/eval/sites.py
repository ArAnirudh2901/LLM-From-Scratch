"""Multi-page crawlers for real blood-relation exam questions. EVALUATION and pattern mining only, never training text.

Every site gets its own link rules; one strict parser reads all text layouts (numbered or "Question : 1" stems,
"(a)", "a)", "A." and inline options, numbered option lists, "ANSWER: (d)", "Ans : A.", "Correct Answer is: Brother",
"S1. Ans.(c)" keys, "Directions (1-5)" blocks shared by a set). Every question is validated (stem, 2-6 options in order,
answer among them); rejects and gaps in a page's numbering are reported, so missed questions are visible.
robots.txt is checked for every URL (general crawlers and AI agents); one request per 2 s per site.

    python -m eval.sites [--sites meritnotes,careerride] [--max-pages 400]
Output (git-ignored): data/real_questions/<site>.jsonl, data/real_questions/report.json
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import re
import urllib.parse
from pathlib import Path

from bs4 import BeautifulSoup

from eval.scrape_real import OUT, Polite, clean, examveda, indiabix

KIN = re.compile(r'\b(father|mother|brother|sister|son|daughter|uncle|aunt|nephew|niece|husband|wife|grand\w*|cousin|'
                 r'in-law|married|sibling|child|children|parent|family|related|relation)\b', re.I)
KIN_NOUN = re.compile(r'\b(?:father|mother|brother|sister|son|daughter|uncle|aunt|nephew|niece|husband|wife|grand\w*|'
                      r'cousin|\w+-in-law|married|sibling|child|children|parent|spouse|couple|bachelor|means)s?\b', re.I)
BLOCKS = ['p', 'div', 'li', 'tr', 'td', 'th', 'table', 'ol', 'ul', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'section',
          'article', 'blockquote', 'pre', 'dd', 'dt', 'center']
LABELS = 'abcdef'
CACHE = OUT / 'cache'   # raw pages, so parser fixes re-run without fetching again

QSTART = re.compile(r'^(?:Q(?:uestion|ues)?\s*(?:no)?\s*[.:\-]?\s*)?(\d{1,3})\s*[.):\]]\s*(.*)$', re.I)
QHEAD = re.compile(r'^Q(?:uestion)?\s*(?:no)?\s*[.:\-]?\s*(\d{1,3})\s*[.:)\-]?$', re.I)     # "Question : 1" alone
OPT = re.compile(r'^(?:\d+\.\s+)?(?:\(([a-fA-F])\)|([a-fA-F])\s*[.)])\s*(.+)$')
ANS = re.compile(r'^(?:correct\s+)?(?:answer|ans|option)\b\s*(?:is|key)?\s*[.:\-–]*\s*(.*)$', re.I)
KEY = re.compile(r'^[SQ]?\s*\.?\s*(\d{1,3})\s*[.):\-]?\s*(?:ans(?:wer)?\b\s*[.:\-]*\s*\(?([a-e])\b\)?|[.):\-]\s*\(?([a-e])\)?\s*$)', re.I)
DIRECTIONS = re.compile(r'^directions?\s*(?:\(?\s*(?:q(?:uestions?|\.)?\s*(?:no\.?)?\s*)?(\d{1,3})\s*(?:-|–|to)\s*(\d{1,3})\s*\)?\s*[:.\-–]?|(?=[:\-–])[:\-–]|\s+(?:for|to)\s+(?:questions?|q\.?|solve)\b)', re.I)
DEF = re.compile(r"^\W*(?:\d{1,2}\s*[.)]\s*)?(?:\(?[a-f][.)]\s*)?['‘\"]?\s*[A-Z]\s*(?:[^\w\s]{1,2}|[xX])\s*[A-Z]\s*['’\"]?\s*(?:means|implies|denotes|represents|indicates|stands for|signifies)\b")
EXPR = re.compile(r"(?<![A-Za-z])[A-Z]\s*(?:[^\w\s,.?;:'\"“”‘’()\[\]]{1,2}|[xX](?=\s))\s*[A-Z](?![a-z])")
INLINE = re.compile(r'(?:^|(?<=\s))\(?([a-fA-F])\s*[.)]\s+')


def html_lines(html, root=None):
    """The page's main content as lines: block elements and <br> end lines; unlabelled numbered option lists get
    "(a) ", "(b) " labels; hidden answer boxes stay (they are in the HTML)."""
    s = BeautifulSoup(html, 'lxml')
    for x in s(['script', 'style', 'noscript', 'nav', 'header', 'footer', 'aside', 'iframe', 'svg', 'button']):
        x.decompose()
    node = (s.select_one(root) if root else None) or s.body or s
    for ol in node.find_all(['ol', 'ul']):
        lis = ol.find_all('li', recursive=False)
        if 2 <= len(lis) <= 6 and not any(OPT.match(clean(li.get_text(' '))) for li in lis) \
                and all(len(clean(li.get_text(' '))) <= 80 for li in lis) and not any(li.find(['ol', 'ul', 'a']) for li in lis):
            for k, li in enumerate(lis):
                li.insert(0, f'({LABELS[k]}) ')
    for br in node.find_all('br'):
        br.replace_with('\n')
    for b in node.find_all(BLOCKS):
        b.insert_before('\n')
        b.insert_after('\n')
    lines = []
    for l in node.get_text(' ').split('\n'):
        l = clean(l.replace(' ', ' '))
        if l:
            lines += split_inline(l)
    return lines


def split_inline(line):
    """'A. Mother B. Cousin C. Aunt D. Sister' -> option lines (labels a, b, c, ... in order, each short)."""
    ms = list(INLINE.finditer(line))
    seq = []
    for m in ms:
        if m.group(1).lower() == LABELS[len(seq)]:
            seq.append(m)
    if len(seq) < 3 or (seq[0].start() > 0 and not re.search(r'[?:.]\s*$', line[:seq[0].start()])):
        return [line]
    parts = [line[:seq[0].start()].strip()] if seq[0].start() > 0 else []
    for k, m in enumerate(seq):
        end = seq[k + 1].start() if k + 1 < len(seq) else len(line)
        txt = line[m.end():end].strip()
        if len(txt) > 90:
            return [line]
        parts.append(f'({LABELS[k]}) {txt}')
    return parts


def _delim(line):
    m = re.match(r'^\W*(?:Q(?:uestion|ues)?\s*(?:no)?\s*[.:\-]?\s*)?\d{1,3}\s*([.):\]])', line, re.I)
    return m.group(1) if m else None


def _norm(t):   # code symbols (%, @, $, #, +, ...) stay: they tell expression options apart
    return re.sub(r"[\s\-–—_'’‘\"“”.,;:()]+", ' ', t.lower()).strip()


def _answer(raw, opts):
    """Label from an answer line: '(d) None of these', 'A.', 'B', '3' (1-based), or the option's text."""
    raw = raw.strip().strip('*').strip()
    labs = [l for l, _ in opts]
    m = re.match(r'^(?:option\s*)?(?:\(([a-fA-F])\)|([a-fA-F])(?:[.):]|$))', raw)
    if m and (m.group(1) or m.group(2)).lower() in labs:
        return (m.group(1) or m.group(2)).lower()
    m = re.match(r'^\(?([1-6])\)?(?:[.):\s]|$)', raw)
    if m and int(m.group(1)) <= len(opts):
        return labs[int(m.group(1)) - 1]
    t = _norm(raw)
    hits = [l for l, o in opts if _norm(o) == t] or \
           [l for l, o in opts if len(_norm(o)) > 3 and t.startswith(_norm(o) + ' ')]   # "Son-in-law. Explanation..."
    return hits[0] if len(hits) == 1 else None


def parse_items(lines, url):
    """Numbered (or 'Question : n') items: stem lines, options in order, then the answer; explanation lines are
    skipped until the next question. Returns (items, rejects, numbers seen)."""
    items, cur, keys = [], None, {}
    direc = None   # (text, lo, hi)
    collecting = None
    defs, last_def = [], False   # the page's latest block of symbol definitions ("A + B means A is the son of B")
    for l in lines:
        if DEF.match(l):
            defs = defs + [l] if last_def else [l]
            last_def = True
        else:
            last_def = False
        d = DIRECTIONS.match(l)
        if d:
            collecting = [l[d.end():].lstrip(' :.-–')]
            direc = None
            lo, hi = (int(d.group(1)), int(d.group(2))) if d.group(1) else (None, None)
            cur = None
            continue
        k = KEY.match(l)
        if k and (k.group(2) or k.group(3)) and len(l) < 60 and not OPT.match(l):
            keys[int(k.group(1))] = (k.group(2) or k.group(3)).lower()
            if cur and cur['n'] == int(k.group(1)) and not cur['ans']:
                cur['ans'] = keys[cur['n']]
            continue
        h = QHEAD.match(l)
        q = QSTART.match(l) if not h else None
        if q and (OPT.match(q.group(2) or '') or DEF.match(q.group(2) or '')):
            q = None                                   # "1. A.Father" or "1. J + K means ...": not a question
        if q and cur and not cur['opts'] and not cur['done'] and \
                (int(q.group(1)) != cur['n'] + 1 or _delim(l) != cur['delim']):
            q = None                                   # numbered statements or code definitions inside a stem
        if h or q:
            if collecting is not None:
                direc = (' '.join(collecting), lo, hi)
                collecting = None
            cur = {'n': int((h or q).group(1)), 'stem': [] if h else [q.group(2)], 'opts': [], 'ans': None,
                   'done': False, 'dir': direc, 'delim': _delim(l), 'defs': list(defs)}
            items.append(cur)
            continue
        if collecting is not None:
            collecting.append(l)
            continue
        if cur is None or cur['done']:
            continue
        o = OPT.match(l)
        if o and (o.group(1) or o.group(2)).lower() == LABELS[len(cur['opts'])] and len(o.group(3)) <= 120:
            t = o.group(3)
            if re.search(r'\s+correct$', t, re.I):
                cur['ans'], t = LABELS[len(cur['opts'])], re.sub(r'\s+correct$', '', t, flags=re.I)
            cur['opts'].append([LABELS[len(cur['opts'])], clean(t)])
            continue
        if re.match(r'^(?:(?:view|show|hide|check)\s+answer.*|answer\s*(?:»|>>)?|explanation|solution|shortcut\s*»?)$|^- published on', l, re.I):
            continue                                   # buttons, tabs and dates
        a = ANS.match(l)
        if a and cur['opts']:
            got = _answer(a.group(1), cur['opts'])
            if got or a.group(1).strip():
                cur['ans'] = cur['ans'] or got
                cur['done'] = True
            continue
        if not cur['opts']:
            cur['stem'].append(l)
        elif re.match(r'^(explanation|solution|sol\b|hint)', l, re.I):
            cur['done'] = True
    out, rejects = [], collections.Counter()
    for it in items:
        it['ans'] = it['ans'] or keys.get(it['n'])
        stem = clean(' '.join(it['stem']))
        if it['dir'] and (it['dir'][1] is None or it['dir'][1] <= it['n'] <= it['dir'][2]):
            stem = clean(it['dir'][0] + ' ' + stem)
        if EXPR.search(stem) and not re.search(r'\b(means|implies|denotes|represents|indicates|stands for|signifies)\b', stem, re.I):
            if not it['defs']:
                rejects['no code definitions'] += 1
                continue
            stem = clean(' '.join(re.sub(r'^\W*(?:\d{1,2}\s*[.)]\s*)?\(?[a-f]?[.)]?\s*', '', d) for d in it['defs']) + ' ' + stem)
        why = (None if 2 <= len(it['opts']) <= 6 else 'options') or (None if it['ans'] else 'answer') or \
              (None if KIN.search(stem) else 'not kinship') or (None if not re.search(r'[ऀ-ॿ]', stem) else 'hindi')
        if why:
            rejects[why] += 1
            continue
        out.append({'n': it['n'], 'question': stem, 'options': it['opts'], 'answer': it['ans'], 'answer_text': None, 'url': url})
    return out, rejects, sorted({it['n'] for it in items})


def generic(html, url, root=None):
    return parse_items(html_lines(html, root), url)


def ixambee(html, url):
    """One question per page; the answer is the 0-based index in the page's check_correct() script."""
    s = BeautifulSoup(html, 'lxml')
    opts = [clean(x.get_text(' ')) for x in s.select('span.allOptions_goes_here')]
    opts = [o for o in opts if o]
    m = re.search(r'answerRaw\s*=\s*"([^"]*)"', html)
    first = s.select_one('div[id^=all_option_]')
    stem = ''
    if first is not None:
        box = first.parent
        while box is not None and not clean(box.get_text(' ')).lower().startswith('question'):
            box = box.parent
        if box is not None:
            t, o1 = clean(box.get_text(' ')), clean(first.get_text(' '))
            stem = clean(t[len('question'):t.find(o1)] if o1 and o1 in t else '')
    rej = collections.Counter()
    if not (stem and 2 <= len(opts) <= 6 and m):
        if '/questions/' in url and re.search(r'/\d+$', url):
            rej['layout'] += 1
        return [], rej, []
    raw = m.group(1).strip()
    idx = int(raw) if raw.isdigit() else ('ABCDEF'.find(raw.upper()) if raw else -1)
    if not 0 <= idx < len(opts):
        rej['answer'] += 1
        return [], rej, []
    if not KIN.search(stem):
        rej['not kinship'] += 1   # the site files inequality questions under "coded blood relationship" too
        return [], rej, []
    return [{'n': 1, 'question': stem, 'options': [[LABELS[i], o] for i, o in enumerate(opts)], 'answer': LABELS[idx],
             'answer_text': None, 'url': url}], rej, [1]


def learntheta(html, url):
    """'Q. n' headings; options are <button data-correct="true|false"> in div.options-grid (left unclosed)."""
    out, rej, nums = [], collections.Counter(), []
    for m in re.finditer(r'<h\d[^>]*>(.*?)</h\d>\s*<div class="options-grid"[^>]*>(.*?)</div>', html, re.S):
        stem = clean(BeautifulSoup(m.group(1), 'lxml').get_text(' '))
        q = re.match(r'^Q\.?\s*(\d+)\s*[.:)]?\s*(.*)$', stem)
        if not q:
            continue
        n = int(q.group(1))
        nums.append(n)
        opts = [(c == 'true', clean(BeautifulSoup(t, 'lxml').get_text(' ')))
                for c, t in re.findall(r'data-correct="(true|false)"[^>]*>([^<]*)', m.group(2))]
        opts = [(c, re.sub(r'^[A-F]\)\s*', '', t)) for c, t in opts]
        right = [i for i, (c, _) in enumerate(opts) if c]
        if not 2 <= len(opts) <= 6 or len(right) != 1:
            rej['options' if not 2 <= len(opts) <= 6 else 'answer'] += 1
            continue
        if not KIN.search(q.group(2)):
            rej['not kinship'] += 1
            continue
        out.append({'n': n, 'question': q.group(2), 'options': [[LABELS[i], t] for i, (_, t) in enumerate(opts)],
                    'answer': LABELS[right[0]], 'answer_text': None, 'url': url})
    return out, rej, nums


def qsm(html, url):
    """WordPress 'Quiz and Survey Master' data: question_list JSON; answers PHP-serialized as (text, points, correct)."""
    import html as H
    out, rej, nums = [], collections.Counter(), []
    for m in re.finditer(r'"question_list":', html):
        try:
            ql, _ = json.JSONDecoder().raw_decode(html[m.end():])
        except ValueError:
            continue
        for q in (ql.values() if isinstance(ql, dict) else ql):
            n = int(q.get('question_order') or len(nums) + 1)
            nums.append(n)
            ctx = clean(BeautifulSoup(H.unescape(q.get('question_name') or ''), 'lxml').get_text(' '))
            t = re.search(r's:14:"question_title";s:\d+:"(.*?)";s:', q.get('question_settings') or '', re.S)
            ask = clean(H.unescape(t.group(1))) if t else ''
            ans = re.findall(r'a:3:\{i:0;s:\d+:"(.*?)";i:1;[^;]*;i:2;i:(\d);\}', q.get('answer_array') or '', re.S)
            opts = [clean(H.unescape(x)) for x, _ in ans]
            right = [i for i, (_, c) in enumerate(ans) if c == '1']
            stem = clean(ctx + ' ' + ask)
            if not 2 <= len(opts) <= 6 or len(right) != 1:
                rej['options' if not 2 <= len(opts) <= 6 else 'answer'] += 1
                continue
            if not KIN.search(stem):
                rej['not kinship'] += 1
                continue
            out.append({'n': n, 'question': stem, 'options': [[LABELS[i], o] for i, o in enumerate(opts)],
                        'answer': LABELS[right[0]], 'answer_text': None, 'url': url})
    return out, rej, nums


def learntheta_any(html, url):
    a, b = learntheta(html, url), qsm(html, url)
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def best_of(html, url):
    """Article pages: whichever parser validates the most questions."""
    from eval.scrape_real import blog, quiz
    a = generic(html, url)
    text = '\n'.join(html_lines(html))
    alt = [r for r in max(blog(text, url), quiz(text, url), key=len) if r.get('answer') and r.get('options')]
    if len(alt) > len(a[0]):
        return [{**r, 'n': i + 1, 'answer_text': None} for i, r in enumerate(alt)], collections.Counter(), []
    return a


def _old(parse):
    def f(html, url):
        rows, _ = parse(html, url)
        return [{**r, 'n': i + 1, 'answer_text': None} for i, r in enumerate(rows)], collections.Counter(), list(range(1, len(rows) + 1))
    return f


SITES = {
    'indiabix': dict(start=['https://www.indiabix.com/verbal-reasoning/blood-relation-test/'],
                     follow=r'indiabix\.com/verbal-reasoning/blood-relation-test/\d{6}$', parse=_old(indiabix)),
    'examveda': dict(start=['https://www.examveda.com/competitive-reasoning/practice-mcq-question-on-blood-relation/',
                            'https://www.examveda.com/reasoning-ability/practice-mcq-question-on-blood-relation/'],
                     follow=r'examveda\.com/[a-z-]+/practice-mcq-question-on-blood-relation/?\?(?:page|section)=\d+(?:&(?:page|section)=\d+)?$',
                     parse=_old(examveda)),
    'meritnotes': dict(start=['https://meritnotes.com/reasoning/reasoning-blood-relations/1-77442/'],
                       follow=r'meritnotes\.com/reasoning/[a-z-]*relation[a-z-]*/\d+-\d+/$', root='div.col-lg-7'),
    'careerride': dict(start=['https://www.careerride.com/mcq/blood-relations-and-family-tree-logical-reasoning-mcq-questions-376.aspx',
                              'https://www.careerride.com/view/blood-relation-test-verbal-reasoning-questions-9495.aspx',
                              'https://www.careerride.com/page/blood-relations-and-family-tree-questions-and-answers-747.aspx',
                              'https://www.careerride.com/subject/verbal-reasoning/blood-relation-test.aspx'],
                       follow=r'careerride\.com/(?:mcq|mchoice|view|page|subject)/[^?#]*blood-relation[^?#]*\.aspx$'),
    'careericons': dict(start=['https://careericons.com/verbal-reasoning/blood-relation/conversation-relations/12-1/',
                               'https://careericons.com/verbal-reasoning/blood-relation/puzzles-relations/13-1/',
                               'https://careericons.com/verbal-reasoning/blood-relation/symbol-code-relations/14-1/',
                               'https://careericons.com/verbal-reasoning/blood-relation/2/',
                               'https://careericons.com/practice-test/ibps-po-mains/blood-relations/355-1/',
                               'https://careericons.com/practice-test/ssc-chsl-tier-1/blood-relations/244-1/'],
                        follow=r'careericons\.com/(?:verbal-reasoning/blood-relation/(?:[a-z-]+/)?\d+(?:-\d+)?/|practice-test/[a-z0-9-]+/blood-relations/\d+-\d+/)$'),
    'objectivequiz': dict(start=['https://www.objectivequiz.com/blood-relation-questions-answers'],
                          follow=r'objectivequiz\.com/blood-relation-questions-answers(?:/\d+|\?page=\d+)?$',
                          probe='https://www.objectivequiz.com/blood-relation-questions-answers?page={n}'),
    'examsbook': dict(start=['https://www.examsbook.com/blood-relations-reasoning-questions-answers/1',
                             'https://www.examsbook.com/blood-relation-questions-with-answers',
                             'https://www.examsbook.com/blood-relation-problems-with-solutions-for-ssc-and-bank-exams/1',
                             'https://www.examsbook.com/coded-relation-questions'],
                      follow=r'examsbook\.com/(?:blood-relations?|coded-relation)[a-z-]*(?:/\d+)?$',
                      probe=['https://www.examsbook.com/blood-relations-reasoning-questions-answers/{n}',
                             'https://www.examsbook.com/blood-relation-problems-with-solutions-for-ssc-and-bank-exams/{n}']),
    'ixambee': dict(start=['https://www.ixambee.com/questions/reasoning-aptitude/blood-relations',
                           'https://www.ixambee.com/questions/reasoning-aptitude/coded-blood-relationship'],
                    follow=r'ixambee\.com/questions/reasoning-aptitude/(?:coded-blood-relationship|blood-relations)(?:/\d+|\?page=\d+)?$',
                    parse=ixambee, max_pages=1500),
    'sawaal': dict(start=['https://www.sawaal.com/aptitude-reasoning/verbal-reasoning-mental-ability/blood-relations-questions-and-answers.htm'],
                   follow=r'sawaal\.com/aptitude-reasoning/verbal-reasoning-mental-ability/blood-relations-questions-and-answers\.htm\?page=\d+(?:&sort=)?$',
                   probe='https://www.sawaal.com/aptitude-reasoning/verbal-reasoning-mental-ability/blood-relations-questions-and-answers.htm?page={n}&sort='),
    'articles': dict(start=['https://www.catmock.com/blog/blood-relations-questions-for-cat-25-solved-questions-with-tricks/',
                            'https://www.bankexamstoday.com/2015/11/blood-relation-questions-with-answers.html',
                            'https://www.hitbullseye.com/Reasoning/Blood-Relation-Questions.php',
                            'https://www.smartkeeda.com/blog/blood-relation-questions',
                            'https://www.mbarendezvous.com/cat/blood-relation-questions-for-cat/',
                            'https://cetking.com/6-most-difficult-blood-relationship-questions/',
                            'https://cetking.com/blood-relationship-coding-mba-cet-2023-actual-questions/',
                            'https://parikshala.in/exams/ibps-po/reasoning/blood-relations',
                            'https://byjus.com/govt-exams/blood-relations-logical-reasoning/',
                            'https://www.bookofexams.in/notes/blood-relation-questions-for-bank-exams/',
                            'https://www.bankersadda.com/reasoning-ability-quiz-for-rbi-assistant-ibps-mains-2020-21st-october-puzzle-blood-relation/',
                            'https://www.pw.live/banking/exams/blood-relation-reasoning-for-bank-exam-2026'],
                     follow=None, parse=best_of),
    'learntheta': dict(start=['https://www.learntheta.com/aptitude-questions-blood-relations/',
                              'https://www.learntheta.com/aptitude-blood-relation/',
                              'https://www.learntheta.com/bank-exam-mock-tests-blood-relations/',
                              'https://www.learntheta.com/ssc-cgl-mock-tests-blood-relations/'],
                       follow=r'learntheta\.com/[a-z0-9-]*blood-relation[a-z0-9-]*/(?:page/\d+/)?$', parse=learntheta_any),
}


class Cached:
    def __init__(self, offline=False):
        self.http, self.offline = Polite(), offline
        CACHE.mkdir(parents=True, exist_ok=True)

    def get(self, url):
        f = CACHE / (hashlib.sha1(url.encode()).hexdigest() + '.html')
        if f.exists():
            return f.read_text()
        if self.offline:
            return None
        h = self.http.get(url)
        if h:
            f.write_text(h)
        return h


def complete(r):
    """None when the question can be answered from its own text, else why not: it must name at least two people
    (or speak in the first person), and a symbol expression needs its definitions."""
    from kinlogic.prepare import prepare
    q = r['question']
    if EXPR.search(q) and not re.search(r'\b(means|implies|denotes|represents|indicates|stands for|signifies)\b', q, re.I):
        return 'no code definitions'
    if len(prepare(q)['ents']) < 2 and not re.search(r'\b(my|me|mine|I|myself)\b', q):
        return 'context missing'
    end = q.rfind('?')
    start = max(q.rfind('.', 0, end), q.rfind('!', 0, end)) + 1 if end >= 0 else len(q)
    ask, body = q[start:end + 1], q[:start] + q[end + 1:]
    if not KIN_NOUN.search(body) and len(KIN_NOUN.findall(ask)) < 2:   # "How is G related to H?" from a set
        return 'context missing'
    return None


def crawl(name, cfg, max_pages, offline=False):
    http = Cached(offline)
    parse = cfg.get('parse') or (lambda h, u: generic(h, u, cfg.get('root')))
    seen, todo = set(), list(cfg['start'])
    rows, rejects, pages, gaps = [], collections.Counter(), 0, []

    def visit(url):
        nonlocal pages
        html = http.get(url)
        if not html:
            return None
        pages += 1
        got, rej, nums = parse(html, url)
        rejects.update(rej)
        keep = []
        for r in got:
            why = complete(r)
            if why:
                rejects[why] += 1
            else:
                keep.append(r)
        got = keep
        if nums and got and len(nums) > 1:
            missing = sorted(set(range(min(nums), max(nums) + 1)) - {r['n'] for r in got})
            if missing:
                gaps.append((url, missing[:20]))
        if cfg.get('follow'):
            s = BeautifulSoup(html, 'lxml')
            for a in s.find_all('a', href=True):
                u = urllib.parse.urljoin(url, a['href']).split('#')[0]
                if re.search(cfg['follow'], u) and u not in seen:
                    todo.append(u)
        return got

    while todo and pages < max_pages:
        url = todo.pop(0)
        if url in seen:
            continue
        seen.add(url)
        got = visit(url)
        if got:
            rows += got
    for tpl in [cfg['probe']] if isinstance(cfg.get('probe'), str) else cfg.get('probe') or []:
        empty, n = 0, 2
        while empty < 2 and pages < max_pages and n < 400:   # numbered pages the links did not reach
            url = tpl.format(n=n)
            n += 1
            if url in seen:
                continue
            seen.add(url)
            before = len({r['question'] for r in rows})
            got = visit(url) or []
            rows += got
            empty = 0 if len({r['question'] for r in rows}) > before else empty + 1
    uniq = {}
    for r in rows:
        r.pop('n', None)
        uniq.setdefault(_norm(r['question']) + '|' + '|'.join(_norm(o) for _, o in r['options']), {**r, 'site': name})
    return list(uniq.values()), {'pages': pages, 'questions': len(uniq), 'raw': len(rows), 'rejects': dict(rejects),
                                 'gaps': gaps[:15]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', default=','.join(SITES))
    ap.add_argument('--max-pages', type=int, default=400)
    ap.add_argument('--offline', action='store_true', help='re-parse cached pages only')
    ap.add_argument('--urls', default=None, help='file of start pages on further sites (one per line): a training crawl')
    ap.add_argument('--out', default=None, help='output folder (a training crawl must not write into the evaluation set)')
    a = ap.parse_args()
    global OUT
    if a.urls:   # one generic site per domain: its start pages plus pages on the same topic it links to
        import urllib.parse
        assert a.out and Path(a.out).resolve() != OUT.resolve(), 'a training crawl needs its own --out'
        by = collections.defaultdict(list)
        for u in open(a.urls).read().split():
            by[urllib.parse.urlparse(u).netloc].append(u)
        for dom, us in by.items():
            SITES[dom] = dict(start=us, follow=rf'{re.escape(dom)}/[^?#]*(?:blood|relation|family|kinship)[^?#]*$', max_pages=a.max_pages)
        a.sites = ','.join(by)
    if a.out:
        OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    names = a.sites.split(',')
    report = json.load(open(OUT / 'report.json')) if (OUT / 'report.json').exists() else {}
    with cf.ThreadPoolExecutor(len(names)) as ex:   # sites in parallel; each keeps its own 2 s pace
        futs = {ex.submit(crawl, n, SITES[n], SITES[n].get('max_pages', a.max_pages), a.offline): n for n in names}
        for f in cf.as_completed(futs):
            n = futs[f]
            try:
                rows, rep = f.result()
            except Exception as e:   # one site's failure never loses the others
                print(f'{n}: failed {e!r}', flush=True)
                continue
            with open(OUT / f'{n}.jsonl', 'w') as fh:
                for r in rows:
                    fh.write(json.dumps(r) + '\n')
            report[n] = rep
            print(f"{n}: {rep['questions']} questions from {rep['pages']} pages; rejects {rep['rejects']}; "
                  f"pages with numbering gaps {len(rep['gaps'])}", flush=True)
    json.dump(report, open(OUT / 'report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
