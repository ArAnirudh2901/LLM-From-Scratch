"""Turns a pasted question into the reader model's input.

- normalizes quotes, dashes and spacing; drops exam boilerplate ("Question 3:", "Directions: ...")
- sets the answer options aside ("(a) Uncle (b) Cousin", "1. ...", "A.V | B.O")
- replaces every person mention with a placeholder @k: names (capitalized words that are not ordinary words),
  single letters used as people ("A is the brother of B", "P + Q"), and descriptions ("a man" -> "a man @2";
  "the man" refers back to the last "man"). The reader decides which placeholders are people.
Returns tokens (lowercase words and placeholders), the placeholder table, and the options.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_HERE = Path(__file__).parent
MAX_ENTS = 32

# Ordinary words (lowercase). Built from the training text (scripts/build_lexicon); falls back to a small core.
_LEX_FILE = _HERE / 'data' / 'lexicon.txt'
LEXICON = set(_LEX_FILE.read_text().split()) if _LEX_FILE.exists() else set()
LEXICON |= set('''a an the is was are were be been being am has have had do does did and or but if then so as of to
in on at by for with from into about than that this these those who whom whose which what how why when where i me my
mine we us our you your he him his she her hers it its they them their there here not no only also all each both one
two three four five six seven eight nine ten said says say told tell tells introduced introduces introducing
introduce pointing points pointed point looking looks looked showing shows showed related relation relationship
man woman boy girl lady gentleman person family member members married couple photograph photo picture portrait
how many much following among statement statements question answer options none above mr mrs ms dr miss shri smt
sir madam'''.split())
# Capitalized words that are not people when they appear mid-sentence.
CAP_WORDS = set('''monday tuesday wednesday thursday friday saturday sunday january february march april june july
august september october november december christmas easter diwali holi eid thanksgiving halloween valentine
valentines english hindi french spanish chinese american indian british mr mrs ms dr miss shri smt sir madam i ii
iii iv v vi statement question tv ceo'''.split())
# Never names, even capitalized mid-sentence ("..., while The grandfather of Q is P").
FUNCTION_WORDS = set('''the a an his her their my your our its this that these those who whom whose which what how
when where why if then and or but so of to in on at by for with from is was are were has have had be not no also
all each both either neither only there here he she it they we you'''.split())
KIN_TITLES = set('''mother father mom dad mum papa mama grandmother grandfather grandma grandpa granny grandson
granddaughter uncle aunt auntie brother sister son daughter husband wife cousin nephew niece'''.split())
# Names that are also ordinary words: a name at the start of a sentence when a verb or 's follows.
AMBIGUOUS_NAMES = set('''will mark bill rose grace hope faith joy may june april august summer dawn eve iris ivy lily
daisy ruby pearl amber jade sky rain sunny hunter carter mason cooper baker taylor parker gray grey brown white king
prince rich frank earnest victor sage river willow autumn star crystal'''.split())
# Frequent words on exam pages and in plain English that are never people here (capitalised at a sentence start, in a
# heading or a solution); names that are also words (Rose, Will) are handled by AMBIGUOUS_NAMES.
COMMON = set('''about above according additional after again against age ages all also although always among amount
answer answers any anyone anything apart approach area arrangement ask asked asks based basic because before begin
below best better between blood both bank banks brief but by can cannot case cases certain chapter check choice choices
choose clearly code coded codes coding complete concept concepts conclusion conclusions condition conditions consider
content contents context correct correctly could data decide decoded definitely description detail detailed details
diagram difficult difficulty directions doctor does done download each easy either else engineer english enough
evaluate even every exactly exam example examples exams except explanation explained expression expressions facts
false family figure final finally find first five following for four from further general given gives giving going
good great group hard hence here how however important information instruction instructions into its just keep key
know known last lawyer least lecturer less let level like likely look main make manager many marks may maybe medium
member members method might more most much must next nine none nor north note now number numbers objective often once
one only option options order other others our out over page paper part pattern people per person please point
possible practice prelims previous problem problems professor question questions quiz quick read reasoning refer
related relation relations relationship relationships remaining result right rule rules said same scientist section
see seven several shall should show shown since six so solution solutions solve solved some source south statement
statements step steps still study such suppose sure table teacher ten test tests than that their them then there
therefore these they thing things think third this those three through thus time title today total tree tricks true
try two type types under understand upon use used using various very view west what when where whether which while
who whom why with within without would write wrong yes yet you your'''.split())
NAME_NEXT = {'is', 'was', "'s", 'has', 'had', 'and', ',', 'said', 'says', 'told', 'tells', 'introduced', 'introduces'}
# After a sentence-initial "Will", "May", "Hope"... these keep it an ordinary word ("Will you...").
NOT_A_NAME_NEXT = {'you', 'we', 'they', 'i', 'he', 'she', 'it', 'not', 'be', 'the', 'a', 'an', 'this', 'that', 'there'}
# After a single capital "A" these make it a person ("A is ..."); any other lowercase word makes it the article.
I_PRONOUN_NEXT = {'am', "'m", 'have', 'had', 'was', 'will', 'would', 'shall', 'can', 'could', 'do', 'did', 'said',
                  'told', 'asked', 'think', 'know', 'met', 'saw', 'went', 'went', 'live', 'love', 'pointed'}
LETTER_NEXT = NAME_NEXT | {'related', 'in', 'on', 'at', 'to', 'from', 'with', 'or', '?', '.', ':', ';', 'x', 'are',
               'wore', 'walks', 'lost', 'keeps', 'got', 'finished', 'called', 'lives', 'loves', 'went', 'works', 'were', 'married', 'who', 'whose', 'being', 'points', 'pointed', 'pointing',
                           'looks', 'looked', 'showed', 'shows', 'went', 'goes', 'loves', 'got', 'gets'}
HEADS = {'man', 'woman', 'boy', 'girl', 'lady', 'gentleman', 'person', 'guy'}
COPULA = {'is', 'was', 'are', 'were', 'be', 'been', 'being', 'become', 'becomes', 'became'}
ADJ = {'old', 'young', 'elderly', 'little', 'small', 'tall', 'certain', 'unknown', 'other'}
DET_NEW = {'a', 'an', 'another', 'one', 'some'}
DET_BACK = {'the', 'this', 'that'}

TOKEN_RE = re.compile(r"'s\b|\w+(?:-\w+)*|[^\w\s]")
_OPTION_START = re.compile(r'(?:^|\s)(?:\(\s*([a-eA-E1-5])\s*\)|([a-eA-E1-5])[.)](?=\s|\S))')
_BOILER = [re.compile(p, re.I) for p in (
    r'^\s*(?:blood relations?\s+)?(?:question|q|que|ques)\s*\.?\s*\d+\s*[:.)-]\s*',
    r'^\s*directions?\s*(?:\([^)]*\))?\s*:\s*[^.?!]*[.?!]\s*',
    r'^\s*(?:study|read)\s+the\s+(?:following\s+)?information\s+carefully[^.?!:]*[.?!:]\s*',
    r'^\s*\d+\s*[.)]\s+(?=[A-Z])',
)]


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    for a, b in (('‘', "'"), ('’', "'"), ('`', "'"), ('“', '"'), ('”', '"'), ('–', '-'), ('—', '-'),
                 ('−', '-'), ('…', '...'), (' ', ' ')):
        text = text.replace(a, b)
    return re.sub(r'\s+', ' ', text).strip()


def strip_boilerplate(text: str) -> str:
    for _ in range(3):
        before = text
        for r in _BOILER:
            text = r.sub('', text)
        if text == before:
            break
    return text.strip()


_VERBISH = re.compile(r"\b(is|are|was|were|has|have|had|married|means)\b", re.I)


def split_options(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Separates trailing answer options ("(a) Uncle (b) Cousin", "1. Uncle 2. Cousin", "A.V | B.O") from the
    question. Options run in order (a, b, c... or 1, 2, 3...) after the question mark when there is one. Numbered
    sentences ("1. A is the father of B.") are statements of the question, not options. Returns (question, options)."""
    q = text.rfind('?')
    head, tail = (text[:q + 1], text[q + 1:]) if q >= 0 else ('', text)
    if '|' in text:   # "A.V | B.O | C.L" or "A D | B C" (also when "?" is a code symbol inside the question)
        bar = text.index('|')
        starts = [m.start(1) for m in re.finditer(r'(?:^|[\s.?])(\(?[aA1](?:[.):]\s*|\s))', text[:bar])]
        for st in reversed(starts):
            chunks = [c.strip() for c in text[st:].split('|')]
            parsed = [re.match(r'^\(?([a-eA-E1-5])\s*[.):]?\s*(.+)$', c) for c in chunks]
            labels = [m.group(1).lower() for m in parsed if m]
            if all(parsed) and labels[0] in ('a', '1') and all(ord(b) == ord(a) + 1 for a, b in zip(labels, labels[1:])):
                return text[:st].strip(), [(m.group(1).lower(), m.group(2).strip()) for m in parsed]
    marks = list(_OPTION_START.finditer(tail))
    found = []
    for start in range(len(marks)):
        lab = (marks[start].group(1) or marks[start].group(2)).lower()
        if lab not in ('a', '1'):
            continue
        seq, want = [marks[start]], chr(ord(lab) + 1)
        for m in marks[start + 1:]:
            if (m.group(1) or m.group(2)).lower() == want:
                seq.append(m)
                want = chr(ord(want) + 1)
        if len(seq) < 2 or (q < 0 and seq[0].start() == 0):
            continue
        opts = []
        for i, m in enumerate(seq):
            end = seq[i + 1].start() if i + 1 < len(seq) else len(tail)
            opts.append(((m.group(1) or m.group(2)).lower(), tail[m.end():end].strip(' |;,')))
        statements = sum(1 for _, o in opts if _VERBISH.search(o) and len(o.split()) >= 4 and not o.lower().startswith(('only', 'both', 'either', 'neither', 'statement', 'all')))
        if statements * 2 >= len(opts) and not (q >= 0 and lab == 'a'):   # lettered options after "?" may be statements
            continue
        found.append((seq[0].start(), opts))
    if found:
        start, opts = found[-1]
        return (head + tail[:start]).strip(), opts
    return text, []


def _sentence_starts(toks: list[str]) -> set[int]:
    """Indexes of words that start a sentence (also after a colon or an opening quote)."""
    starts, new = set(), True
    for i, t in enumerate(toks):
        if t[0].isalnum() or t[0] == '_':
            if new:
                starts.add(i)
            new = False
        elif t in ('.', '?', '!', ':', '"', "'"):
            new = True
    return starts


def prepare(text: str, names: list[str] | None = None) -> dict:
    """Model input for a pasted question. names: extra strings to treat as people (the user's "add a name").
    Returns {tokens, ents: [{id, text, kind}], options, question}."""
    text = strip_boilerplate(normalize(text))
    text, options = split_options(text)
    raw = []
    for t in TOKEN_RE.findall(text):   # "S-T" in a code expression is two letters, not one word
        if '-' in t and any(len(part) == 1 and part.isupper() for part in t.split('-')):
            for k, part in enumerate(t.split('-')):
                raw += ['-', part] if k else [part]
        else:
            raw.append(t)
    starts = _sentence_starts(raw)
    extra = {n.strip().lower() for n in names or [] if n.strip()}
    lowercase_text = not any(t[0].isupper() for t in raw if t[0].isalpha())
    roman = 'IV' in raw
    letter_names = sum(1 for x in raw if re.fullmatch(r'[A-HJ-Z]', x)) >= 2
    ents, by_text, by_head = [], {}, {}
    out = []

    def ent(key, surface, kind):
        if key in by_text:
            return by_text[key]
        if len(ents) >= MAX_ENTS:
            return None
        e = f'@{len(ents)}'
        ents.append({'id': e, 'text': surface, 'kind': kind})
        by_text[key] = e
        return e

    i = 0
    while i < len(raw):
        t = raw[i]
        low = t.lower()
        nxt = raw[i + 1].lower() if i + 1 < len(raw) else ''
        kind = None
        roman_v = t == 'V' and nxt == '.' and roman and (i == 0 or raw[i - 1] in '.?!')
        prv = raw[i - 1].lower() if i else ''
        letter_i = t == 'I' and letter_names and not roman and nxt not in I_PRONOUN_NEXT and (   # "I is the brother of H"
            nxt in ('is', 'has', 'does', "'s", 'are') or prv in ('of', 'and', 'by', 'to', 'than', 'with', ','))
        if re.fullmatch(r'[A-Z]', t) and (t != 'I' or letter_i) and not roman_v:
            article = t == 'A' and re.fullmatch(r'[a-z]\w*', raw[i + 1] if i + 1 < len(raw) else '') and nxt not in LETTER_NEXT
            if not article:
                kind = 'letter'
        elif low in extra:
            kind = 'name'
        elif re.fullmatch(r'[A-Za-z][A-Za-z-]*', t) and len(t) > 1 and low not in COMMON and not (t.isupper() and len(t) <= 6 and low not in extra):
            if t[0].isupper():
                if i in starts:
                    if low not in LEXICON and low not in KIN_TITLES:
                        kind = 'name'
                    elif low in AMBIGUOUS_NAMES and nxt not in NOT_A_NAME_NEXT:
                        kind = 'name'
                elif low not in CAP_WORDS and low not in KIN_TITLES and low not in FUNCTION_WORDS:
                    kind = 'name'
            elif lowercase_text and low not in LEXICON and low not in KIN_TITLES:
                kind = 'name'
        if kind:
            e = ent(t if kind == 'letter' else low, t, kind)
            out.append(e or '<unk>')
            i += 1
            continue
        out.append(low)
        # descriptions: "a man" / "an old lady" make a new person; "the man" / "that lady" the last one
        prev = raw[i - 1].lower() if i else ''
        predicate = low in DET_NEW and prev in COPULA and not (i >= 2 and raw[i - 2].lower() == 'there')   # "X is a boy" names no one new
        if low in DET_NEW | DET_BACK and not predicate:
            j = i + 1
            while j < len(raw) and raw[j].lower() in ADJ:
                j += 1
            if j < len(raw) and raw[j].lower() in HEADS:
                head = raw[j].lower()
                out += [w.lower() for w in raw[i + 1:j + 1]]
                if low in DET_BACK and head in by_head:
                    e = by_head[head]
                else:
                    e = ent(f'desc:{len(ents)}', ' '.join(raw[i:j + 1]), 'desc')
                    if e:
                        by_head[head] = e
                out.append(e or '<unk>')
                i = j + 1
                continue
        i += 1
    return {'tokens': out, 'ents': ents, 'options': options, 'question': text}
