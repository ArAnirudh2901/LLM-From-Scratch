"""Exact reading of formulaic speech questions: "Pointing to a man, Ravi said, 'His mother is the only daughter of my
father.' How is the man related to Ravi?". The utterance equates two possessive chains ("his mother", "the only
daughter of my father", "my father's son's wife") that end on one person; each step becomes a fact with an unnamed
person (#k), pronouns resolve to the person pointed at, "my/I" to the speaker, "your" to the one spoken to.
Returns None unless the whole question is understood (then the model reads it).

Works on prepare()'s tokens.
"""
from __future__ import annotations

import re

from .kin import KIN

E = r'@\d+'
HEADS = r'(?:man|woman|boy|girl|lady|gentleman|person|old man|old woman|young man|young woman|old lady|young lady)'
MALE, FEMALE = {'man', 'boy', 'gentleman', 'old man', 'young man'}, {'woman', 'girl', 'lady', 'old woman', 'young woman', 'old lady', 'young lady'}
SAY = r'(?:said|says|told|tells|replied|remarked|says to|said to|told to)'


class _Chain:
    def __init__(self):
        self.facts, self.n = [], 0

    def new(self):
        self.n += 1
        return f'#{self.n}'


def _rel(w):
    w = w.strip()
    only = w.startswith('only ')
    w = w[5:] if only else w
    w = re.sub(r'^(?:own|real|elder|younger|eldest|youngest) ', '', w)
    side = re.match(r'(paternal|maternal) ', w)
    base = w[side.end():] if side else w
    from .kin import SIDED
    if side and base not in SIDED:
        return None
    return (('only ' if only else '') + w) if base in KIN else None


def _np(words, who, ch, end=None):
    """Node for a noun phrase: 'my father 's son', 'the daughter of my grandfather 's only son', 'his mother', 'he'.
    who: {'my': speaker, 'his': target, ...}. end: the node the phrase must end on (identity), else a new one."""
    s = ' '.join(words).strip()
    s = re.sub(r'^(?:the|that|this) ', '', s)
    s = re.sub(rf'^(?:{HEADS}) (?={E}$)', '', s)   # "this lady @2"
    if s in who:
        node = who[s]
        return node if end is None or end == node else (ch.facts.append(('same', node, end)) or end)
    m = re.fullmatch(r'(?:the )?(?:younger|elder|youngest|eldest|older|oldest) of (?:the |his |her )?(two|three|four) (brothers?|sisters?|sons?|daughters?) of (.+)', s)
    if m:   # "the younger of the two brothers of X": one of X's two brothers (age is not modelled)
        base = _np(m.group(3).split(), who, ch)
        r = m.group(2).rstrip('s')
        if base is None:
            return None
        ch.facts.append(f"{base} has {({'two': 2, 'three': 3, 'four': 4})[m.group(1)]} {r}")
        return _step(base, r, ch, end)
    m = re.fullmatch(rf'(?:(?:{HEADS}) )?({E}) whose (.+?) is (.+)', s)   # "@2 whose father is my son"
    if m and _rel(m.group(2)) and not _rel(m.group(2)).startswith('only'):
        node = _step(m.group(1), _rel(m.group(2)), ch)
        if _np(m.group(3).split(), who, ch, end=node) is None:
            return None
        return m.group(1) if end is None or end == m.group(1) else (ch.facts.append(('same', m.group(1), end)) or end)
    m = re.fullmatch(rf'(?:(?:{HEADS}) )?({E})(?: ,)? who is (.+)', s)   # "the woman @2 who is the mother of ..."
    if m:
        if _np(m.group(2).split(), who, ch, end=m.group(1)) is None:
            return None
        return m.group(1) if end is None or end == m.group(1) else (ch.facts.append(('same', m.group(1), end)) or end)
    m = re.fullmatch(r'(.+?) of (.+)', s)            # "the A of B": B first, then A
    if m and _rel(re.sub(r'^(?:the|an?) ', '', m.group(1))):
        base = _np(m.group(2).split(), who, ch)
        if base is None:
            return None
        return _step(base, _rel(re.sub(r'^(?:the|an?) ', '', m.group(1))), ch, end)
    m = re.fullmatch(r"(.+) 's (.+)", s)              # "X 's A" (rightmost split: "my father 's son 's wife")
    if m and _rel(m.group(2)):
        base = _np(m.group(1).split(), who, ch)
        if base is None:
            return None
        return _step(base, _rel(m.group(2)), ch, end)
    m = re.fullmatch(r'(my|his|her|your|our) (.+)', s)
    if m and _rel(m.group(2)) and m.group(1) in who:
        return _step(who[m.group(1)], _rel(m.group(2)), ch, end)
    m = re.fullmatch(rf"({E}) 's (.+?) ({E})", s)   # "@1 's father @2": the relative is named
    if m and _rel(m.group(2)) and (end is None or end == m.group(3)):
        ch.facts.append(f'{m.group(1)} {_rel(m.group(2))} {m.group(3)}')
        return m.group(3)
    m = re.fullmatch(rf'(my|his|her|your|our) (.+?) ({E})', s)   # "my father-in-law @2": the relative is named
    if m and _rel(m.group(2)) and m.group(1) in who and (end is None or end == m.group(3)):
        ch.facts.append(f'{who[m.group(1)]} {_rel(m.group(2))} {m.group(3)}')
        return m.group(3)
    if re.fullmatch(E, s):
        return s if end is None or end == s else (ch.facts.append(('same', s, end)) or end)
    return None


def _step(base, rel, ch, end=None):
    if base is None:
        return None
    node = end or ch.new()
    ch.facts.append(f'{base} {rel} {node}')
    return node


HEAD_ = rf'(?:(?:a|an|the|that|this) )?(?:({HEADS}) )?'
OPEN = [   # (regex, groups): who is pointed at / introduced (target), who speaks, who is spoken to
    (rf'^(?:while |on )?(?:pointing (?:out )?(?:to|towards|at)|looking at|showing|introducing)(?: (.*?))?(?: ,)? {HEAD_}({E}) {SAY}'
     rf'(?: (?:to )?{HEAD_}({E}))?(?: ,)? ["\'] (.+?)(?: [.,])* ["\'] (?:, )?(.+)$', ('tdesc', 'sh', 's', 'ah', 'a')),
    (rf'^(?:if )?{HEAD_}({E}) (?:introduces|introduced|introducing) {HEAD_}({E})(?: to (?:his|her|my) \w+)?(?: ,)? (?:by saying|saying|and says|and said|says|said)(?: ,)? ["\'] (.+?)(?: [.,])* ["\'] (?:, )?(.+)$', ('sh', 's', 'th', 't')),
    (rf'^{HEAD_}({E}) (?:points|pointed|pointing) (?:to|at|towards) {HEAD_}({E}) (?:and|,) (?:says|said)(?: ,)? ["\'] (.+?)(?: [.,])* ["\'] (?:, )?(.+)$', ('sh', 's', 'th', 't')),
    (rf'^(?:if )?{HEAD_}({E}) {SAY}(?: (?:to )?{HEAD_}({E}))?(?: ,| :| -)? ["\'] (.+?)(?: [.,])* ["\'] (?:, )?(.+)$', ('sh', 's', 'ah', 'a')),
    (rf'^(?:pointing (?:out )?(?:to|towards|at)|looking at|introducing)(?: (.*?))? , {HEAD_}({E}) {SAY}(?: (?:to )?{HEAD_}({E}))? (?:that |, )((?:{E}|he|she|his|her|my|i|this|that) .+?) \. (.+)$', ('tdesc', 'sh', 's', 'ah', 'a')),
]
OPEN.append((rf'^(?:if )?{HEAD_}({E}) {SAY}(?: (?:to )?{HEAD_}({E}))? that (.+?) \. (.+)$', ('sh', 's', 'ah', 'a')))   # reported speech
NARRATED = rf'^(?:if )?{HEAD_}({E}) (?:introduces|introduced) {HEAD_}({E}) as (?:he is |she is )?(.+?) \. (.+)$'   # his/her: the speaker
EXTRA = [   # sentences after the speech: "if X has no siblings", "X is the only child"
    (rf'(?:if )?({E}) (?:has no siblings?|has no brothers? or sisters?|is the only child(?: of (?:his|her) parents)?|is an only child)', 'no-sibling'),
    (rf'(?:if )?({E}) has no brothers?', 'no-brother'),
    (rf'(?:if )?({E}) has no sisters?', 'no-sister'),
]


def _subject(s, genders):
    """'the woman @1 at the counter' -> '@1' (gender noted); '@1 's father' and plain phrases unchanged."""
    m = re.fullmatch(rf'(?:(?:the|this|that) )?(?:({HEADS}) )?({E})(.*)', s)
    if not m:
        return s
    head, e, rest = m.groups()
    if head:
        genders[e] = 'male' if head in MALE else 'female' if head in FEMALE else None
    if rest.startswith(" 's "):
        return e + rest
    return e


def _question(q, ids):
    q = q.strip()
    q = re.sub(rf'\b(?:the|that|this) (?:{HEADS}) (?:in|on) the (?:photograph|photo|picture|portrait|stage)\b(?! ?@)', 'he', q)
    d = r"(?: (?:(?!'s\b|of\b)[^?])*?)?"   # "playing football", "in the photograph"; never "'s husband"
    m = (re.fullmatch(rf'(?:then(?: ,)? )?how is {HEAD_}({E}|he|she|the person|this person){d} (?:related |connected )?(?:to|with) {HEAD_}({E}){d} \?', q)
         or re.fullmatch(rf'who is {HEAD_}({E}) (?:to|of) {HEAD_}({E}){d} \?', q)
         or re.fullmatch(rf'(?:then(?: ,)? )?how {HEAD_}({E}|he|she){d} is (?:related |connected )(?:to|with) {HEAD_}({E}){d} \?', q)
         or re.fullmatch(rf'(?:then(?: ,)? )?how is {HEAD_}({E}|he|she){d} (?:related |connected )?(?:to|with) ()(#\d+) \?', q)
         or re.fullmatch(rf"what is {HEAD_}({E}) 's relation(?:ship)? (?:to|with) {HEAD_}({E}){d} \?", q))
    if not m:
        mw = re.fullmatch(rf'(?:at )?whose (?:photograph|photo|picture|portrait) (?:was it|is it|is this|was this|was {HEAD_}({E}) looking at|is {HEAD_}({E}) looking at|was {HEAD_}({E}) looking) \?', q)
        if mw and ids.get('he') and ids.get('me'):   # "Whose photograph was it?": the person pictured, as related to the speaker
            return ids['he'], ids['me'], None, None
        return None
    ha, a, hb, b = m.groups()
    a = ids.get('he') if a in ('he', 'she', 'the person', 'this person') else a
    return a, b, ha, hb


def read(tokens, why=None) -> str | None:
    """why: a list that receives the stage that failed (diagnostics)."""
    fail = lambda stage, x: (why.append(f'{stage}: {x}') if why is not None else None)
    text = ' '.join(tokens)
    if re.search(r'\b(?:means|denotes|implies)\b', text):
        return None
    text = re.sub(r'^(?:in this type of questions?\b.*? \. |(?:directions?|read the following[^.]*?|answer the questions?[^.]*?) [.:] )+', '', text)
    narrated = False
    for pat, names in OPEN:
        m = re.match(pat, text)
        if m:
            break
    else:
        m = re.match(NARRATED, text)
        if not m:
            return fail('opening', text[:100])
        names, narrated = ('sh', 's', 'th', 't'), True
    g = dict(zip(names, m.groups()[:len(names)]))
    utter, rest = m.groups()[len(names):]
    if narrated:   # "X introduces Y as the son of the brother of her mother": he is ..., her = X
        utter = 'he is ' + utter
    if 'tdesc' in g:   # "a man @0 on the stage", "a photograph", "@0", "a photograph of a girl @0"
        d = g.pop('tdesc') or ''
        if re.search(r'\b(?:said|says|told|tells)\b', d):
            return fail('opening', text[:100])
        mt = re.search(rf'(?:({HEADS}) )?({E})', d) or re.search(rf'(?:({HEADS}) )({E}) (?:in|on) the (?:photograph|photo|picture|portrait|stage)', rest)
        g['th'], g['t'] = (mt.group(1), mt.group(2)) if mt else ((re.search(rf'\b({HEADS})\b', d) or [None, None])[1], None)
        if not mt and not re.search(r'\b(?:photo|photograph|picture|portrait|man|woman|boy|girl|lady|gentleman|person)\b', d):
            return fail('opening', text[:100])
        names = ('th',) + names[1:]
    target, speaker, addressee = g.get('t'), g['s'], g.get('a')
    ch = _Chain()
    genders = {}
    pointed = target is not None or names[0] == 'th'
    utter = re.sub(r'\bi am\b', 'i is', utter)
    if re.search(rf'\b(?:that|this) (?:{HEADS}) {speaker}\b', utter):   # "a lady @0 ... 'I am the daughter of this lady @0'": not the speaker
        if g.get('t') is not None or not pointed:
            return fail('speaker named in the speech', utter)
        target = ch.new()
        utter = re.sub(rf'\b(?:that|this) (?:{HEADS}) {speaker}\b', 'this', utter)
    if target is None and pointed:   # "pointing to a photograph": the person in it, named in the speech if it says "that man"
        mt2 = re.search(rf'\b(?:that|this) (?:{HEADS}) ({E})', utter)
        target = mt2.group(1) if mt2 else ch.new()
    for k, e in (('th', target), ('sh', speaker), ('ah', addressee)):
        if g.get(k) and e:
            genders[e] = 'male' if g[k] in MALE else 'female' if g[k] in FEMALE else None
    who = {'my': speaker, 'i': speaker, 'me': speaker}
    if narrated:
        who.update({'he': target, 'she': target, 'his': speaker, 'her': speaker})
        for p_, gg in (('his', 'male'), ('her', 'female')):   # the narrator's pronoun gives the speaker's gender
            if re.search(rf'\b{p_}\b', utter):
                if genders.get(speaker) not in (None, gg):
                    return None
                genders[speaker] = gg
    elif target:
        who.update({'he': target, 'she': target, 'his': target, 'her': target, 'him': target, 'this': target})
    elif re.search(r'\b(?:he|she|his|her|him)\b', utter):   # nobody pointed at: whose pronoun?
        return fail('pronoun without a pointed person', utter)
    if addressee:
        who.update({'your': addressee, 'you': addressee})
    if target and not narrated:
        for p, gg in (('he', 'male'), ('his', 'male'), ('him', 'male'), ('she', 'female'), ('her', 'female')):
            if re.search(rf'\b{p}\b', utter):
                if genders.get(target) not in (None, gg):
                    return None
                genders[target] = gg
    if re.search(r"\bmy (?:wife|wife 's)\b", utter):
        genders[speaker] = 'male'
    if re.search(r"\bmy (?:husband|husband 's)\b", utter):
        genders[speaker] = 'female'
    extra = []
    parts = [p.strip() for p in re.split(r' \. | , and | and (?=i have)|(?: ,)? but | and (?=(?:his|her|my|your) \w+ is )', utter) if p.strip()]
    mains = []
    for p in parts:
        if re.fullmatch(r'i have no (?:brother or sister|brothers or sisters|siblings|sibling)|brothers and sisters i have none|i have no brothers? and no sisters?|i (?:am|is) the only child(?: of my parents)?', p):
            extra.append(f'{speaker} no-sibling')
        elif re.fullmatch(r'i have no brothers?', p):
            extra.append(f'{speaker} no-brother')
        else:
            mains.append(p)
    if not mains or len(mains) > 2:
        return fail('no statement' if not mains else 'speech has three statements', utter)
    for main in mains:   # "I am the only daughter of this lady and her son is your maternal uncle"
        mm = re.fullmatch(r'(.+?) is (.+)', main)
        if not mm:
            return fail('statement without "is"', main)
        left = _np(_subject(mm.group(1), genders).split(), who, ch)
        if left is None or _np(mm.group(2).split(), who, ch, end=left) is None:
            return fail('description', main)
    if any(isinstance(f, tuple) for f in ch.facts):   # identities between two named nodes: not handled
        return fail('identity of two named people', main)
    # after the speech: extra facts, then the question ("if X has no siblings , how is ...")
    from .rules import clause
    sents = [x.strip() for x in re.split(r' \. | ; ', ' ' + rest.strip() + ' ') if x.strip()]
    if not sents:
        return None
    q = sents[-1]
    mq = re.fullmatch(r'(if [^,]+) , (.+)', q)
    if mq:
        sents[-1:] = [mq.group(1), mq.group(2)]
        q = mq.group(2)
    mq = re.fullmatch(r'(.+?)(?: ,)? if (.+?) \?', q)   # "how is A related to B if C is the sister of D ?"
    if mq:
        sents[-1:] = [mq.group(2), mq.group(1) + ' ?']
        q = mq.group(1) + ' ?'
    for x in sents[:-1]:
        if re.fullmatch(r'(?:brothers and sisters i have none|i have no (?:brothers? or sisters?|siblings?))', x):
            extra.append(f'{speaker} no-sibling')
            continue
        got = next((f'{mm2.group(1)} {kind}' for pat, kind in EXTRA for mm2 in [re.fullmatch(pat, x)] if mm2), None)
        cl = None if got else clause(re.sub(r'^if ', '', x))
        if got:
            extra.append(got)
        elif cl:
            extra += cl
        else:
            return fail('sentence after the speech', x)
    q = re.sub(r'\bthe speaker\b', speaker, q)
    q = re.sub(r'\bwomen\b', 'woman', q)
    for h, e in ((g.get('sh'), speaker), (g.get('th'), target)):   # "how is girl related to X": the speaker introduced as "a girl"
        if h and e and h != (g.get('th') if e == speaker else g.get('sh')):
            q = re.sub(rf"\b(?:the |that |this )?{h}\b(?! (?:@|#)\d)", e, q)
    q = re.sub(rf"^({E}|he|she) is {HEAD_}({E}) 's \s*[?:]$", r'how is \1 related to \3 ?', q).replace('  ', ' ')   # "the boy @1 is girl 's ?"
    mc = re.fullmatch(rf"(.+ (?:to|with)) ({E}) 's ([a-z-]+) \?", q)   # "how is the speaker related to @1 's father ?"
    if mc and _rel(mc.group(3)) and not _rel(mc.group(3)).startswith('only'):
        k = ch.new()
        extra.append(f'{mc.group(2)} {_rel(mc.group(3))} {k}')
        q = f'{mc.group(1)} {k} ?'
    qq = _question(q, {'he': target, 'me': speaker})
    if not qq:
        return fail('question', q)
    a, b, ha, hb = qq
    facts = list(ch.facts) + extra
    named = set(re.findall(r'[@#]\d+', ' '.join(facts))) | {x for x in (target, speaker, addressee) if x}
    if a is None or a == b or a not in named or b not in named:
        return None
    for e, h in ((a, ha), (b, hb)):
        if h and e.startswith('@'):
            gg = 'male' if h in MALE else 'female' if h in FEMALE else None
            if gg and genders.get(e) not in (None, gg):
                return None
            genders[e] = gg or genders.get(e)
    facts += [f'{p} {gg}' for p, gg in genders.items() if gg and p.startswith('@')]
    return ' ; '.join(facts + [f'ask {b} {a}'])
