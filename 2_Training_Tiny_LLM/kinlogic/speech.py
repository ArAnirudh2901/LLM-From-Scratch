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
    return (('only ' if only else '') + w) if w in KIN else None


def _np(words, who, ch, end=None):
    """Node for a noun phrase: 'my father 's son', 'the daughter of my grandfather 's only son', 'his mother', 'he'.
    who: {'my': speaker, 'his': target, ...}. end: the node the phrase must end on (identity), else a new one."""
    s = ' '.join(words).strip()
    s = re.sub(r'^(?:the|that|this) ', '', s)
    if s in who:
        node = who[s]
        return node if end is None or end == node else (ch.facts.append(('same', node, end)) or end)
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
     rf'(?: (?:to )?{HEAD_}({E}))?(?: ,)? " (.+?)(?: [.,])* " (?:, )?(.+)$', ('tdesc', 'sh', 's', 'ah', 'a')),
    (rf'^(?:if )?{HEAD_}({E}) (?:introduces|introduced|introducing) {HEAD_}({E})(?: to (?:his|her|my) \w+)?(?: ,)? (?:by saying|saying|and says|and said|says|said)(?: ,)? " (.+?)(?: [.,])* " (?:, )?(.+)$', ('sh', 's', 'th', 't')),
    (rf'^{HEAD_}({E}) (?:points|pointed|pointing) (?:to|at|towards) {HEAD_}({E}) (?:and|,) (?:says|said)(?: ,)? " (.+?)(?: [.,])* " (?:, )?(.+)$', ('sh', 's', 'th', 't')),
    (rf'^(?:if )?{HEAD_}({E}) {SAY}(?: (?:to )?{HEAD_}({E}))?(?: ,)? " (.+?)(?: [.,])* " (?:, )?(.+)$', ('sh', 's', 'ah', 'a')),
]
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
    m = (re.fullmatch(rf'(?:then ,? )?how is {HEAD_}({E}|he|she|the person|this person){d} (?:related |connected )?(?:to|with) {HEAD_}({E}){d} \?', q)
         or re.fullmatch(rf'who is {HEAD_}({E}) (?:to|of) {HEAD_}({E}){d} \?', q))
    if not m:
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
    for pat, names in OPEN:
        m = re.match(pat, text)
        if m:
            break
    else:
        return fail('opening', text[:100])
    g = dict(zip(names, m.groups()[:len(names)]))
    utter, rest = m.groups()[len(names):]
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
    if target is None and pointed:   # "pointing to a photograph": the person in it
        target = ch.new()
    for k, e in (('th', target), ('sh', speaker), ('ah', addressee)):
        if g.get(k) and e:
            genders[e] = 'male' if g[k] in MALE else 'female' if g[k] in FEMALE else None
    who = {'my': speaker, 'i': speaker, 'me': speaker}
    if target:
        who.update({'he': target, 'she': target, 'his': target, 'her': target, 'him': target, 'this': target})
    elif re.search(r'\b(?:he|she|his|her|him)\b', utter):   # nobody pointed at: whose pronoun?
        return fail('pronoun without a pointed person', utter)
    if addressee:
        who.update({'your': addressee, 'you': addressee})
    if target:
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
    parts = [p.strip() for p in re.split(r' \. | , and | and (?=i have)', utter) if p.strip()]
    main = None
    for p in parts:
        if re.fullmatch(r'i have no (?:brother or sister|brothers or sisters|siblings|sibling)|brothers and sisters i have none|i have no brothers? and no sisters?|i am the only child(?: of my parents)?', p):
            extra.append(f'{speaker} no-sibling')
        elif re.fullmatch(r'i have no brothers?', p):
            extra.append(f'{speaker} no-brother')
        elif main is None:
            main = p
        else:
            return fail('speech has two statements', utter)
    if main is None:
        return fail('no statement', utter)
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
    for x in sents[:-1]:
        got = next((f'{mm2.group(1)} {kind}' for pat, kind in EXTRA for mm2 in [re.fullmatch(pat, x)] if mm2), None)
        cl = None if got else clause(re.sub(r'^if ', '', x))
        if got:
            extra.append(got)
        elif cl:
            extra += cl
        else:
            return fail('sentence after the speech', x)
    qq = _question(q, {'he': target})
    if not qq:
        return fail('question', q)
    a, b, ha, hb = qq
    facts = list(ch.facts) + extra
    named = set(re.findall(E, ' '.join(facts))) | {x for x in (target, speaker, addressee) if x}
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
