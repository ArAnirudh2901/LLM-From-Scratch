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


def read(tokens) -> str | None:
    text = ' '.join(tokens)
    if re.search(r'\b(?:means|denotes|implies)\b', text):
        return None
    m = re.match(rf'^(?:(?:while |on )?(?:pointing (?:to|towards|at)|looking at|showing|introducing|pointing out to) )'
                 rf'(?:(?:a|an|the) )?(?:(?:photograph|photo|picture|portrait) (?:of )?)?(?:(?:a|an|the) )?(?:({HEADS}) )?({E})?(?: in (?:the |a )?(?:photograph|photo|picture))?'
                 rf' , (?:(?:a|an|the) )?(?:({HEADS}) )?({E}) {SAY}(?: (?:to )?(?:(?:a|an|the) )?(?:{HEADS} )?({E}))? ,? " (.+?) \.? " (.+)$', text)
    if not m:
        return None
    t_head, target, s_head, speaker, addressee, utter, question = m.groups()
    ch = _Chain()
    genders = {}
    if target is None:   # "pointing to a photograph": the person in it
        target = ch.new()
    if t_head:
        genders[target] = 'male' if t_head in MALE else 'female' if t_head in FEMALE else None
    if s_head:
        genders[speaker] = 'male' if s_head in MALE else 'female' if s_head in FEMALE else None
    who = {'my': speaker, 'i': speaker, 'me': speaker, 'he': target, 'she': target, 'his': target, 'her': target,
           'him': target, 'this': target}
    if addressee:
        who.update({'your': addressee, 'you': addressee})
    for p, g in (('he', 'male'), ('his', 'male'), ('him', 'male'), ('she', 'female'), ('her', 'female')):
        if re.search(rf'\b{p}\b', utter):
            if genders.get(target) not in (None, g):
                return None
            genders[target] = g
    if re.search(r"\bmy (?:wife|wife 's)\b", utter):
        genders[speaker] = 'male'
    if re.search(r"\bmy (?:husband|husband 's)\b", utter):
        genders[speaker] = 'female'
    extra = []
    parts = [p.strip() for p in re.split(r' \. | , and | and (?=i have)', utter) if p.strip()]
    main = None
    for p in parts:
        if re.fullmatch(r'i have no (?:brother or sister|brothers or sisters|siblings|sibling)|brothers and sisters i have none|i have no brothers? and no sisters?', p):
            extra.append(f'{speaker} no-sibling')
        elif re.fullmatch(r'i have no brothers?', p):
            extra.append(f'{speaker} no-brother')
        elif main is None:
            main = p
        else:
            return None
    if main is None:
        return None
    mm = re.fullmatch(r'(.+?) is (.+)', main)
    if not mm:
        return None
    left = _np(mm.group(1).split(), who, ch)
    if left is None or _np(mm.group(2).split(), who, ch, end=left) is None:
        return None
    if any(isinstance(f, tuple) for f in ch.facts):   # identities between two named nodes: not handled
        return None
    q = re.fullmatch(rf'how is (?:the )?(?:{HEADS} )?({E}|he|she|the person|this person)(?: in the (?:photograph|photo|picture))? (?:related|connected) (?:to|with) (?:the )?(?:{HEADS} )?({E})(?: in the (?:photograph|photo|picture))? \?', question.strip())
    if not q:
        return None
    a, b = q.groups()
    a = target if a in ('he', 'she', 'the person', 'this person') else a
    if a not in (target, speaker, addressee) or b not in (target, speaker, addressee) or a == b:
        return None
    facts = list(ch.facts) + extra + [f'{p} {g}' for p, g in genders.items() if g and p.startswith('@')]
    return ' ; '.join(facts + [f'ask {b} {a}'])
