"""The worked solution for one answer, from the facts and the family the engine chose:

    [TARGET]: Sanghoti (female: "her")
    [REFERENCE]: Dinabandhu (male: assumed from the name)
    [CONTEXT MAP]:
    - Sanghoti's father = A
    - Dinabandhu's father = B
    - the only son of B = A
    [IDENTITY]: A is Dinabandhu (Dinabandhu is a man and the only son of his father)
    [EQUATION]: Father(Sanghoti) = Dinabandhu
    [INFERENCE]: Sanghoti is Dinabandhu's daughter.
    [ANSWER]: Daughter
"""
from __future__ import annotations

from .kin import KIN, NONKIN

GENDER = {'M': 'male', 'F': 'female'}


def explain(stmts: list[dict], result: dict, q_index: int = 0, names: dict | None = None, priors: dict | None = None,
            options=None) -> dict:
    """Steps for question q_index of solve()'s result. names: {placeholder: text}; options: [(label, text)].
    Returns {'lines': [...], 'text': str, 'answer': str, 'option': label or None}."""
    names, priors = names or {}, priors or {}
    q = result['questions'][q_index]
    ask = q['q']
    if not q['answers']:
        msg = 'The statements contradict each other.' if q['status'] == 'contradiction' else 'No answer.'
        return {'lines': [msg], 'text': msg, 'answer': msg, 'option': None}
    best = q['answers'][0]
    w, ids = best['world'], result['ids']
    anon = [e for e in ids if e.startswith('#')]
    letter = {e: f'Person {i + 1}' for i, e in enumerate(anon)}

    def show(e):
        n = names.get(e)
        if n and n.split()[0].lower() in ('a', 'an', 'the', 'this', 'that', 'another'):
            return 'the ' + ' '.join(n.split()[1:])   # "A woman" -> "the woman"
        return n or letter.get(e, e)
    lines = []

    # who the question is about, and how their gender is known
    def gender_note(e):
        g = w.g(ids[e])
        if not g:
            return 'gender not given'
        if any(s['t'] == 'gender' and s['x'] == e for s in stmts):
            return f'{GENDER[g]}: stated'
        if any(s['t'] == 'rel' and s['y'] == e and not s.get('neg') and KIN.get(s['r']) and all(p[-1].g for p in KIN[s['r']]) for s in stmts):
            return f'{GENDER[g]}: from a relation word'
        if priors.get(e) == g:
            return f'{GENDER[g]}: assumed from the name'
        return f'{GENDER[g]}: worked out'

    if ask['t'] == 'ask':
        lines.append(f"[TARGET]: {show(ask['y'])} ({gender_note(ask['y'])})")
        lines.append(f"[REFERENCE]: {show(ask['x'])} ({gender_note(ask['x'])})")

    # each unnamed person as the text describes them
    desc = {}
    for s in stmts:
        if s['t'] == 'rel' and not s.get('neg') and s['y'] in letter and s['y'] not in desc:
            desc[s['y']] = (s['x'], s['r'], s.get('only'), s.get('side'))
    def phrase(e):
        x, r, only, side = desc[e]
        rel = ' '.join(t for t in ('only' if only else None, side, r) if t)
        return f"the {rel} of {show(x)}" if only else f"{show(x)}'s {rel}"
    cmap = []
    for s in stmts:
        if s['t'] != 'rel' or s.get('neg') or s['r'] in NONKIN:
            continue
        if s['y'] in letter and desc.get(s['y'])[:2] == (s['x'], s['r']):
            cmap.append(f"- {phrase(s['y'])} = {show(s['y'])}")
        elif s['x'] in letter or s['y'] in letter:
            rel = ' '.join(t for t in ('only' if s.get('only') else None, s.get('side'), s['r']) if t)
            cmap.append(f"- the {rel} of {show(s['x'])} = {show(s['y'])}")
    if cmap:
        lines.append('[CONTEXT MAP]:')
        lines += cmap

    # unnamed people the family shows to be someone named
    same = []
    for e in anon:
        root = w.find(ids[e])
        named = [n for n in ids if n.startswith('@') and w.find(ids[n]) == root]
        if named:
            same.append((e, named[0]))
    for e, n in same:
        why = []
        if any(s['t'] == 'attr' and s['a'] in ('no-sibling', 'no-brother', 'no-sister') and s['x'] == n for s in stmts):
            why.append(f'{show(n)} has no brother or sister')
        if any(s.get('only') and s['y'] == e for s in stmts):
            g = w.g(ids[n])
            only = next(s for s in stmts if s.get('only') and s['y'] == e)
            named_g = priors.get(n) == g and not any(s['t'] == 'gender' and s['x'] == n for s in stmts)
            why.append(f"{show(n)} is {'male' if g == 'M' else 'female'}{' (from the name)' if named_g else ''}, so the only "
                       f"{only['r']} of {show(only['x'])} is {show(n)} {'himself' if g == 'M' else 'herself'}")
        lines.append(f"[IDENTITY]: {show(e)} is {show(n)}" + (f" ({'; '.join(why)})" if why else ''))
        if e in desc:
            x, r, _, _ = desc[e]
            lines.append(f"[EQUATION]: {r.capitalize()}({show(x)}) = {show(n)}")

    # the answer
    if ask['t'] == 'ask':
        x, y = show(ask['x']), show(ask['y'])
        if best['key'] == 'none':
            ans = f'No family link between {y} and {x} is given.'
        elif best.get('nonkin'):
            ans = f"{y} is {x}'s {best['nonkin']}, not a family member."
        else:
            path = best.get('desc')
            word = best['label']
            ans = f"{y[:1].upper() + y[1:]} is {x}'s {word}" + (f" ({path})" if path and path != best.get('base') else '') + '.'
            if best.get('hindi'):
                ans = ans[:-1] + f"; in Hindi, {best['hindi']}."
        lines.append(f'[INFERENCE]: {ans}')
        answer = best['label'] if best['key'] not in ('none',) else 'No relation given'
    elif ask['t'] == 'who':
        who = [show(n) for n in best.get('names', [])]
        answer = ', '.join(who) or 'not named in the question'
        lines.append(f"[INFERENCE]: {show(ask['x'])}'s {ask['r']} is {answer}.")
    else:
        answer = best['key']
        lines.append(f'[INFERENCE]: {answer}.')
    if q['status'] in ('conditional', 'ambiguous', 'assumed') and len(q['answers']) > 1:
        alt = q['answers'][1]
        import re
        conds = [re.sub(r'@\d+', lambda m: show(m.group(0)), c) for c in alt['conds']]
        cond = f" if {', '.join(conds)}" if conds else ''
        lines.append(f"[NOTE]: Other reading{cond}: {alt.get('label', alt['key'])}.")
    if q['status'] == 'gender':
        answer = q['neutral']
        lines.append(f"[NOTE]: The gender is not given, so the answer is {answer}.")
    option = match_option(answer, options) if options else None
    lines.append(f"[ANSWER]: {f'({option}) ' if option else ''}{answer[:1].upper() + answer[1:]}")
    return {'lines': lines, 'text': '\n'.join(lines), 'answer': answer, 'option': option}


def match_option(answer: str, options) -> str | None:
    """The option whose text names the answer ("Maternal cousin" for "maternal cousin", "Cousin" also fits)."""
    import re
    norm = lambda t: ' '.join(t.lower().replace('-', ' ').split())
    opt = lambda t: re.sub(r"(?:\s*'s|')$", '', re.sub(r'^(?:his|her|my|your|their|the) ', '', norm(t).rstrip('.')))   # "His son's"
    a = norm(answer)
    exact = [lab for lab, text in options if norm(text) == a or opt(text) == a]
    if exact:
        return exact[0]
    base = a.replace('paternal ', '').replace('maternal ', '').replace(' (by marriage)', '')
    loose = [lab for lab, text in options if opt(text) == base or norm(text) == base or norm(text).endswith(' ' + base)]
    return loose[0] if len(loose) == 1 else None
