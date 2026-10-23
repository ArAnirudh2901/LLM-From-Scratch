"""Answers a whole exam question: text in, chosen option and worked solution out.

    answer(text, read) -> {'answer', 'option', 'status', 'explanation', 'facts'}

read(list of token lists) -> list of fact texts is the reader model (or any function standing in for it). Handles
relation, who, how-many, gender and pair questions, statements to check ("which is definitely true"), data
sufficiency, and coded "which expression shows..." questions (each option is read on its own).
"""
from __future__ import annotations

import os
import re

from .engine import alternatives, claim_holds, pair_holds, solve
from .explain import explain, match_option
from .facts import parse
from .kin import paths_of
from .names import name_priors
from .prepare import prepare

CANT = re.compile(r"cannot be determined|can.?t be determined|data (?:is )?inadequate|insufficient|not possible to determine|"
                  r"cannot be established|can.?t say", re.I)
NONE = re.compile(r'none of (?:these|the above)', re.I)
NUM_WORDS = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve'.split())}
SYN = {'dad': 'father', 'mom': 'mother', 'mum': 'mother', 'grand father': 'grandfather', 'grand mother': 'grandmother',
       'grand son': 'grandson', 'grand daughter': 'granddaughter', 'in law': 'in-law', 'cousin brother': 'cousin',
       'cousin sister': 'cousin', 'great grand': 'great-grand', 'grand nephew': 'grand-nephew', 'grand niece': 'grand-niece'}


# ---------- options ----------

def option_moves(text: str):
    """A relation option as steps: "Sister of father-in-law" / "wife's father's sister" -> [(move, gender)];
    None when it is not a chain of relation words."""
    t = ' ' + text.lower().strip(' .') + ' '
    for a, b in SYN.items():
        t = t.replace(f' {a} ', f' {b} ').replace(f' {a}-', f' {b}-').replace(f'{a} ', f'{b} ')
    t = re.sub(r"\b(his|her|my|the|a|an|their)\b", ' ', t)
    parts = [p.strip() for p in t.split(' of ') if p.strip()][::-1]   # "x of y" -> y then x
    words = []
    for part in parts:
        for w in part.split("'s"):
            w = w.strip()
            if w:
                words.append(w)
    steps = []
    for w in words:
        toks = w.split()
        side = toks[0] if toks and toks[0] in ('paternal', 'maternal') else None
        word = '-'.join(toks[1:] if side else toks).replace('--', '-')
        paths = paths_of(word, side)
        if not paths:
            return None
        for st in paths[0]:
            steps += [('U', None), ('D', st.g)] if st.m == 'B' else [(st.m, st.g)]
    return steps or None


def _same_path(moves, ans_moves) -> bool:
    if len(moves) != len(ans_moves):
        return False
    return all(m == am and (g is None or g == ag) for (m, g), (am, ag) in zip(moves, ans_moves))


def _side_ok(a, text, status):
    """An option naming the other side ("Maternal uncle" for a paternal one) does not match a settled side."""
    t = text.lower()
    side = a.get('side')
    return not (side and status != 'side' and ('paternal' in t or 'maternal' in t) and side not in t)


def _relation_option(cands, options, status):
    """First answer (in order of preference) that some option names, by word or by spelled-out path."""
    if status == 'assumed' or (cands and cands[0].get('assumed')):   # only a name suggests the gender: an option leaving it open is the setter's intent
        a0 = next((a for a in cands if a.get('key') not in (None, 'none')), None)
        neutral = [(m, None) for m, _, _ in (a0 or {}).get('moves') or []]
        for lab, text in options if neutral else []:
            alts = [option_moves(x) for x in re.split(r'\s+or\s+|/', text)]
            if len(alts) > 1 and all(alts) and all(_same_path([(m, None) for m, _ in x], neutral) for x in alts):
                return a0, lab
        if a0 and _labelled(options, CANT) and not NAMES_DECIDE:
            return a0, _labelled(options, CANT)
    real = [a for a in cands if a.get('key') not in (None, 'none') and a.get('moves')]
    if status in ('conditional', 'ambiguous', 'gender', 'side') and len(real) > 1:   # "Grandfather or Grandmother" names every reading
        for lab, text in options:
            alts = [option_moves(x) for x in re.split(r'\s+or\s+|/', text)]
            if len(alts) > 1 and all(alts):
                paths = [[(m, g) for m, g, _ in a['moves']] for a in real]
                if all(any(_same_path(x, pth) for x in alts) for pth in paths) and all(any(_same_path(x, pth) for pth in paths) for x in alts):
                    return real[0], lab
    for a in cands:
        if a.get('key') in (None, 'none'):
            continue
        fit = [(l, t) for l, t in options if _side_ok(a, t, status)]
        lab = match_option(a.get('label') or a['key'], fit) or match_option(a.get('base') or a['key'], fit)
        if lab:
            return a, lab
        am = [(m, g) for m, g, _ in a.get('moves') or []]
        for lab, text in options:
            for alt in re.split(r'\s+or\s+|/', text):
                mv = option_moves(alt)
                if mv and _same_path(mv, am):
                    return a, lab
            alts = [option_moves(x) for x in re.split(r'\s+or\s+|/', text)]
            if status == 'gender' and len(alts) > 1 and all(alts):   # "Grandson or granddaughter"
                neutral = [(m, None) for m, _ in am]
                if all(_same_path([(m, None) for m, _ in x], neutral) for x in alts):
                    return a, lab
    if not _labelled(options, CANT):   # gender open, one option fits one of its readings ("Sister" for a sibling)
        for a in cands:
            if a.get('key') in (None, 'none') or not a.get('moves'):
                continue
            neutral = [(m, None) for m, _, _ in a['moves']]
            fit = {lab for lab, text in options for alt in re.split(r'\s+or\s+|/', text)
                   if (mv := option_moves(alt)) and _same_path([(m, None) for m, _ in mv], neutral)}
            if len(fit) == 1:
                return a, fit.pop()
    return None, None


def _labelled(options, regex):
    return next((lab for lab, t in options if regex.search(t)), None)


def _names_in(text, ents):
    """Placeholders an option names ("A and C", "P, Q")."""
    byname = {e['text'].lower(): e['id'] for e in ents}
    toks = re.findall(r"[A-Za-z][\w-]*", text)
    found = [byname[t.lower()] for t in toks if t.lower() in byname]
    return found


def _number(text):
    m = re.search(r'\d+', text)
    if m:
        return int(m.group(0))
    for w, n in NUM_WORDS.items():
        if re.search(rf'\b{w}\b', text.lower()):
            return n
    return None


ROMAN = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6}


def _roman_set(text):
    t = text.lower()
    if re.search(r'\ball\b', t):
        return 'all'
    if re.search(r'\bnone\b|\bneither\b', t):
        return set()
    return {ROMAN[r] for r in re.findall(r'\b(i{1,3}|iv|vi?)\b', t) if r in ROMAN} or {int(n) for n in re.findall(r'\b\d\b', t)}


# ---------- the answer ----------

NAMES_DECIDE = os.environ.get('KIN_NAMES_DECIDE', '1') == '1'   # a name's gender is taken as given (real exam keys: +8 of 576 dev questions vs preferring 'cannot be determined')
SMALLEST = os.environ.get('KIN_SMALLEST', '0') == '1'   # smallest family for every spoken description (dev +1.1, locked test -0.4 with r3); off: exact speech readings only


def answer(text: str, read, priors: dict | None = None) -> dict:
    """Never raises: a failure comes back as status 'error' with the reason."""
    try:
        return _answer(text, read, priors)
    except Exception as e:   # one odd question must not stop a batch or the page
        return {'answer': 'error', 'option': None, 'status': 'error', 'explanation': f'{type(e).__name__}: {e}', 'facts': ''}


def _answer(text: str, read, priors: dict | None = None) -> dict:
    """read may return one reading per question or several, best first, optionally with scores (mean log-probability
    per token). Every reading is solved; readings that fail (contradiction, no question) drop out; the rest vote for
    their option, weighted by the reader's confidence. confidence: the winning option's share of the votes."""
    import math
    from .codes import with_codes
    p = prepare(text)
    read = with_codes(read, p)   # coded questions are read exactly; the model reads the rest
    got = read([p['tokens']])[0]
    readings = [(got, 0.0)] if isinstance(got, str) else [(g, 0.0) if isinstance(g, str) else (g[0], g[1]) for g in got]
    results = []
    for k, (lf, score) in enumerate(readings):
        res = _answer_one(p, lf, read, priors)
        res['reading'] = k
        results.append((res, score))
    good = [(r, sc) for r, sc in results if r['status'] not in ('error', 'contradiction', 'none')]
    if not good:
        return {**results[0][0], 'confidence': 0.0}
    top = max(sc for _, sc in good)
    weights = [math.exp(8 * (sc - top)) for _, sc in good]   # sharper than the raw per-token average
    votes = {}
    for (r, _), wt in zip(good, weights):
        key = r['option'] if p['options'] else r['answer']
        if key is not None:
            votes[key] = votes.get(key, 0.0) + wt
    if not votes:
        return {**good[0][0], 'confidence': 0.0}
    best = max(votes, key=lambda k: (votes[k], -next(i for i, (r, _) in enumerate(good) if (r['option'] if p['options'] else r['answer']) == k)))
    winner = next(r for r, _ in good if (r['option'] if p['options'] else r['answer']) == best)
    return {**winner, 'confidence': round(votes[best] / sum(weights), 3)}


def _people(tokens, stmts):
    """Who belongs to a closed family: the placeholders of the member list (the first sentence naming three or more)
    and everyone the facts mention; a stray capitalized word elsewhere is not a member."""
    if not any(s['t'] == 'members' for s in stmts):
        return []
    out, sent = [], []
    for t in tokens + ['.']:
        if t in ('.', '?', '!'):
            if len(set(sent)) >= 3:
                out = list(dict.fromkeys(sent))
                break
            sent = []
        elif t[0] == '@':
            sent.append(t)
    for s in stmts:
        for k in ('x', 'y'):
            if s.get(k, '').startswith('@') and s[k] not in out:
                out.append(s[k])
    return out


def _answer_one(p, lf, read, priors):
    ents, options = p['ents'], p['options']
    priors = {**name_priors(ents), **(priors or {})}
    names = {e['id']: e['text'] for e in ents}
    stmts, _ = parse(lf)
    people = _people(p['tokens'], stmts)
    out = {'facts': lf, 'options': options}
    low = p['question'].lower()

    expr_opts = options and sum(bool(re.search(r'[A-Za-z]\s*[^\w\s,.()\'"]{1,2}\s*[A-Za-z]', o)) for _, o in options) * 2 > len(options)
    if any(s.get('claim') for s in stmts) and expr_opts:   # the options are expressions: read each one
        return {**out, **_which_expression(p, stmts, read, priors)}
    if any(s.get('claim') for s in stmts):
        return {**out, **_claims(stmts, options, low, people, priors)}
    if any(s.get('part') for s in stmts):
        return {**out, **_sufficiency(stmts, options, people, priors)}
    spoken = '"' in p['tokens'] and re.search(r'\b(?:said|says|say|told|tells|introduc\w*|pointing|points?)\b', ' '.join(p['tokens']))
    if spoken and not SMALLEST:   # the convention only on readings the exact speech reader wrote (model misreadings: off)
        from .speech import read as read_speech
        spoken = read_speech(p['tokens']) == lf
    r = solve(stmts, people=people, priors=priors, smallest=bool(spoken))
    if not r['questions']:
        return {**out, 'answer': 'no question found', 'option': None, 'status': 'none', 'explanation': ''}
    if not r['ok']:
        return {**out, 'answer': 'contradiction', 'option': None, 'status': 'contradiction',
                'explanation': 'The facts as read contradict each other.'}
    q = r['questions'][-1]
    qi = len(r['questions']) - 1
    t = q['q']['t']
    if t == 'pair':
        what = q['q']['what']
        ok = []
        for lab, o in options:
            pair = _names_in(o, ents)
            if len(pair) == 2 and r['worlds'] and all(pair_holds(w, what, r['ids'][pair[0]], r['ids'][pair[1]])
                                                      for w in r['worlds'] if pair[0] in r['ids'] and pair[1] in r['ids']):
                ok.append(lab)
        return {**out, 'status': 'pair', 'explanation': f'[CHECKED]: pairs that are {what}: {", ".join(ok) or "none"}',
                'answer': ', '.join(ok) or 'none', 'option': ok[0] if len(ok) == 1 else _labelled(options, NONE)}
    ex = explain(stmts, r, qi, names=names, priors=priors, options=options) if q['answers'] else None
    res = {'status': q['status'], 'explanation': ex['text'] if ex else 'The statements contradict each other.'}
    if not q['answers']:
        return {**out, **res, 'answer': 'contradiction', 'option': _labelled(options, CANT)}
    if t == 'ask':
        cands = list(q['answers'])
        status = 'gender' if q['status'] == 'unique' and cands[0].get('moves') and not cands[0].get('gender') else q['status']
        if options and q['status'] == 'gender' and len({a.get('key') for a in cands}) > 1 and _labelled(options, CANT):
            return {**out, **res, 'answer': 'cannot be determined', 'option': _labelled(options, CANT)}   # only a gender the text never gives decides
        a, lab = _relation_option(cands, options, status) if options else (cands[0], None)
        if options and not lab:
            if status in ('conditional', 'ambiguous', 'assumed') and _labelled(options, NONE) and not _relation_option(alternatives(r, qi), options, status)[1]:
                return {**out, **res, 'answer': cands[0].get('label') or cands[0]['key'], 'option': _labelled(options, NONE)}   # the natural reading is not offered
            if status in ('gender', 'conditional', 'ambiguous') and _labelled(options, CANT):
                return {**out, **res, 'answer': 'cannot be determined', 'option': _labelled(options, CANT)}
            a, lab = _relation_option(alternatives(r, qi), options, status)
            if lab:
                res['explanation'] += f"\n[NOTE]: Read with {names.get(a['merged'][1], a['merged'][1])} as the person the text leaves unnamed."
        a = a or cands[0]
        label = a.get('label') or a['key']
        text_ans = q.get('neutral') or label if q['status'] == 'gender' and a is cands[0] else label
        return {**out, **res, 'answer': text_ans, 'option': lab or (_labelled(options, NONE) if options else None)}
    if t == 'who':
        want = q['answers'][0].get('names', [])
        lab = next((l for l, o in options if want and sorted(_names_in(o, ents)) == sorted(want)), None)
        return {**out, **res, 'answer': ', '.join(names.get(n, n) for n in want) or 'not named', 'option': lab or _labelled(options, NONE)}
    if t == 'count':
        vals = [a['key'] for a in q['answers']]
        if len(vals) > 1 and _labelled(options, CANT):
            return {**out, **res, 'answer': 'cannot be determined', 'option': _labelled(options, CANT)}
        lab = next((l for l, o in options if str(_number(o)) == vals[0]), None)
        return {**out, **res, 'answer': vals[0], 'option': lab}
    if t == 'gender?':
        g = q['answers'][0]['key']
        lab = _labelled(options, re.compile(rf'^\s*{g}\b', re.I)) if g in ('male', 'female') else _labelled(options, CANT)
        return {**out, **res, 'answer': g if g != 'unknown' else 'cannot be determined', 'option': lab}
    return {**out, **res, 'answer': q['answers'][0]['key'], 'option': None}


def _claims(stmts, options, low, people, priors):
    r = solve(stmts, people=people, priors=priors)
    verdicts = r['claims']
    if re.search(r"cannot be (?:definitely )?true|can.?t be (?:definitely )?true|not (?:definitely|necessarily) true", low):
        want, hits = 'not definitely true', {k for k, v in verdicts.items() if v != 'true'}
    else:
        want = 'false' if re.search(r"\bfalse\b|not true|incorrect|not correct|cannot be derived|can't be derived|not be derived", low) else 'true'
        hits = {k for k, v in verdicts.items() if v == want}
    ans = f"{len(hits)} statement{'s' if len(hits) != 1 else ''} {want}: " + (', '.join(str(k) for k in sorted(hits)) or 'none')
    lab = None
    if 'how many' in low:
        lab = next((l for l, o in options if _number(o) == len(hits)), None)
    elif options and len(options) == len(verdicts) and all(re.search(r'\b(is|are|was)\b', o) for _, o in options):
        picked = [l for (l, _), k in zip(options, sorted(verdicts)) if k in hits]   # the options are the statements
        lab = picked[0] if len(picked) == 1 else None
    else:
        for l, o in options:
            s = _roman_set(o)
            if s == 'all' and hits == set(verdicts) or s == hits:
                lab = l
                break
    lines = [f'[CLAIM {k}]: {v}' for k, v in sorted(verdicts.items())]
    return {'answer': ans, 'option': lab, 'status': 'claims', 'explanation': '\n'.join(lines)}


def settled(r) -> bool:
    """Data sufficiency: one relation with its gender fixed by the statements (not by names); the side may stay open
    ("grandmother" answers "how is P related to Q?", "son or daughter" does not)."""
    if not r['questions'] or not r['questions'][-1]['answers']:
        return False
    q, a = r['questions'][-1], r['questions'][-1]['answers'][0]
    genderless = (a.get('base') or a.get('key') or '').endswith('cousin')   # English has one word for both
    return q['status'] in ('unique', 'side') and (genderless or (bool(a.get('gender')) and not a.get('assumed')))


def _sufficiency(stmts, options, people, priors):
    parts = sorted({s['part'] for s in stmts if s.get('part')})

    def enough(sub):
        return settled(solve(stmts, parts=set(sub), people=people, priors=priors))

    alone = {k: enough([k]) for k in parts}
    together = enough(parts)
    lines = [f"[STATEMENT {k}]: {'sufficient alone' if v else 'not sufficient alone'}" for k, v in alone.items()]
    lines.append(f"[TOGETHER]: {'sufficient' if together else 'not sufficient'}")
    lab = None
    if len(parts) == 2:
        a, b = alone[parts[0]], alone[parts[1]]
        verdict = 'either' if a and b else 'only1' if a else 'only2' if b else 'both' if together else 'neither'
        pats = {'only1': r'(only|alone).*\bi\b(?!i)|\bi\b(?!i).*alone', 'only2': r'(only|alone).*\bii\b|\bii\b.*alone',
                'either': r'either', 'both': r'both|together', 'neither': r'neither|not sufficient|even'}
        lab = next((l for l, o in options if re.search(pats[verdict], o.lower())), None)
    else:
        for l, o in options:
            nums = {int(n) for n in re.findall(r'\d', o)}
            if re.search(r'not sufficient', o.lower()):
                if not together:
                    lab = l
            elif re.search(r'\ball\b', o.lower()) and together and not any(enough(sorted(set(parts) - {k})) for k in parts):
                lab = l
            elif nums and set(nums) <= set(parts) and enough(sorted(nums)):
                lab = lab or l
    return {'answer': lines[-1].split(': ')[1], 'option': lab, 'status': 'sufficiency', 'explanation': '\n'.join(lines)}


def _which_expression(p, stmts, read, priors):
    """ "Which expression shows that M is the maternal uncle of T?": read each option with the code definitions."""
    claim = [s for s in stmts if s.get('claim') and s['t'] == 'rel']
    if not claim:
        return {'answer': 'no claim found', 'option': None, 'status': 'none', 'explanation': ''}
    c = claim[0]
    names = {e['id']: e['text'] for e in p['ents']}
    q = p['question']
    cut = re.search(r'which (?:of the following|one)', q, re.I)
    defs = q[:cut.start()] if cut else q
    texts = [f"{defs} In the expression '{o}', how is {names.get(c['y'], c['y'])} related to {names.get(c['x'], c['x'])}?"
             for _, o in p['options']]
    preps = [prepare(t) for t in texts]
    lfs = [g if isinstance(g, str) else (g[0][0] if isinstance(g[0], tuple) else g[0]) for g in read([pp['tokens'] for pp in preps])]
    ok, options_ = [], p['options']
    for (lab, o), pp, lf in zip(p['options'], preps, lfs):
        by = {e['text']: e['id'] for e in pp['ents']}
        x, y = by.get(names.get(c['x'])), by.get(names.get(c['y']))
        st, _ = parse(lf)
        if x is None or y is None:
            continue
        r = solve(st, people=[e['id'] for e in pp['ents']])
        if r['worlds'] and all(claim_holds(w, [{**c, 'x': x, 'y': y, 'claim': 1}], r['ids']) is True for w in r['worlds']):
            ok.append(lab)
    return {'answer': ', '.join(ok) or 'none', 'option': ok[0] if len(ok) == 1 else (_labelled(options_, NONE) if not ok else None),
            'status': 'expression',
            'explanation': f"[CHECKED]: options {', '.join(ok) or 'none'} make {names.get(c['y'])} the {c['r']} of {names.get(c['x'])}"}
