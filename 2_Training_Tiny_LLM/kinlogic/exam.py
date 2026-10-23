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
NEAREST = os.environ.get('KIN_NEAREST', '1') == '1'   # still nothing: the closest option by generation, gender and kind
INVERSE = os.environ.get('KIN_INVERSE', '1') == '1'   # no option names the answer but one names the reverse relation: take it
RELAX = os.environ.get('KIN_RELAX', '1') == '1'   # a self-contradictory question still gets the answer its keys expect
DESC_UNNAMED = os.environ.get('KIN_DESC_UNNAMED', '1') == '1'   # described people may turn out to be someone named
SMALLEST = os.environ.get('KIN_SMALLEST', '0') == '1'   # smallest family for every spoken description (dev +1.1, locked test -0.4 with r3); off: exact speech readings only


def answer(text: str, read, priors: dict | None = None) -> dict:
    """Never raises: a failure comes back as status 'error' with the reason."""
    try:
        return _answer(text, read, priors)
    except Exception as e:   # one odd question must not stop a batch or the page
        return {'answer': 'error', 'option': None, 'status': 'error', 'explanation': f'{type(e).__name__}: {e}', 'facts': ''}


def answer_adaptive(text: str, read, read_more, threshold: float = 0.6, priors: dict | None = None) -> dict:
    """More readings only where the first ones disagree (test-time compute where it is needed, Snell et al. 2024):
    answer with `read`; when the winning option holds less than `threshold` of the vote or nothing fits, answer again
    with `read_more` (a wider beam plus samples)."""
    first = answer(text, read, priors)
    if first.get('status') not in ('error', 'none', 'contradiction') and first.get('confidence', 1.0) >= threshold:
        return first
    second = answer(text, read_more, priors)
    return second if second.get('status') not in ('error',) else first


def with_draft(tokens, draft: str, verdict: str):
    """Correction input (build_corrections.py): the question, then a draft reading and the engine's verdict on it."""
    return list(tokens) + ['draft', ':'] + draft.split() + ['check', ':', verdict]


def answer_corrected(text: str, read, threshold: float = 0.6, priors: dict | None = None) -> dict:
    """Re-read and correct: when no family fits the readings or the winning option holds less than `threshold` of
    the vote, the reader reads again with its draft and the engine's verdict in view (a reader trained on
    build_corrections.py rows); the second answer is kept when it fits."""
    first = answer(text, read, priors)
    status = first.get('status')
    if status == 'error' or (status not in ('none', 'contradiction') and first.get('confidence', 1.0) >= threshold):
        return first
    draft = (first.get('facts') or '').split('  [')[0].strip()
    if not draft:
        return first
    verdict = 'contradiction' if status in ('none', 'contradiction') else 'unsure'
    main = prepare(text)['tokens']
    second = answer(text, lambda batch: read([with_draft(t, draft, verdict) if t == main else t for t in batch]), priors)
    return second if second.get('status') not in ('error', 'none', 'contradiction') else first


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


def _option_statements(options, ents):
    """Claims for options that are statements about the question's people, or None."""
    from .codes import option_claims
    ids = {e['text']: e['id'] for e in ents if e.get('kind') in ('name', 'letter')}
    return option_claims(options, ids)


ROLE_PAIR = re.compile(r'^\s*([a-z][a-z -]*?)\s*(?:–|-|—|and|/|&)\s*([a-z][a-z -]*?)\s*$', re.I)
PLURAL_ROLE = {'cousins': 'cousin', 'brothers': 'brother', 'sisters': 'sister', 'siblings': 'sibling', 'couple': 'spouse',
               'husband and wife': 'spouse', 'brothers-in-law': 'brother-in-law', 'sisters-in-law': 'sister-in-law'}


def _pair_option(r, stmts, options, ents):
    """For "how is Y related to X?" / "relationship between X and Y": the one option that holds in every family, when
    the options are statements ("P is the husband of T") or role pairs ("Uncle – Niece", "Cousins")."""
    from .codes import _rel as code_rel
    ask = next((st for st in reversed(stmts) if st['t'] == 'ask'), None)
    if ask is None or not r['worlds']:
        return None
    x, y = ask['x'], ask['y']
    ids = {e['text']: e['id'] for e in ents if e.get('kind') in ('name', 'letter')}
    st_claims = _option_statements(options, ents)

    def holds(facts):
        return all(claim_holds(w, facts, r['ids']) is True for w in r['worlds'])

    def rel_fact(a, word, b):   # b is a's WORD
        rr = code_rel(word.strip().lower())
        return None if rr is None else {'t': 'rel', 'x': a, 'r': rr[1], 'side': rr[0], 'y': b, 'neg': False, 'only': False, 'claim': 1}

    ok = []
    if st_claims:
        groups, cur = [], None
        for st in parse(' ; '.join(st_claims))[0]:
            if cur is None or st.get('claim') != cur[0]:
                cur = (st.get('claim'), [])
                groups.append(cur)
            cur[1].append(st)
        labs = [l for l, o in options if not NONE.search(o) and not CANT.search(o)]
        if len(groups) != len(labs):
            return None
        ok = [l for l, (_, fs) in zip(labs, groups) if all(f.get('x') in r['ids'] and (f.get('y', '@x')[0] == '#' or f.get('y') in r['ids']) for f in fs) and holds(fs)]
    else:
        for lab, o in options:
            t = o.strip().lower().rstrip('.')
            if t in PLURAL_ROLE:   # "Cousins": each is the other's
                f1, f2 = rel_fact(x, PLURAL_ROLE[t], y), rel_fact(y, PLURAL_ROLE[t], x)
            elif (m := ROLE_PAIR.match(t)) and code_rel(m.group(1).strip()) and code_rel(m.group(2).strip()):
                f1, f2 = rel_fact(x, m.group(1), y), rel_fact(y, m.group(2), x)   # "Uncle – Niece": the first named (y) is the uncle
            else:
                continue
            if f1 and f2 and x in r['ids'] and y in r['ids'] and holds([f1, f2]):
                ok.append(lab)
    return ok[0] if len(ok) == 1 else None


def _shape(mv):
    """(generations up, gender, by marriage, collateral) of a move path."""
    ms = [m[0] for m in mv]
    return ms.count('U') - ms.count('D'), mv[-1][1], 'S' in ms, any(m == 'U' and 'D' in ms[i + 1:] for i, m in enumerate(ms))


def _nearest_option(a, options):
    """The one option with the answer's generation and gender that best matches its kind (blood or marriage,
    direct or collateral); None unless exactly one is best."""
    if not a.get('moves'):
        return None
    gen, g, aff, coll = _shape(a['moves'])
    g = a.get('gender') or g
    scored = []
    for lab, text in options:
        mv = option_moves(text)
        if not mv:
            continue
        ogen, og, oaff, ocoll = _shape(mv)
        if ogen == gen and (not g or not og or g == og):
            scored.append(((oaff == aff) + (ocoll == coll), lab))
    top = max((sc for sc, _ in scored), default=None)
    best = [lab for sc, lab in scored if sc == top]
    return best[0] if len(best) == 1 else None


def _inverse(stmts, people, priors, smallest):
    """The last asked relation the other way round (y's relation to x), or None."""
    i = max((k for k, st in enumerate(stmts) if st['t'] == 'ask'), default=None)
    if i is None:
        return None
    alt = stmts[:i] + [{**stmts[i], 'x': stmts[i]['y'], 'y': stmts[i]['x']}] + stmts[i + 1:]
    r = solve(alt, people=people, priors=priors, smallest=smallest)
    return r['questions'][-1] if r['ok'] and r['questions'] and r['questions'][-1]['answers'] else None


def _relaxed(stmts, people, priors, smallest):
    """(result, facts) for the first single change that makes contradictory facts consistent: a gendered relation word
    made neutral (wife -> spouse, mother -> parent) or a stated gender dropped; None if no single change does."""
    from .engine import NEUTRAL
    from .kin import KIN
    for i, st in enumerate(stmts):
        if st.get('claim') or st.get('part') or st.get('neg'):
            continue
        if st['t'] == 'rel' and NEUTRAL.get(st['r']) in KIN:
            alt = stmts[:i] + [{**st, 'r': NEUTRAL[st['r']]}] + stmts[i + 1:]
        elif st['t'] == 'gender':
            alt = stmts[:i] + stmts[i + 1:]
        else:
            continue
        r = solve(alt, people=people, priors=priors, smallest=smallest)
        if r['ok'] and r['questions']:
            return r, alt
    return None


def _answer_one(p, lf, read, priors):
    ents, options = p['ents'], p['options']
    priors = {**name_priors(ents), **(priors or {})}
    names = {e['id']: e['text'] for e in ents}
    out = {'facts': lf, 'options': options}
    lf_read = lf
    if DESC_UNNAMED:   # "the boy", "a man": a description, not a name, so facts like "only grandson" may make it someone named
        desc = {e['id'] for e in ents if e.get('kind') == 'desc'}
        lf = re.sub(r'@(\d+)\b', lambda m: f'#{900 + int(m.group(1))}' if m.group(0) in desc else m.group(0), lf)
        names.update({f'#{900 + int(d[1:])}': names[d] for d in desc})
    stmts, _ = parse(lf)
    people = _people(p['tokens'], stmts)
    low = p['question'].lower()
    if options and stmts and not any(st['t'] in ('ask', 'who', 'count', 'gender?', 'pair') or st.get('claim') or st.get('part') for st in stmts) \
            and re.search(r'which (?:of the following|one|among the following)[^?]*\b(?:true|correct|false|incorrect)\b', low):
        cl = _option_statements(options, ents)   # "which of the following statements is not definitely true?" after an exact reading
        if cl:
            stmts = stmts + parse(' ; '.join(cl))[0]

    expr_opts = options and sum(bool(re.search(r'\b[A-Za-z]\s*(?:[^\w\s,.()\'"]{1,2}|\d)\s*[A-Za-z]\b', o)) for _, o in options) * 2 > len(options)
    if any(s.get('claim') for s in stmts) and expr_opts:   # the options are expressions: read each one
        return {**out, **_which_expression(p, stmts, read, priors)}
    if any(s.get('claim') for s in stmts):
        return {**out, **_claims(stmts, options, low, people, priors)}
    if any(s.get('part') for s in stmts):
        return {**out, **_sufficiency(stmts, options, people, priors)}
    spoken = '"' in p['tokens'] and re.search(r'\b(?:said|says|say|told|tells|introduc\w*|pointing|points?)\b', ' '.join(p['tokens']))
    if spoken and not SMALLEST:   # the convention only on readings the exact speech reader wrote (model misreadings: off)
        from .rules import read as read_rules
        from .speech import read as read_speech
        spoken = lf_read in (read_speech(p['tokens']), read_rules(p['tokens']))
    r = solve(stmts, people=people, priors=priors, smallest=bool(spoken))
    if not r['questions']:
        return {**out, 'answer': 'no question found', 'option': None, 'status': 'none', 'explanation': ''}
    if not r['ok'] and RELAX:   # self-contradictory question (a coded chain that makes P both a wife and a mother):
        relaxed = _relaxed(stmts, people, priors, bool(spoken))   # answer with the one gender clash taken out
        if relaxed:
            r, stmts = relaxed
            out['facts'] = out['facts'] + '  [relaxed: one gender clash ignored]'
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
            elif (lab := _pair_option(r, stmts, options, ents)):   # "(a) P is the husband of T", "(b) Uncle – Niece"
                res['explanation'] += '\n[NOTE]: The option that holds in every family that fits.'
                return {**out, **res, 'answer': cands[0].get('label') or cands[0]['key'], 'option': lab}
            elif INVERSE and not _labelled(options, NONE) and not _labelled(options, CANT):
                q2 = _inverse(stmts, people, priors, bool(spoken))   # "How is Riya related to that man? (a) Father": keyed the other way round
                a, lab = _relation_option(list(q2['answers']), options, q2['status']) if q2 else (None, None)
                if lab:
                    res['explanation'] += '\n[NOTE]: No option names this relation; the options name the reverse one.'
                    return {**out, **res, 'answer': a.get('label') or a['key'], 'option': lab}
                lab = _nearest_option(cands[0], options) if NEAREST else None
                if lab:   # "father's cousin" with options Sister / Mother / Mother-in-law / Aunt: the setter's loose "Aunt"
                    res['explanation'] += '\n[NOTE]: No option names this relation exactly; the closest one (same generation, gender and kind) is chosen.'
                    return {**out, **res, 'answer': cands[0].get('label') or cands[0]['key'], 'option': lab}
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
    if t == 'prof?':   # "What is the profession of X?"
        keys = {a['key'] for a in q['answers']}
        if len(keys) > 1 or 'unknown' in keys:
            return {**out, **res, 'answer': 'cannot be determined', 'option': _labelled(options, CANT)}
        p_ = keys.pop()
        return {**out, **res, 'answer': p_, 'option': next((l for l, o in options if re.fullmatch(rf'\s*{p_}s?\s*\.?', o, re.I)), None)}
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
    elif options and len(stmt_opts := [(l, o) for l, o in options if re.search(r'\b(is|are|was)\b', o) and not NONE.search(o)]) == len(verdicts):
        picked = [l for (l, _), k in zip(stmt_opts, sorted(verdicts)) if k in hits]   # the options are the statements
        lab = picked[0] if len(picked) == 1 else (_labelled(options, NONE) if not picked else None)   # "(d) None of the above"
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
    ok, maybe, options_ = [], [], p['options']
    for (lab, o), pp, lf in zip(p['options'], preps, lfs):
        by = {e['text']: e['id'] for e in pp['ents']}
        x, y = by.get(names.get(c['x'])), by.get(names.get(c['y']))
        st, _ = parse(lf)
        if x is None or y is None:
            continue
        r = solve(st, people=[e['id'] for e in pp['ents']])
        vs = [claim_holds(w, [{**c, 'x': x, 'y': y, 'claim': 1}], r['ids']) for w in r['worlds']]
        if vs and all(v is True for v in vs):
            ok.append(lab)
        elif vs and all(v is not False for v in vs):
            maybe.append(lab)   # holds once the gender the expression leaves open is the asked one
    if not ok and len(maybe) == 1:
        ok = maybe
    return {'answer': ', '.join(ok) or 'none', 'option': ok[0] if len(ok) == 1 else (_labelled(options_, NONE) if not ok else None),
            'status': 'expression',
            'explanation': f"[CHECKED]: options {', '.join(ok) or 'none'} make {names.get(c['y'])} the {c['r']} of {names.get(c['x'])}"}
