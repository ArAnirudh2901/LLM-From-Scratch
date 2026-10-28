#!/usr/bin/env python3
"""Create a verified, exactly balanced supplement of speech and long-story questions.

The data is generated from explicit family graphs. The source graph is not treated as ground truth by itself:
each emitted text is independently read by the same exact speech/rule readers used by the project and its answer is
re-solved before it is accepted.

    python generate_balanced_question_types.py --per-cell 100

The default yields 3,750 rows: 1,250 direct speech, 1,250 indirect speech, and 1,250 long stories. Each format has
50 rows in each of 25 canonical answer types, including the gender-specific core relations and common in-law types.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import random
import re
import time
from pathlib import Path

from kinlogic import generate as gen
from kinlogic.deep import _neutral
from kinlogic.engine import World, relation_in, solve
from kinlogic.facts import parse
from kinlogic.generate import F, Family, Names, Q, _family, fact_for
from kinlogic.prepare import prepare
from kinlogic.rules import read as read_rules
from kinlogic.speech import read as read_speech


KIN_TYPES = ('father', 'mother', 'son', 'daughter', 'brother', 'sister', 'husband', 'wife',
             'grandfather', 'grandmother', 'grandson', 'granddaughter', 'uncle', 'aunt', 'nephew', 'niece',
             'cousin', 'father_in_law', 'mother_in_law', 'brother_in_law', 'sister_in_law',
             'son_in_law', 'daughter_in_law', 'other_affinal')
ANSWER_TYPES = (*KIN_TYPES, 'no_link')
FORMATS = ('direct_speech', 'indirect_speech', 'long_story')
STORY_LENGTHS = {
    'short_long': {'words': (80, 120), 'clues': 11, 'per_component': 7},
    'medium': {'words': (121, 170), 'clues': 16, 'per_component': 10},
    'extended': {'words': (171, 240), 'clues': 21, 'per_component': 13},
}


def relation_type(answer: dict) -> str | None:
    """Map exact engine labels to a balanced, exam-level kinship relation type."""
    key = str(answer.get('key') or '').lower()
    label = str(answer.get('label') or answer.get('base') or key).lower()
    text = f'{key} {label} {answer.get("note") or ""}'.lower()
    if key in {'none', 'not family'} or key.startswith('not family:'):
        return 'no_link'
    if 'grand' in label and ('in-law' in text or 'in law' in text):
        return 'other_affinal'
    for label_part, bucket in (('father-in-law', 'father_in_law'), ('mother-in-law', 'mother_in_law'),
                               ('brother-in-law', 'brother_in_law'), ('sister-in-law', 'sister_in_law'),
                               ('son-in-law', 'son_in_law'), ('daughter-in-law', 'daughter_in_law')):
        if label_part in text:
            return bucket
    if (answer.get('note') == 'by marriage' or any(x in text for x in ('in-law', 'in law', 'co-brother', 'co-sister', 'co-sibling', 'by marriage'))
            or (key not in {'husband', 'wife', 'spouse'} and any(x in label for x in ("'s husband", "'s wife")))):
        return 'other_affinal'
    if 'cousin' in text:
        return 'cousin'
    for part in ('grandfather', 'grandmother', 'grandson', 'granddaughter', 'father', 'mother', 'son', 'daughter',
                 'brother', 'sister', 'husband', 'wife', 'uncle', 'aunt', 'nephew', 'niece'):
        if part in text:
            return part
    return None


class DisjointFamilies:
    """A World containing independent Family components, used to make explicit no-link examples."""

    def __init__(self, rng: random.Random, sizes=(7, 7)):
        self.w = World()
        self.people = []
        self.level = {}
        self.components = []
        for size in sizes:
            src = Family(rng, size=size, gens=3)
            mapping = {}
            component = []
            for old in src.people:
                new = self.w.add(f'@p{len(self.people)}')
                self.w.gen[new] = src.g(old)
                mapping[old] = new
                component.append(new)
                self.people.append(new)
                self.level[new] = src.level[old]
            for old, new in mapping.items():
                for slot in ('fa', 'mo', 'sp'):
                    old_rel = src.w.rel(old, slot)
                    if old_rel >= 0:
                        getattr(self.w, slot)[new] = mapping[old_rel]
            self.components.append(component)

    def g(self, node):
        return self.w.g(node)

    def hops(self, a, b):
        path = self.w.path(a, b) or []
        out, i = [], 0
        while i < len(path):
            move, _, node = path[i]
            if move == 'U' and i + 1 < len(path) and path[i + 1][0] == 'D':
                out.append(('B', path[i + 1][2]))
                i += 2
            else:
                out.append((move, node))
                i += 1
        return out

    def truth(self, a, b):
        return relation_in(self.w, a, b)


def solve_reading(text: str, parser, expected_family: str | None = None):
    prepared = prepare(text)
    why = []
    lf = parser(prepared['tokens'], why)
    if not lf:
        return None, 'reader:' + (why[0] if why else 'no reading')
    stmts, bad = parse(lf)
    if bad:
        return None, 'fact-parse:' + str(bad[0])
    people = sorted({t for t in prepared['tokens'] if t.startswith('@')})
    result = solve(stmts, people=people, max_steps=20000)
    if not result.get('ok') or not result.get('questions'):
        return None, 'engine:no-question-or-inconsistent'
    question = result['questions'][-1]
    answers = question.get('answers') or []
    if len(answers) != 1:
        return None, 'engine:non-unique'
    answer = answers[0]
    bucket = relation_type(answer)
    if bucket != expected_family:
        return None, f'engine:family:{bucket or "unknown"}'
    return {'lf': lf, 'answer': answer, 'bucket': bucket, 'prepared': prepared, 'question': question}, None


def _options_match(prepared: dict, answer: dict) -> bool:
    expected = str(answer.get('label') or answer.get('key') or '').strip().lower()
    options = prepared.get('options') or []
    values = [str(item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item).strip().lower() for item in options]
    return sum(value == expected for value in values) == 1


def _source_family(ex: dict) -> str | None:
    return relation_type({'key': ex.get('truth'), 'label': ex.get('relation_label')})


def equivalent_lf(left: str, right: str) -> bool:
    """Compare logical facts without treating repeated gender clues as different meanings."""
    return tuple(sorted(set(part.strip() for part in left.split(';')))) == tuple(
        sorted(set(part.strip() for part in right.split(';'))))


def speech_pair(rng: random.Random):
    """Same graph, names, question, options, and LF; only quote/reporting form differs."""
    state0 = rng.getstate()
    direct = gen.q_speech(rng, speech_mode='direct')
    rng.setstate(state0)
    indirect = gen.q_speech(rng, speech_mode='indirect')
    if not direct or not indirect:
        return None
    if direct.get('lf') != indirect.get('lf') or direct.get('truth') != indirect.get('truth'):
        return None
    return direct, indirect


def no_link_speech_pair(rng: random.Random):
    """Paired direct/reported questions whose two named people have independent family contexts."""
    fam = DisjointFamilies(rng, (7, 7))
    q = Q(rng, fam, Names(rng, rng.choice(['indian', 'world', 'mixed'])))
    speaker = rng.choice(fam.components[0])
    target = rng.choice(fam.components[1])
    rs, rp = q.ref(speaker), q.ref(target)
    gender = fam.g(speaker)
    relation, kinword = ('father', 'son') if gender == 'M' else ('mother', 'daughter')
    pronoun = 'he' if gender == 'M' else 'she'
    parent_pronoun = 'his' if gender == 'M' else 'her'
    parent = q.anon()
    quote = f'I am the only {kinword} of my {relation}.'
    facts = [F(rs, relation, parent), F(parent, kinword, rs, only=True)]
    opening = f'Pointing to {rp.surface}, {rs.surface} said, "{quote}"'
    q.add(opening, facts)
    # A true clue about the pointed-at person's separate family makes absence of a link a real reasoning case.
    other = next((node for node in fam.components[1] if node != target and len(fam.hops(target, node)) == 1), None)
    if other is not None:
        clue = make_clue(q, target, other)
        if clue:
            q.add(clue[0] + '.', clue[1])
    q.add(f'How is {rp.surface} related to {rs.surface}?', [{'t': 'ask', 'x': rs, 'y': rp}])
    direct = q.finish('speech', ['No family link', 'Brother', 'Aunt', 'Cousin', 'Cannot be determined'], {'truth': 'none'}, noise=False)
    if direct is None:
        return None

    # Reuse the exact same q and facts, replacing only the opening clause. Keep this branch RNG-free so paired rows
    # are guaranteed to have matching fact targets and options.
    indirect = dict(direct)
    # Use the speaker's explicit name in the report: the exact speech reader resolves third-person pronouns in
    # this construction to the person being pointed at, so "said that he is ..." would change the meaning.
    reported = f'{rs.surface} is the only {kinword} of {rs.surface}\'s {relation}.'
    indirect['text'], replacements = re.subn(
        re.escape(opening), f'Pointing to {rp.surface}, {rs.surface} said that {reported}',
        direct['text'], count=1, flags=re.I)
    if replacements != 1:
        return None
    indirect['tokens'] = prepare(indirect['text'])['tokens']
    return direct, indirect


def make_clue(q: Q, a: int, b: int):
    run = q.fam.hops(a, b)
    if not run or tuple(move for move, _ in run) not in gen.WORDS:
        return None
    x, y, word, side = fact_for(q, run, a, b, side_p=0.1)
    rx, ry = q.ref(x), q.ref(y)
    relation = ' '.join(part for part in (side, word) if part)
    text = q.rng.choice((
        f'{ry.surface} is the {relation} of {rx.surface}',
        f'{ry.surface} is {rx.surface}\'s {relation}',
        f'{rx.surface}\'s {relation} is {ry.surface}',
    ))
    return text, [F(rx, word, ry, side)]


def candidate_pairs(fam, nodes):
    pairs = []
    for x in nodes:
        for y in nodes:
            if x == y:
                continue
            hops = fam.hops(x, y)
            if not 1 <= len(hops) <= 3 or tuple(move for move, _ in hops) not in gen.WORDS:
                continue
            pairs.append((x, y))
    return pairs


def add_story_clues(q: Q, nodes, target_nodes=(), count=10, rng=None):
    rng = rng or q.rng
    pairs = candidate_pairs(q.fam, nodes)
    rng.shuffle(pairs)
    selected, seen = [], set()
    for x, y in pairs:
        key = frozenset((x, y))
        if key in seen:
            continue
        clue = make_clue(q, x, y)
        if not clue:
            continue
        selected.append((x, y, clue))
        seen.add(key)
        if len(selected) >= count and all(any(x in (a, b) for a, b, _ in selected) for x in target_nodes):
            break
    if not all(any(x in (a, b) for a, b, _ in selected) for x in target_nodes):
        return 0
    for _, _, (text, facts) in selected:
        q.add(text + '.', facts)
    return len(selected)


def add_verified_constraints(q: Q, nodes, rng: random.Random):
    """Add one true exclusivity and one false relation when the generated graph makes them exact."""
    only_candidates = []
    direct = ('father', 'mother', 'son', 'daughter', 'brother', 'sister', 'husband', 'wife')
    for x in nodes:
        for word in direct:
            rels = q.fam.w.relatives(x, word)
            if len(rels) == 1:
                only_candidates.append((x, rels[0], word))
    if only_candidates:
        x, y, word = rng.choice(only_candidates)
        rx, ry = q.ref(x), q.ref(y)
        q.add(f'{ry.surface} is the only {word} of {rx.surface}.', [F(rx, word, ry, only=True)])

    neg_candidates = []
    for x in nodes:
        for y in nodes:
            if x == y:
                continue
            for word in direct:
                if y not in q.fam.w.relatives(x, word):
                    neg_candidates.append((x, y, word))
    if neg_candidates:
        x, y, word = rng.choice(neg_candidates)
        rx, ry = q.ref(x), q.ref(y)
        q.add(f'{ry.surface} is not the {word} of {rx.surface}.', [F(rx, word, ry, neg=True)])


def add_gender_clues_until(q: Q, query: str, min_words: int, rng: random.Random):
    """Pad longer exam stories only with solver-verifiable gender clues about named family members."""
    nodes = list(q.refs)
    rng.shuffle(nodes)
    for node in nodes:
        if len((' '.join(q.sentences + [query])).split()) >= min_words + 1:
            return True
        ref = q.ref(node)
        noun = 'man' if ref.g == 'M' else 'woman'
        q.add(f'{ref.surface} is a {noun}.', [gen.G(ref, ref.g)])
    return len((' '.join(q.sentences + [query])).split()) >= min_words


def connected_long_story(rng: random.Random, wanted: str, min_words: int, clue_count: int):
    fam = _family(rng, 11, 12)
    choices = []
    for a in fam.people:
        for b in fam.people:
            if a == b:
                continue
            truth = fam.truth(a, b)
            if relation_type(truth) == wanted:
                choices.append((a, b, truth))
    if not choices:
        return None
    a, b, truth = rng.choice(choices)
    q = Q(rng, fam, Names(rng, rng.choice(['indian', 'world', 'mixed'])))
    # The shortest path is always stated, so the query answer follows from story facts rather than dataset metadata.
    route = [a] + [node for _, node in fam.hops(a, b)]
    for x, y in zip(route, route[1:]):
        clue = make_clue(q, x, y)
        if not clue:
            return None
        q.add(clue[0] + '.', clue[1])
    if add_story_clues(q, fam.people, (a, b), count=clue_count, rng=rng) < clue_count:
        return None
    add_verified_constraints(q, fam.people, rng)
    ra, rb = q.ref(a), q.ref(b)
    for r in (ra, rb):
        if not gen.implied(q, r):
            noun = 'man' if r.g == 'M' else 'woman'
            q.add(f'{r.surface} is a {noun}.', [gen.G(r, r.g)])
    question = f'How is {rb.surface} related to {ra.surface}?'
    if not add_gender_clues_until(q, question, min_words, rng):
        return None
    q.add(question, [{'t': 'ask', 'x': ra, 'y': rb}])
    return q.finish('story', None, {'truth': truth['key'], 'format': 'long_story'}, noise=False)


def disconnected_long_story(rng: random.Random, min_words: int, clues_per_component: int):
    fam = DisjointFamilies(rng, (8, 8))
    q = Q(rng, fam, Names(rng, rng.choice(['indian', 'world', 'mixed'])))
    a = rng.choice(fam.components[0])
    b = rng.choice(fam.components[1])
    for comp, target in ((fam.components[0], a), (fam.components[1], b)):
        if add_story_clues(q, comp, (target,), count=clues_per_component, rng=rng) < clues_per_component:
            return None
    ra, rb = q.ref(a), q.ref(b)
    question = f'How is {rb.surface} related to {ra.surface}?'
    if not add_gender_clues_until(q, question, min_words, rng):
        return None
    q.add(question, [{'t': 'ask', 'x': ra, 'y': rb}])
    return q.finish('story', None, {'truth': 'none', 'format': 'long_story'}, noise=False)


def _label(answer: dict) -> str:
    return str(answer.get('label') or answer.get('key') or answer.get('base') or '')


def validate_row(ex: dict, fmt: str, expected: str):
    parser = read_speech if fmt in ('direct_speech', 'indirect_speech') else read_rules
    solved, why = solve_reading(ex['text'], parser, expected)
    if not solved:
        return None, why
    prepared = solved['prepared']
    answer = solved['answer']
    if fmt in ('direct_speech', 'indirect_speech') and expected != 'no_link' and not _options_match(prepared, answer):
        return None, 'options:gold-not-exactly-once'
    if fmt in ('direct_speech', 'indirect_speech') and expected == 'no_link' and not _options_match(prepared, {'label': 'No family link'}):
        return None, 'options:no-link-not-exactly-once'
    out = dict(ex)
    out['format'] = fmt
    out['relation_type'] = expected
    out['lf'] = solved['lf']
    out['truth'] = answer.get('key')
    out['answer'] = 'No family link' if expected == 'no_link' else _label(answer)
    out['word_count'] = len(ex['text'].split())
    out['relation_label'] = _label(answer)
    out['tokens'] = prepared['tokens']
    if fmt == 'long_story' and not 80 <= out['word_count'] <= 240:
        return None, f'story:length:{out["word_count"]}'
    return out, None


def row_key(ex):
    return hashlib.sha256(re.sub(r'\s+', ' ', ex['text'].lower()).strip().encode()).hexdigest()


def balanced_dataset(per_cell: int, seed: int, max_attempts: int):
    counts = {fmt: collections.Counter() for fmt in FORMATS}
    rows, seen = [], set()
    rejects = collections.Counter()
    attempts = {fmt: collections.Counter() for fmt in FORMATS}
    story_length_counts = collections.Counter()
    cursor = seed

    def add(ex, fmt, bucket):
        nonlocal rows
        checked, reason = validate_row(ex, fmt, bucket)
        if not checked:
            rejects[f'{fmt}:{reason}'] += 1
            return False
        key = row_key(checked)
        if key in seen:
            rejects[f'{fmt}:duplicate'] += 1
            return False
        seen.add(key)
        rows.append(checked)
        counts[fmt][bucket] += 1
        return True

    # Speech is generated as true same-seed pairs, which guarantees direct and indirect examples have equal counts
    # and the exact same underlying graph, answer, options, and target relation.
    target_pairs = {b: 0 for b in ANSWER_TYPES}
    speech_tries = 0
    while any(target_pairs[b] < per_cell for b in KIN_TYPES) and speech_tries < max_attempts:
        current_seed = cursor
        cursor += 1
        speech_tries += 1
        rng = random.Random(current_seed)
        pair = speech_pair(rng)
        if not pair:
            rejects['speech:pair-render-or-mismatch'] += 1
            continue
        direct, indirect = pair
        bucket = _source_family(direct)
        if bucket not in KIN_TYPES or target_pairs[bucket] >= per_cell:
            rejects['speech:unneeded-or-unclassified-source-bucket'] += 1
            continue
        dcheck, dwhy = validate_row(direct, 'direct_speech', bucket)
        icheck, iwhy = validate_row(indirect, 'indirect_speech', bucket)
        if not dcheck or not icheck:
            rejects['speech:' + str(dwhy or iwhy)] += 1
            continue
        if (not equivalent_lf(dcheck['lf'], icheck['lf']) or dcheck['truth'] != icheck['truth']
                or dcheck['answer'] != icheck['answer']):
            rejects['speech:reader-pair-not-equivalent'] += 1
            continue
        pair_id = hashlib.sha256(f'{current_seed}\n{dcheck["lf"]}'.encode()).hexdigest()[:20]
        dcheck['pair_id'] = icheck['pair_id'] = pair_id
        # insert pre-validated rows without repeating the solve
        for checked, fmt in ((dcheck, 'direct_speech'), (icheck, 'indirect_speech')):
            key = row_key(checked)
            if key in seen:
                rejects[f'{fmt}:duplicate'] += 1
                break
            seen.add(key)
            rows.append(checked)
            counts[fmt][bucket] += 1
        else:
            target_pairs[bucket] += 1
            attempts['direct_speech'][bucket] += 1
            attempts['indirect_speech'][bucket] += 1

    if target_pairs['no_link'] < per_cell:
        no_link_count = 0
        no_link_tries = 0
        while no_link_count < per_cell and no_link_tries < max_attempts:
            no_link_seed = cursor
            rng = random.Random(no_link_seed)
            cursor += 1
            no_link_tries += 1
            pair = no_link_speech_pair(rng)
            if not pair:
                rejects['speech-no-link:render'] += 1
                continue
            for ex, fmt in zip(pair, ('direct_speech', 'indirect_speech')):
                checked, why = validate_row(ex, fmt, 'no_link')
                if not checked:
                    rejects[f'{fmt}:{why}'] += 1
                    break
                # exact pair equivalence is checked after both validate below
                if fmt == 'direct_speech':
                    direct_checked = checked
                else:
                    indirect_checked = checked
            else:
                if (not equivalent_lf(direct_checked['lf'], indirect_checked['lf'])
                        or direct_checked['answer'] != indirect_checked['answer']):
                    rejects['speech-no-link:reader-pair-not-equivalent'] += 1
                    continue
                pair_id = hashlib.sha256(f'{no_link_seed}\n{direct_checked["lf"]}'.encode()).hexdigest()[:20]
                direct_checked['pair_id'] = indirect_checked['pair_id'] = pair_id
                keys = [row_key(direct_checked), row_key(indirect_checked)]
                if any(k in seen for k in keys):
                    rejects['speech-no-link:duplicate'] += 1
                    continue
                for checked, fmt, key in zip((direct_checked, indirect_checked), ('direct_speech', 'indirect_speech'), keys):
                    checked['tokens'] = prepare(checked['text'])['tokens']
                    seen.add(key)
                    rows.append(checked)
                    counts[fmt]['no_link'] += 1
                no_link_count += 1
                attempts['direct_speech']['no_link'] += 1
                attempts['indirect_speech']['no_link'] += 1

    # Split every answer type evenly across three story lengths so long stories are genuinely represented, not just
    # short stories with extra sentence wording.
    quotient, remainder = divmod(per_cell, len(STORY_LENGTHS))
    length_quotas = {name: quotient + (i < remainder) for i, name in enumerate(STORY_LENGTHS)}

    # Story format gets both connected kinship stories and genuine two-family/no-link stories.
    for bucket in ANSWER_TYPES:
        for length_bin, quota in length_quotas.items():
            accepted = 0
            tried = 0
            spec = STORY_LENGTHS[length_bin]
            min_words, max_words = spec['words']
            while accepted < quota and tried < max_attempts:
                rng = random.Random(cursor)
                cursor += 1
                tried += 1
                ex = (disconnected_long_story(rng, min_words, spec['per_component']) if bucket == 'no_link'
                      else connected_long_story(rng, bucket, min_words, spec['clues']))
                if not ex:
                    rejects[f'long_story:{bucket}:{length_bin}:render'] += 1
                    continue
                raw_word_count = len(ex['text'].split())
                if not min_words <= raw_word_count <= max_words:
                    rejects[f'long_story:{bucket}:{length_bin}:length:{raw_word_count}'] += 1
                    continue
                checked, why = validate_row(ex, 'long_story', bucket)
                if not checked:
                    rejects[f'long_story:{why}'] += 1
                    continue
                if not min_words <= checked['word_count'] <= max_words:
                    rejects[f'long_story:{length_bin}:length:{checked["word_count"]}'] += 1
                    continue
                key = row_key(checked)
                if key in seen:
                    rejects['long_story:duplicate'] += 1
                    continue
                checked['story_length_bin'] = length_bin
                seen.add(key)
                rows.append(checked)
                counts['long_story'][bucket] += 1
                story_length_counts[length_bin] += 1
                accepted += 1
                attempts['long_story'][bucket] += 1

    missing = {fmt: {bucket: per_cell - counts[fmt][bucket] for bucket in ANSWER_TYPES
                     if counts[fmt][bucket] != per_cell} for fmt in FORMATS}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise RuntimeError(f'Could not fill exact balance after attempt limit: {missing}; rejection summary: {rejects.most_common(12)}')
    rows.sort(key=lambda r: (FORMATS.index(r['format']), ANSWER_TYPES.index(r['relation_type']), row_key(r)))
    return rows, counts, attempts, rejects, story_length_counts, cursor


def save_gzip(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=9) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='\n') as text:
                for row in rows:
                    text.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-cell', type=int, default=50, help='accepted examples per format × answer type (default: 50)')
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--max-attempts-cell', type=int, default=200000)
    ap.add_argument('--out', default='data/balanced_question_formats_v1.jsonl.gz')
    args = ap.parse_args()
    started = time.time()
    rows, counts, attempts, rejects, story_length_counts, final_seed = balanced_dataset(args.per_cell, args.seed, args.max_attempts_cell)
    path = Path(args.out)
    save_gzip(path, rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'version': 1,
        'seed': args.seed,
        'next_seed': final_seed,
        'per_cell': args.per_cell,
        'formats': list(FORMATS),
        'relation_types': list(ANSWER_TYPES),
        'counts': {fmt: {bucket: counts[fmt][bucket] for bucket in ANSWER_TYPES} for fmt in FORMATS},
        'total': len(rows),
        'total_by_format': {fmt: sum(counts[fmt].values()) for fmt in FORMATS},
        'accepted_per_format_type': {fmt: {bucket: attempts[fmt][bucket] for bucket in ANSWER_TYPES} for fmt in FORMATS},
        'long_story_length_bins': {name: story_length_counts[name] for name in STORY_LENGTHS},
        'exact_answer_labels': {
            fmt: {label: count for label, count in collections.Counter(row['relation_label'] for row in rows if row['format'] == fmt).most_common()}
            for fmt in FORMATS
        },
        'story_constraint_coverage': {
            'with_only': sum(row['format'] == 'long_story' and ' only ' in f' {row["text"].lower()} ' for row in rows),
            'with_negation': sum(row['format'] == 'long_story' and ' not ' in f' {row["text"].lower()} ' for row in rows),
            'long_story_total': sum(row['format'] == 'long_story' for row in rows),
        },
        'rejections': dict(rejects),
        'word_count': {
            'min': min(row['word_count'] for row in rows if row['format'] == 'long_story'),
            'max': max(row['word_count'] for row in rows if row['format'] == 'long_story'),
            'mean': round(sum(row['word_count'] for row in rows if row['format'] == 'long_story') /
                          sum(row['format'] == 'long_story' for row in rows), 2),
        },
        'data_file': path.name,
        'sha256': digest,
        'seconds': round(time.time() - started, 2),
        'validation': 'Every row independently parsed with kinlogic.speech.read or kinlogic.rules.read and solved by kinlogic.engine.solve; one answer per row.',
        'scope': 'Training candidate supplement. No model training or locked-test evaluation is performed by this script.',
    }
    manifest_path = path.with_name(path.name.removesuffix('.jsonl.gz') + '.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('total', 'total_by_format', 'counts', 'word_count', 'sha256', 'seconds')}, indent=2))
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
