import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kinlogic.facts import allowed_next, parse  # noqa: E402
from kinlogic.prepare import prepare  # noqa: E402


def toks(text):
    return ' '.join(prepare(text)['tokens'])


def test_names_letters_descriptions():
    assert toks('Introducing Sanghoti to guests, Dinabandhu said, “Her father is the only son of my father”. '
                'How is Sanghoti related to Dinabandhu?') == (
        'introducing @0 to guests , @1 said , " her father is the only son of my father " . how is @0 related to @1 ?')
    assert toks('A woman introduces a man as the son of the brother of her mother. How is the man related to the woman?') == (
        'a woman @0 introduces a man @1 as the son of the brother of her mother . how is the man @1 related to the woman @0 ?')
    assert toks("Pointing to a photograph, Arun said, 'He is my son.' How is the man in the photograph related to Arun?").endswith(
        'how is the man @1 in the photograph related to @0 ?')


def test_options_and_boilerplate():
    r = prepare("Blood Relations Question 3: Santosh is the son of the brother of Varun's mother. How is Santosh related "
                'to Varun? 1. Maternal cousin 2. Aunt 3. Maternal grandmother 4. Mother')
    assert ' '.join(r['tokens']) == "@0 is the son of the brother of @1 's mother . how is @0 related to @1 ?"
    assert r['options'] == [('1', 'Maternal cousin'), ('2', 'Aunt'), ('3', 'Maternal grandmother'), ('4', 'Mother')]
    r = prepare('B is the son of C. Who is the mother of B? A D | B C | C B | D A')
    assert r['options'] == [('a', 'D'), ('b', 'C'), ('c', 'B'), ('d', 'A')]
    assert prepare('Who is X to Y? (a) Uncle (b) Cousin (c) Son')['options'] == [('a', 'Uncle'), ('b', 'Cousin'), ('c', 'Son')]


def test_letters_and_codes():
    assert toks('If A + B means A is the brother of B, what is C - D?') == (
        'if @0 + @1 means @0 is the brother of @1 , what is @2 - @3 ?')
    assert toks('A man said to A.') == 'a man @0 said to @1 .'
    assert toks('A is the father of B.') == '@0 is the father of @1 .'


def test_ambiguous_and_lowercase_names():
    assert toks('Will is the brother of Rose. Mr. Sharma is their father.') == '@0 is the brother of @1 . mr . @2 is their father .'
    assert toks('ravi is the son of mohan.') == '@0 is the son of @1 .'
    assert toks('On Sunday, Ravi met his Uncle.') == 'on sunday , @0 met his uncle .'


def test_grammar_only_allows_valid_facts():
    ents = ['@0', '@1']
    seq = '@1 mother #1 ; #1 brother #2 ; #2 son @0 ; ask @1 @0 ;'.split()
    for k, t in enumerate(seq):
        assert t in allowed_next(seq[:k], ents), (seq[:k], t)
    assert '<eos>' in allowed_next(seq, ents)
    assert '#3' not in allowed_next([], ents) and '#1' in allowed_next([], ents)
    assert '@1' not in allowed_next(['@1', 'mother'], ents)
    after_only = allowed_next(['@0', 'only'], ents)
    assert {'son', 'grandson', 'daughter-in-law', 'uncle', 'paternal', 'maternal'} <= set(after_only)
    assert 'friend' not in after_only and 'not' not in after_only
    assert 'uncle' in allowed_next(['@0', 'maternal'], ents) and 'son' not in allowed_next(['@0', 'maternal'], ents)
    assert parse(' '.join(seq))[1] == []


def test_compact_code_expressions():
    assert toks('If S-T^U, how is S related to U?') == 'if @0 - @1 ^ @2 , how is @0 related to @2 ?'
    assert toks('Ram is the father-in-law of Sita.') == '@0 is the father-in-law of @1 .'


def test_numbered_statements_are_not_options():
    r = prepare('What is the relation of Myra to Vivek? 1. Balaji is the brother of Vivek. 2. Myra has a husband named Balaji. '
                '(a) Only 1 is sufficient (b) Both 1 and 2 are sufficient')
    assert '@2' in r['tokens'] and r['options'] == [('a', 'Only 1 is sufficient'), ('b', 'Both 1 and 2 are sufficient')]
    r = prepare('A is the father of B. Which is true? I. B is a son. II. A is male. III. B is A\'s child. IV. A has a son. V. B is female.')
    assert [e['text'] for e in r['ents']] == ['A', 'B']
    assert toks('Hunter wrote to his mother Diana.') == '@0 wrote to his mother @1 .'


def test_lexicon_covers_generated_words():
    """Rebuild with `python -m kinlogic.lexicon` after changing the generator: an unknown sentence-initial word
    would be taken for a name."""
    import random
    import re
    from kinlogic import generate
    from kinlogic.prepare import LEXICON, TOKEN_RE
    rng = random.Random(1)
    missing = set()
    for _ in range(3000):
        ex = generate.sample(rng)
        names = {r.lower() for r in ex['names'].values()}
        for t in TOKEN_RE.findall(ex['text']):
            w = t.lower()
            if len(w) > 1 and re.fullmatch(r"[a-z][a-z'-]*", w) and all(len(x) > 1 for x in w.split('-')) and w not in names and w not in LEXICON and w not in {p.lower() for p in generate.PLACES}:
                missing.add(w)
    assert not missing, sorted(missing)[:20]


def test_pipe_options_riddlebench_style():
    r = prepare('A family consists of three members - G, K and L. Who is the mother of K? A.G | B.L | C.None of the above')
    assert r['options'] == [('a', 'G'), ('b', 'L'), ('c', 'None of the above')]
    assert r['tokens'][:2] == ['a', 'family']
    r = prepare('How is W related to K in J@K!T*W?X#Z. A. Sister|B. Daughter|C. Son')
    assert r['options'] == [('a', 'Sister'), ('b', 'Daughter'), ('c', 'Son')]
