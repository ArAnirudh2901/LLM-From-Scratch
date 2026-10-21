import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kinlogic import format, parse, solve  # noqa: E402


def answer(lf, q=0):
    stmts, bad = parse(lf)
    assert not bad, bad
    return solve(stmts)['questions'][q]


def keys(lf, q=0):
    return [a['key'] for a in answer(lf, q)['answers']]


def one(lf):
    v = answer(lf)
    assert v['status'] == 'unique', v
    return v['answers'][0]


# ---------- the user's questions ----------

def test_santosh_son_of_brother_of_varuns_mother():
    # "Santosh is the son of the brother of Varun's mother. How is Santosh related to Varun?"
    a = one('@1 mother #1 ; #1 brother #2 ; #2 son @0 ; ask @1 @0')
    assert a['key'] == 'maternal cousin' and a['desc'] == "mother's brother's son" and a['hindi'] == 'mamera bhai'


def test_sanghoti_only_son_of_my_father():
    # Introducing Sanghoti, Dinabandhu said, "Her father is the only son of my father."
    v = answer('@0 female ; @0 father #1 ; @1 father #2 ; #2 only son #1 ; ask @1 @0')
    assert v['status'] == 'conditional'
    first, second = v['answers']
    assert first['key'] == 'daughter' and first['conds'] == ['@1 is a man']
    assert second['key'] == 'niece' and second['conds'] == ['@1 is a woman']


def test_woman_introduces_man():
    # A woman introduces a man as the son of the brother of her mother.
    assert one('@0 female ; @1 male ; @0 mother #1 ; #1 brother #2 ; #2 son @1 ; ask @0 @1')['key'] == 'maternal cousin'


# ---------- exam patterns ----------

def test_statements_force_gender():
    # A is the brother of B. B is the daughter of C. D is the father of A. How is C related to D?
    a = one('@1 brother @0 ; @2 daughter @1 ; @0 father @3 ; ask @3 @2')
    assert a['key'] == 'wife' and a['conds'] == []


def test_photo_only_son_of_my_grandfather():
    # Pointing to a photo, Ravi said, "He is the son of the only son of my grandfather."
    v = answer('@0 grandfather #1 ; #1 only son #2 ; #2 son @1 ; ask @0 @1')
    assert [a['key'] for a in v['answers']] == ['brother', 'maternal cousin']   # fewer people first


def test_only_daughter_of_my_mother_is_me():
    v = one('@0 female ; @1 male ; @1 mother #1 ; @0 mother #2 ; #2 only daughter #1 ; ask @0 @1')
    assert v['key'] == 'son'


def test_no_brothers_or_sisters():
    # Looking at a portrait, Harsh said, "His mother is the wife of my father's son. I have no brothers or sisters."
    assert one('@1 male ; @1 mother #1 ; @0 father #2 ; #2 son #3 ; #3 wife #1 ; @0 no-sibling ; ask @0 @1')['key'] == 'son'


def test_fathers_son_without_only_is_a_brother():
    assert one('@0 male ; @0 father #1 ; #1 son #2 ; #2 son @1 ; ask @0 @1')['key'] == 'nephew'


def test_gender_not_given():
    v = answer('@1 child @0 ; ask @0 @1')
    assert v['status'] == 'gender' and v['neutral'] == 'parent'
    assert one('@1 child @0 ; ask @1 @0')['key'] == 'child'
    v = answer('@0 sibling @1 ; ask @0 @1')
    assert v['status'] == 'unique' and v['answers'][0]['key'] == 'sibling'


def test_who_and_negation_puzzle():
    # Six members. B is the son of C but C is not the mother of B. A and C are a married couple.
    # E is the brother of C. D is the daughter of A. F is the brother of B. Who is the mother of B?
    lf = ('members 6 ; @2 son @1 ; @1 not mother @2 ; @0 spouse @2 ; @2 brother @4 ; @0 daughter @3 ; '
          '@1 brother @5 ; who @1 mother')
    v = answer(lf)
    assert v['status'] == 'unique' and v['answers'][0]['names'] == ['@0']


def test_counts():
    lf = ('@0 wife @1 ; @0 son @2 ; @0 daughter @3 ; @2 wife @4 ; @4 son @5 ; '
          'count male ; count female ; count @0 child ; count couples')
    stmts, _ = parse(lf)
    qs = solve(stmts)['questions']
    assert [q['answers'][0]['key'] for q in qs] == ['3', '3', '2', '2']


def test_opposite_gender_link():
    assert one('@0 child @1 ; @1 opposite @2 ; @2 female ; ask @0 @1')['key'] == 'son'
    assert one('@0 child @1 ; @0 opposite @2 ; @2 female ; ask @1 @0')['key'] == 'father'
    assert answer('@0 child @1 ; @0 opposite @1 ; ask @1 @0')['status'] == 'gender'


def test_contradictions():
    assert answer('@0 father @1 ; @1 father @0 ; ask @0 @1')['status'] == 'contradiction'
    assert answer('@0 male ; @0 female ; @0 son @1 ; ask @0 @1')['status'] == 'contradiction'
    assert answer('@0 brother @1 ; @1 wife @0 ; ask @0 @1')['status'] == 'contradiction'


def test_not_family_and_no_link():
    assert one('@0 friend @1 ; ask @0 @1')['key'] == 'not family: friend'
    assert answer('@0 son @1 ; @2 daughter @3 ; ask @0 @3')['status'] == 'none'


# ---------- every relation name ----------

NAMES = [
    ('@0 father @1', 'father', 'pita'), ('@0 mother @1', 'mother', 'maa'),
    ('@0 son @1', 'son', 'beta'), ('@0 daughter @1', 'daughter', 'beti'),
    ('@0 brother @1', 'brother', 'bhai'), ('@0 sister @1', 'sister', 'behen'),
    ('@0 husband @1', 'husband', 'pati'), ('@0 wife @1', 'wife', 'patni'),
    ('@0 father #1 ; #1 father @1', 'paternal grandfather', 'dada'),
    ('@0 father #1 ; #1 mother @1', 'paternal grandmother', 'dadi'),
    ('@0 mother #1 ; #1 father @1', 'maternal grandfather', 'nana'),
    ('@0 mother #1 ; #1 mother @1', 'maternal grandmother', 'nani'),
    ('@0 father #1 ; #1 father #2 ; #2 father @1', 'paternal great-grandfather', 'pardada'),
    ('@0 mother #1 ; #1 mother #2 ; #2 mother @1', 'maternal great-grandmother', 'parnani'),
    ('@0 father #1 ; #1 father #2 ; #2 father #3 ; #3 father @1', 'paternal great-great-grandfather', None),
    ('@0 son #1 ; #1 son @1', 'grandson', 'pota'), ('@0 daughter #1 ; #1 daughter @1', 'granddaughter', 'natin'),
    ('@0 son #1 ; #1 son #2 ; #2 daughter @1', 'great-granddaughter', None),
    ('@0 father #1 ; #1 brother @1', 'paternal uncle', 'chacha or tau'),
    ('@0 father #1 ; #1 sister @1', 'paternal aunt', 'bua'),
    ('@0 mother #1 ; #1 brother @1', 'maternal uncle', 'mama'),
    ('@0 mother #1 ; #1 sister @1', 'maternal aunt', 'mausi'),
    ('@0 mother #1 ; #1 brother #2 ; #2 wife @1', 'maternal aunt (by marriage)', 'mami'),
    ('@0 father #1 ; #1 sister #2 ; #2 husband @1', 'paternal uncle (by marriage)', 'phupha'),
    ('@0 father #1 ; #1 father #2 ; #2 brother @1', 'paternal great-uncle', None),
    ('@0 brother #1 ; #1 son @1', 'nephew', 'bhatija'), ('@0 sister #1 ; #1 daughter @1', 'niece', 'bhanji'),
    ('@0 brother #1 ; #1 son #2 ; #2 son @1', 'grand-nephew', None),
    ('@0 father #1 ; #1 brother #2 ; #2 son @1', 'paternal cousin', 'chachera bhai'),
    ('@0 mother #1 ; #1 sister #2 ; #2 daughter @1', 'maternal cousin', 'mausera behen'),
    ('@0 father #1 ; #1 father #2 ; #2 brother #3 ; #3 son #4 ; #4 son @1', 'paternal second-cousin', None),
    ('@0 wife #1 ; #1 father @1', 'father-in-law', 'sasur'), ('@0 husband #1 ; #1 mother @1', 'mother-in-law', 'saas'),
    ('@0 daughter #1 ; #1 husband @1', 'son-in-law', 'damaad'), ('@0 son #1 ; #1 wife @1', 'daughter-in-law', 'bahu'),
    ('@0 brother #1 ; #1 wife @1', 'sister-in-law', 'bhabhi'), ('@0 sister #1 ; #1 husband @1', 'brother-in-law', 'jija'),
    ('@0 male ; @0 wife #1 ; #1 brother @1', 'brother-in-law', 'saala'),
    ('@0 female ; @0 husband #1 ; #1 sister @1', 'sister-in-law', 'nanad'),
    ('@0 wife #1 ; #1 sister #2 ; #2 husband @1', 'co-brother', None),
    ('@0 son #1 ; #1 son #2 ; #2 wife @1', 'granddaughter-in-law', None),
    ('@0 wife #1 ; #1 brother #2 ; #2 son @1', 'nephew (by marriage)', None),
]


@pytest.mark.parametrize('facts,label,hindi', NAMES)
def test_relation_names(facts, label, hindi):
    a = one(f'{facts} ; ask @0 @1')
    assert a['label'] == label
    assert a['hindi'] == hindi


def test_words_compose_through_facts():
    # composite words in the facts themselves
    assert one('@0 maternal uncle #1 ; #1 son @1 ; ask @0 @1')['key'] == 'maternal cousin'
    assert one('@0 paternal grandfather #1 ; #1 wife @1 ; ask @0 @1')['key'] == 'paternal grandmother'
    assert one('@0 father-in-law #1 ; #1 daughter @1 ; @0 male ; ask @0 @1')['key'] in ('wife', 'sister-in-law')
    assert one('@0 great-grandfather @1 ; ask @1 @0')['key'] == 'great-grandchild'


def test_unnamed_people_are_new_unless_only():
    # a father-in-law's daughter is another person (sister-in-law), like a father's son is a brother;
    # "only daughter" makes her the wife
    assert one('@0 male ; @0 father-in-law #1 ; #1 daughter @1 ; ask @0 @1')['key'] == 'sister-in-law'
    assert one('@0 male ; @0 father-in-law #1 ; #1 only daughter @1 ; ask @0 @1')['key'] == 'wife'


def test_brother_in_law_two_meanings():
    # wife's brother's wife (co-sister) or sister's husband's wife (the sister herself)
    assert set(keys('@0 brother-in-law @1 ; @1 wife @2 ; ask @0 @2')) == {'sister', 'co-sister'}


def test_format_round_trip():
    lf = '@1 not only son @2 ; part 1 ; who @0 maternal uncle ; count @0 son ; members 6 ; @3 no-sibling'
    stmts, bad = parse(lf)
    assert not bad and parse(format(stmts))[0] == stmts


def test_rejects_bad_statements():
    stmts, bad = parse('@0 father @0 ; @0 only uncle @1 ; @0 paternal son @1 ; ask @0 ; blah')
    assert not stmts and len(bad) == 5


# ---------- question types from bank, SSC, CAT, CLAT and UPSC CSAT papers ----------

def solved(lf, **kw):
    stmts, bad = parse(lf)
    assert not bad, bad
    return solve(stmts, **kw)


def test_multi_statement_sufficiency():
    # What is the relation of E to B? 1. A and B are a married couple. 2. D is the father of C.
    # 3. E is D's son. 4. A and C are sisters.
    lf = ('ask @1 @4 ; part 1 ; @0 spouse @1 ; part 2 ; @2 father @3 ; part 3 ; @3 son @4 ; '
          'part 4 ; @0 sister @2 ; @2 sister @0')
    full = solved(lf)['questions'][0]
    assert full['status'] == 'unique' and full['answers'][0]['key'] == 'brother-in-law'
    assert solved(lf, parts={1, 2, 3})['questions'][0]['status'] == 'none'


def test_gender_question_and_negation():
    # M is the brother of N. B is the brother of N. M is the brother of D. But N is not the brother of B.
    lf = '@3 brother @0 ; @3 brother @1 ; @2 brother @0 ; @1 not brother @3 ; gender @3 ; ask @1 @3'
    q = solved(lf)['questions']
    assert q[0]['answers'][0]['key'] == 'female' and q[1]['answers'][0]['key'] == 'sister'


def test_has_counts():
    assert one('@0 female ; @0 has 0 son ; @0 child @1 ; ask @0 @1')['key'] == 'daughter'
    assert solved('@0 has 2 son ; @0 son @1 ; count @0 son')['questions'][0]['answers'][0]['key'] == '2'
    assert not solved('@0 has 1 son ; @0 son @1 ; @0 son @2 ; ask @0 @1')['ok']


def test_gender_totals_and_generations():
    # Four members, two men and two women. A is B's father, C is A's wife, D is B's sister.
    lf = 'members 4 ; males 2 ; females 2 ; @1 father @0 ; @0 wife @2 ; @1 sister @3 ; gender @1'
    assert solved(lf)['questions'][0]['answers'][0]['key'] == 'male'
    assert solved('@0 father @1 ; @1 father @2 ; generations 3 ; ask @0 @2')['ok']
    assert not solved('@0 father @1 ; @1 father @2 ; generations 2 ; ask @0 @2')['ok']


def test_closed_family_puzzle_with_couples():
    # Six members, two married couples. D is A's grandmother and B's mother. C is B's wife and F's mother.
    # F is E's granddaughter. How is C related to A?
    lf = ('members 6 ; couples 2 ; @0 grandmother @3 ; @1 mother @3 ; @1 wife @2 ; @5 mother @2 ; '
          '@4 granddaughter @5 ; ask @0 @2')
    v = solved(lf)['questions'][0]
    assert v['answers'][0]['key'] == 'mother'


def test_claims_definitely_true():
    lf = ('@0 son @1 ; @0 son @2 ; @0 wife @3 ; claim 1 ; @1 mother @3 ; claim 2 ; @3 male ; '
          'claim 3 ; @0 son #1 ; #1 brother @2 ; claim 4 ; @1 sister @2')
    assert solved(lf)['claims'] == {1: 'true', 2: 'false', 3: 'true', 4: 'false'}
    assert solved('@0 child @1 ; claim 1 ; @1 male')['claims'] == {1: 'possible'}


def test_pairs():
    from kinlogic.engine import pair_holds
    r = solved('@0 wife @1 ; @0 son @2 ; @0 daughter @3 ; @1 mother @4')
    w, ids = r['worlds'][0], r['ids']
    assert pair_holds(w, 'couple', ids['@0'], ids['@1'])
    assert pair_holds(w, 'female', ids['@1'], ids['@3'])
    assert not pair_holds(w, 'female', ids['@1'], ids['@2'])
    assert pair_holds(w, 'sibling', ids['@2'], ids['@3'])


def test_coded_chain_with_in_law_codes():
    # T@X: T is mother-in-law of X; T%X: child; T*X: parent; T#X: sibling; T!X: daughter-in-law; T&X: married.
    # If B#R!P*S&H, how is R's husband related to H?
    lf = '@1 sibling @0 ; @2 daughter-in-law @1 ; @3 parent @2 ; @3 spouse @4 ; @1 husband #1 ; ask @4 #1'
    keys_ = [a['key'] for a in answer(lf)['answers']]
    assert keys_[0] in ('brother-in-law',) and 'brother-in-law' in keys_


def test_narrator_without_names():
    # How is the father of my mother's sister's son related to me?
    a = one('#1 mother #2 ; #2 sister #3 ; #3 son #4 ; #4 father #5 ; ask #1 #5')
    assert a['label'] == 'maternal uncle (by marriage)' and a['hindi'] == 'mausa'
