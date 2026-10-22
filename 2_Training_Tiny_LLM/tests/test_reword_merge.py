from kinlogic.prepare import prepare
from kinlogic.reword_merge import check, structure


def row(text, lf, truth):
    return {'text': text, 'tokens': prepare(text)['tokens'], 'lf': lf, 'truth': truth}


def test_structure_ignores_relations_and_genders():
    assert structure('ask @1 @0 ; part 1 ; @2 wife @1 ; part 2 ; @2 father @0') == structure('ask @1 @0 ; part 1 ; @2 husband @1 ; @1 male ; part 2 ; @2 son @0')
    assert structure('ask @1 @0 ; part 1 ; @2 wife @1') != structure('@2 wife @1 ; ask @1 @0')


def test_sufficiency_rewrite_without_parts_is_dropped():
    src = row('How is Rakhi related to Jayesh? Statement I: Jayesh has a brother named Hemant. Statement II: Hemant is the husband of Rakhi.',
              'ask @1 @0 ; part 1 ; @1 brother @2 ; part 2 ; @2 wife @0', 'sister-in-law')
    new = row('How is Rakhi connected to Jayesh? Statement I: Jayesh has a brother named Hemant. Statement II: Hemant is married to Rakhi.',
              '@1 brother @2 ; @2 wife @0 ; ask @1 @0', 'sister-in-law')
    assert check((new, src)) == 'structure lost'


def test_gender_needs_a_cue_or_the_source():
    src = row('Esha is the wife of Sagar. Pratibha is the mother of Esha. How is Pratibha related to Sagar?',
              '@0 wife @1 ; @1 mother @2 ; ask @0 @2', 'mother-in-law')
    guessed = row("Sagar's spouse is Esha. Pratibha is the mother of Esha. How is Pratibha related to Sagar?",
                  '@0 spouse @1 ; @1 mother @2 ; @1 female ; ask @0 @2', 'mother-in-law')
    cued = row("Sagar's spouse is Esha, and she is the daughter of Pratibha. How is Pratibha related to Sagar?",
               '@0 spouse @1 ; @1 mother @2 ; @1 female ; ask @0 @2', 'mother-in-law')
    assert check((guessed, src)) == 'gender from name'
    assert check((cued, src)) == 'kept'
