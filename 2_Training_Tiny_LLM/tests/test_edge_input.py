"""Website input edge cases: typos, scripts, apostrophes, HTML, word-names, and the GUARDS answers."""
from kinlogic import exam
from kinlogic.prepare import fix_kin_typos, prepare


def _ents(text, names=None):
    return [e['text'] for e in prepare(text, names=names)['ents']]


def test_kin_typos_fixed_only_in_relation_slots():
    assert fix_kin_typos('Raju is the brohter of Sita.') == 'Raju is the brother of Sita.'
    assert fix_kin_typos('Her mohter is Anu.') == 'Her mother is Anu.'
    assert fix_kin_typos('The brothel was closed.') == 'The brothel was closed.'   # a real word
    assert fix_kin_typos('Ravi is the other son.') == 'Ravi is the other son.'
    assert fix_kin_typos('Anu is the Mothr of Ravi.') == 'Anu is the Mothr of Ravi.'   # capitalised: may be a name


def test_names_in_any_script_and_with_apostrophes():
    assert _ents('José is the father of Zoë.') == ['José', 'Zoë']
    assert _ents('राजु is the father of सीता.') == ['राजु', 'सीता']
    assert _ents("D'Souza is the son of Ravi. Ravi's wife is Anu.") == ["D'Souza", 'Ravi', 'Anu']
    assert prepare("L is the aunt of I'G child.")['tokens'][5:8] == ['@1', "'", '@2']   # a page's "I'G": two letters


def test_word_names_in_person_slots():
    p = prepare('Will is the brother of May. Hope is the mother of Will. How is Hope related to May?')
    assert [e['text'] for e in p['ents']] == ['Will', 'May', 'Hope'] and p['tokens'][-3:] == ['to', '@1', '?']
    assert _ents("Who is the Lawyer's wife? Ravi is the son of Anu.") == ['Ravi', 'Anu']   # "the Lawyer": a description


def test_html_invisible_marks_and_codes():
    assert _ents('<b>Raju</b> is the father of <i>Sita</i> &amp; Kiran.') == ['Raju', 'Sita', 'Kiran']
    assert prepare('Ra​ju is the father of Sita.')['tokens'][0] == '@0'
    toks = prepare('If A <B means A is the father of B, what does P <Q mean?')['tokens']
    assert toks.count('<') == 2   # code symbols are not tags


def test_user_names_reach_the_answer():
    t = 'RAJU IS THE FATHER OF SITA. How is SITA related to RAJU?'
    read = lambda batch: [['@1 father @0 ; ask @0 @1'] for _ in batch]
    assert exam.answer(t, read, names=['RAJU', 'SITA'])['answer'] in ('daughter', 'child')


def test_guards(monkeypatch):
    monkeypatch.setattr(exam, 'GUARDS', True)
    never = lambda batch: [['@0 wife @1 ; ask @0 @1'] for _ in batch]   # a reader that guesses
    assert exam.answer('Raju and Sita went to the market. How is Sita related to Raju?', never)['status'] == 'none'
    assert exam.answer('Raju is the stepfather of Sita. How is Raju related to Sita?', never)['status'] == 'none'
    t = 'Raju is the father of Sita. Sita is the father of Raju. How is Sita related to Raju?'
    assert exam.answer(t, never)['status'] == 'contradiction'


def test_name_clash_counts_genders_against_clear_names():
    pri = {'@0': 'F', '@1': 'M', '@4': 'F'}
    assert exam._name_clashes('@0 sibling @1 ; @4 only son @0 ; ask @4 @0', pri) == 1   # "Barkha" made a son
    assert exam._name_clashes('@0 male ; @2 daughter @1', pri) == 2
    assert exam._name_clashes('@4 has 1 son ; @2 not son @0 ; @2 daughter @0', pri) == 0


def test_unnamed_chains_use_the_fewest_kin_words():
    from kinlogic.kin import name_moves
    label = lambda *mv: name_moves(list(mv))['label']
    assert label(('S', 'F'), ('U', 'M'), ('U', None), ('D', 'F')) == "father-in-law's sister"
    assert label(('U', 'M'), ('D', 'M'), ('D', 'M'), ('S', 'F')) == "nephew's wife"
    assert label(('U', 'F'), ('U', None), ('D', 'M'), ('D', 'M'), ('S', 'F')) == "maternal cousin's wife"
    assert label(('D', 'M'), ('S', 'F'), ('U', 'F')) == "daughter-in-law's mother"
    assert name_moves([('S', 'F'), ('U', 'M'), ('U', None), ('D', 'F')])['desc'] == "wife's father's sister"   # steps stay literal
