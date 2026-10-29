import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval.mine_patterns import features, mine, skeleton, text_of  # noqa: E402

SPEECH = {'question': 'Pointing to Ravi, Sita said, "He is the only son of my mother\'s mama." How is Ravi related to Sita?',
          'options': [['a', 'Uncle'], ['b', 'Cousin'], ['c', 'Brother'], ['d', 'Cannot be determined']], 'answer': 'b'}
CODED = {'question': 'If A + B means A is the father of B and A × B means A is the sister of B, how is P related to R in P + Q × R?',
         'options': [], 'answer_text': 'Father'}


def test_features_name_the_exam_forms():
    f = features(SPEECH)
    assert {'pointing / showing', 'direct speech (quotes)'} <= f['opening']
    assert 'how is X related to Y' in f['question form']
    assert {'only', 'first person (my / I / me)', 'Hindi kin words', "possessive chain (X's R's R)"} <= f['construction'] or \
        {'only', 'first person (my / I / me)', 'Hindi kin words'} <= f['construction']
    assert f['answer'] == {'cousin'} and f['options'] == {'<= 4'}
    c = features(CODED)
    assert 'coded symbols ("A + B means")' in c['opening'] and 'expression with 2 operators' in c['coded']


def test_skeleton_hides_people():
    assert skeleton(SPEECH['question']) == 'how is @ related to @ ?'


def test_report_has_no_question_text_and_flags_gaps():
    real = [SPEECH, CODED] * 3
    gen = [{'text': 'A is the father of B. How is B related to A?', 'truth': 'son'}] * 4
    rep = mine(real, gen, trace={text_of(SPEECH): False}, min_skeleton=3)
    dump = json.dumps(rep)
    assert 'Ravi' not in dump and 'Sita' not in dump and 'only son of my' not in dump
    gaps = {(g['family'], g['feature']) for g in rep['gaps']}
    assert ('opening', 'pointing / showing') in gaps and ('construction', 'Hindi kin words') in gaps
    assert rep['features']['opening']['pointing / showing']['missed'] == 3
    assert rep['skeletons'][0] == {'skeleton': 'how is @ related to @ ?', 'real': 3, 'gen': 4}
