import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval.locked import check_once, freeze, hashes, record  # noqa: E402
from eval.run_real import run  # noqa: E402


def q(text, site):
    return {'question': text, 'options': [], 'site': site, 'answer_text': 'son'}


DEV = [q('Ravi is the father of Sita. Sita is the mother of Mohan. How is Ravi related to Mohan?', 'a')]


def test_freeze_keeps_new_sites_far_from_dev():
    new = [q('Kiran is the son of Asha. How is Kiran related to Asha?', 'a'),                            # a dev site
           q('Arun is the father of Gita. Gita is the mother of Vivek. How is Arun related to Vivek?', 'b'),  # dev, renamed
           q('Pointing to a girl, Meena said, "She is the daughter of my only brother." How is the girl related to Meena?', 'c')]
    kept, counts = freeze(new, DEV)
    assert [r['site'] for r in kept] == ['c'] and counts == {'dev site': 1, 'near a dev question': 1}


def test_locked_score_is_recorded_once(tmp_path):
    ledger = str(tmp_path / 'ledger.jsonl')
    check_once(None, ledger=ledger)   # never scored: fine
    record({'splits': {'locked2': '1/2 = 50.0%'}}, 'missing.pt', ['--split', 'locked2'], None, ledger=ledger)
    with pytest.raises(SystemExit):
        check_once(None, ledger=ledger)
    check_once('v11 selected on dev', ledger=ledger)
    with pytest.raises(SystemExit):
        check_once('partial', limit=10, ledger=ledger)
    assert json.loads(open(ledger).readline())['splits'] == {'locked2': '1/2 = 50.0%'}
    assert hashes(str(tmp_path / 'none.json')) == set()


def test_run_labels_the_locked_split():
    read = lambda batch: [[('@1 son @0 ; ask @1 @0', -0.1)] for _ in batch]
    score, _ = run(read, [q('Kiran is the son of Asha. How is Kiran related to Asha?', 'c')], split=lambda r: 'locked2')
    assert set(score['splits']) <= {'locked2', 'locked2 fair'} and score['splits']
