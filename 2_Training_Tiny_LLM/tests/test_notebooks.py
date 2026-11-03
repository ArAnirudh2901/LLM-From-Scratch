"""The Kaggle notebooks (colab/r9p .. r15) are generated code that only runs on a GPU: check every code cell for undefined names before a
run finds them (a bad edit once left `REPO, X` in every notebook's first cell)."""
import glob
import json
import os

import pytest

pyflakes_api = pytest.importorskip('pyflakes.api')
from pyflakes.messages import UndefinedName  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = sorted(glob.glob(os.path.join(HERE, 'colab/r*/*/notebook.ipynb')) + glob.glob(os.path.join(HERE, 'colab/r*/*_colab.ipynb')))


class Collect:
    def __init__(self):
        self.messages = []

    def flake(self, m):
        self.messages.append(m)

    def unexpectedError(self, filename, msg):
        self.messages.append(msg)

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.messages.append(f'syntax error line {lineno}: {msg}')


def test_notebooks_exist():
    assert len(NOTEBOOKS) >= 6


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: os.path.relpath(p, HERE))
def test_notebook_has_no_undefined_names(path):
    nb = json.load(open(path))
    source = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    out = Collect()
    pyflakes_api.check(source, path, out)
    bad = [str(m) for m in out.messages if isinstance(m, (UndefinedName, str))]
    assert not bad, bad
