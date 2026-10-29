"""Per-question traces of the answer pipeline, for `eval.run_real --trace`: which route read the question (an exact
reader or the model), every reading the vote saw with the engine's result, whether any reading reached the key and
whether the vote picked it. The hooks only observe; answers are unchanged. `eval.trace_report` turns a trace into the
table of where points are lost.

Shadow mode (`--trace_shadow`) also answers every exactly-read question with the exact readers switched off, so a
confidently wrong exact reading can be told apart from a question the model would miss as well.
"""
from __future__ import annotations

import contextlib
import json
import re

import kinlogic.codes as codes
import kinlogic.exam as exam
import kinlogic.rules as rules
import kinlogic.speech as speech

KEEP = ('answer', 'option', 'status', 'facts', 'via', 'trace')
NONE_KEY = re.compile(r'none of|cannot be|can.?t be|inadequate|insufficient|no relation|not related|unrelated|stranger',
                      re.I)


def key_text(r) -> str:
    """The keyed option's text (or the answer text when the question has no options)."""
    return next((t for l, t in r.get('options') or [] if l == r.get('answer')), None) or r.get('answer_text') or ''


class Tracer:
    def __init__(self, path: str | None = None, shadow: bool = False):
        self.path, self.shadow = path, shadow
        self.out = open(path, 'w') if path else None
        self.records = []       # kept in memory when there is no path (tests)
        self._exact = {}        # tuple(tokens) -> exact reader that read them ('codes', 'rules', 'speech')
        self._off = False       # shadow pass: every exact reader returns None
        self._passes = []       # candidates() calls for the current question
        self._shadow_passes = []
        self._question = None   # prepared tokens of the current question
        self._undo = []

    # ---- hooks ----

    def install(self):
        for mod, name in ((codes, 'codes'), (rules, 'rules'), (speech, 'speech')):
            self._patch(mod, 'read', self._exact_hook(mod.read, name))
        self._patch(exam, 'candidates', self._candidates_hook(exam.candidates))
        return self

    def uninstall(self):
        for mod, attr, orig in reversed(self._undo):
            setattr(mod, attr, orig)
        self._undo = []

    def _patch(self, mod, attr, fn):
        self._undo.append((mod, attr, getattr(mod, attr)))
        setattr(mod, attr, fn)

    def _exact_hook(self, orig, name):
        def read(tokens, *a, **kw):
            if self._off:
                return None
            got = orig(tokens, *a, **kw)
            if got is not None:
                self._exact.setdefault(tuple(tokens), name)
            return got
        return read

    def _candidates_hook(self, orig):
        def candidates(text, read, priors=None):
            calls = []

            def model(batch):   # exam.candidates only calls the model when no exact reader took the question
                calls.append(len(batch))
                return read(batch)
            p, results = orig(text, model, priors)
            toks = tuple(p['tokens'])
            seen = self._exact.get(toks)
            (self._shadow_passes if self._off else self._passes).append({
                'original': toks == self._question,
                'route': 'model' if calls else 'exact',
                'exact_reader': None if calls else seen,
                'exact_rejected': seen if calls else None,   # an exact reader read it, but the reading did not fit
                'cands': [{**{k: res.get(k) for k in KEEP}, 'score': round(float(sc), 4)} for res, sc in results]})
            return p, results
        return candidates

    @contextlib.contextmanager
    def exact_off(self):
        self._off = True
        try:
            yield
        finally:
            self._off = False

    # ---- one question ----

    def begin(self, text: str):
        self._question = tuple(exam.prepare(text)['tokens'])
        self._passes, self._shadow_passes, self._exact = [], [], {}

    def end(self, r, res, ok: bool, solve=None, **meta) -> dict:
        """Record the question. solve(text) -> result: the run's answer function, used for the shadow pass."""
        from eval.run_real import as_text, correct
        for pa in self._passes:
            for c in pa['cands']:
                c['hit'] = bool(c['status'] != 'error' and correct(r, c))
        main = [pa for pa in self._passes if pa['original']]
        first = main[0] if main else {'route': None, 'exact_reader': None, 'exact_rejected': None}
        vkey = meta.get('verified_key')
        rec = {**meta, 'question': r['question'][:400], 'text': as_text(r), 'options': r.get('options') or [],
               'answer': r.get('answer'), 'answer_text': r.get('answer_text'), 'n_tokens': len(self._question or ()),
               'key': r.get('answer') or r.get('answer_text'),
               'key_text': key_text(r), 'key_is_none': bool(NONE_KEY.search(key_text(r))), 'ok': bool(ok),
               'ok_verified': None if vkey is None else res.get('option') == vkey,
               'final': {**{k: res.get(k) for k in KEEP}, 'confidence': res.get('confidence')},
               'route': first['route'], 'exact_reader': first['exact_reader'], 'exact_rejected': first['exact_rejected'],
               'n_cands': sum(len(pa['cands']) for pa in main),
               'hits': sum(c['hit'] for pa in main for c in pa['cands']),
               'oracle': bool(ok) or any(c['hit'] for pa in main for c in pa['cands']),
               'oracle_variants': any(c['hit'] for pa in self._passes if not pa['original'] for c in pa['cands']),
               'passes': self._passes, 'shadow': None}
        if self.shadow and solve is not None and rec['route'] == 'exact':
            with self.exact_off():
                got = solve(as_text(r))
            rec['shadow'] = {**{k: got.get(k) for k in KEEP}, 'confidence': got.get('confidence'),
                             'ok': bool(correct(r, got)), 'agrees': (got.get('option') or got.get('answer'))
                             == (res.get('option') or res.get('answer'))}
        if self.out:
            self.out.write(json.dumps(rec, separators=(',', ':')) + '\n')
            self.out.flush()   # Kaggle sessions can die: keep what is done
        else:
            self.records.append(rec)
        return rec

    def close(self):
        if self.out:
            self.out.close()
        self.uninstall()
