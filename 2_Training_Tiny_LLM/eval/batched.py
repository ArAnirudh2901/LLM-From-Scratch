"""Answer many questions with the reader batched across them, without changing the answering code.

Answering one question at a time decodes ~16 rows per step and leaves the GPU idle. Here every question runs until it
asks for a reading not decoded yet (Miss); all missing readings are then decoded together in large batches, and the
questions run again until none is missing. Beam readings depend only on (input, settings), so the answers are those
of the one-at-a-time loop (up to float noise from batch padding); samples are drawn per request, as before."""
from __future__ import annotations

import collections


class Miss(BaseException):   # not Exception: exam.answer turns any Exception into an 'error' answer
    pass


class Rounds:
    def __init__(self, gen, rows: int = 256):
        self.gen, self.rows = gen, rows   # rows: readings decoded at once (beam width x questions)
        self.cache, self.want, self.live = {}, {}, False
        self.seen = collections.Counter()

    def _key(self, toks, kw):
        sk = tuple(sorted((k, id(v) if callable(v) or hasattr(v, 'state_dict') else v) for k, v in kw.items()))
        base = (tuple(toks), sk)
        occ = 0
        if kw.get('sample'):   # a repeated sample request gets its own draw
            occ = self.seen[base]
            self.seen[base] += 1
        return base + (occ,)

    def __call__(self, batch, **kw):
        """Stands in for gen(batch, **kw)."""
        if self.live:
            return self.gen(batch, **kw)
        keys = [self._key(t, kw) for t in batch]
        miss = [(k, t) for k, t in zip(keys, batch) if k not in self.cache]
        if miss:
            for k, t in miss:
                self.want[k] = (t, kw)
            raise Miss()
        return [self.cache[k] for k in keys]

    def flush(self):
        """Decodes every wanted reading, grouped by settings. A batch holds about `rows` readings of a 300-token
        question: fewer when the questions are longer, so long questions cannot run the GPU out of memory."""
        groups = collections.defaultdict(list)
        for k, (t, kw) in self.want.items():
            groups[k[1]].append((k, t, kw))
        cap = self.rows * 300
        for items in groups.values():
            items.sort(key=lambda x: len(x[1]))   # similar lengths share a batch: less padding, the same budget as alone
            kw, width = items[0][2], max(1, items[0][2].get('k', 1))
            chunk = []
            for it in items + [None]:
                if chunk and (it is None or width * (len(chunk) + 1) * (len(it[1]) + 98) > cap):
                    for (k, _, _), g in zip(chunk, self.gen([t for _, t, _ in chunk], bs=len(chunk), **kw)):
                        self.cache[k] = g
                    chunk = []
                if it is not None:
                    chunk.append(it)
        self.want.clear()

    def memo(self, solve, texts):
        """solve, with `texts` answered first in batched rounds; any other text is answered one at a time."""
        uniq = list(dict.fromkeys(texts))
        got = dict(zip(uniq, self.answer_all(solve, uniq)))

        def run(t):
            if t in got:
                return got[t]
            self.live = True
            try:
                return solve(t)
            finally:
                self.live = False
        return run

    def answer_all(self, solve, texts, max_rounds: int = 16):
        """[solve(text)] for every text, the reader batched across texts."""
        out, todo = {}, list(range(len(texts)))
        for _ in range(max_rounds):
            nxt = []
            for i in todo:
                self.seen.clear()
                try:
                    out[i] = solve(texts[i])
                except Miss:
                    nxt.append(i)
            todo = nxt
            if not todo:
                break
            self.flush()
        self.live = True   # a question still asking for more after max_rounds: finish it one at a time
        try:
            for i in todo:
                out[i] = solve(texts[i])
        finally:
            self.live = False
        return [out[i] for i in range(len(texts))]
