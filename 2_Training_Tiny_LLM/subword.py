"""Sub-word pieces for words outside the word vocabulary, so no word is ever unknown: byte-pair merges (Sennrich et al.
2016) learned on our own corpus. A word splits into a first piece '▁ab' and continuation pieces '##cd'; every
character seen in training is a piece, so any word made of those characters encodes.

    merges = train(word_counts, n_merges=4000)       # [(left, right), ...] in merge order
    pieces(word, ranks(merges))                       # 'grandniece' -> ['▁grand', '##niece']
"""
from __future__ import annotations

import collections
import heapq


def _symbols(word: str) -> list[str]:
    return ['▁' + word[0]] + ['##' + c for c in word[1:]]


def _join(a: str, b: str) -> str:
    return a + b[2:]


def train(word_counts: dict[str, int], n_merges: int) -> list[tuple[str, str]]:
    """Merges in order: each joins the most frequent adjacent pair (ties: the pair seen first). Incremental pair counts,
    so a few thousand merges over a large word list take seconds."""
    words = [(_symbols(w), c) for w, c in word_counts.items() if w]
    pairs = collections.Counter()
    where = collections.defaultdict(set)
    for i, (s, c) in enumerate(words):
        for p in zip(s, s[1:]):
            pairs[p] += c
            where[p].add(i)
    heap = [(-c, p) for p, c in pairs.items()]
    heapq.heapify(heap)
    merges = []
    while heap and len(merges) < n_merges:
        c, p = heapq.heappop(heap)
        if pairs.get(p, 0) != -c or c == 0:   # stale entry
            continue
        merges.append(p)
        new = _join(*p)
        for i in list(where.pop(p, ())):
            s, wc = words[i]
            j, out = 0, []
            while j < len(s):
                if j + 1 < len(s) and (s[j], s[j + 1]) == p:
                    out.append(new)
                    j += 2
                else:
                    out.append(s[j])
                    j += 1
            for q in zip(s, s[1:]):
                pairs[q] -= wc
                where[q].discard(i)
            for q in zip(out, out[1:]):
                pairs[q] += wc
                where[q].add(i)
                heapq.heappush(heap, (-pairs[q], q))
            words[i] = (out, wc)
        pairs.pop(p, None)
    return merges


def ranks(merges) -> dict[tuple[str, str], int]:
    return {tuple(m): i for i, m in enumerate(merges)}


def pieces(word: str, rank: dict[tuple[str, str], int]) -> list[str]:
    """The word's pieces: repeatedly join the adjacent pair merged earliest in training."""
    s = _symbols(word)
    while len(s) > 1:
        best = min(((rank.get((a, b), 1 << 30), k) for k, (a, b) in enumerate(zip(s, s[1:]))), default=None)
        if best is None or best[0] == 1 << 30:
            break
        k = best[1]
        s = s[:k] + [_join(s[k], s[k + 1])] + s[k + 2:]
    return s


def alphabet(word_counts) -> list[str]:
    """Every single-character piece the words need."""
    out = set()
    for w in word_counts:
        out.update(_symbols(w))
    return sorted(out)
