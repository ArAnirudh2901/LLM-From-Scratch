"""English for pretraining our own reader from scratch (Furrer et al. 2020: pre-training beats specialized architectures
on compositional semantic parsing; TinyStories: models of a few million parameters learn fluent English from simple,
varied text). Sources: TinyStories V2 (GPT-4 stories, CDLA-Sharing-1.0) and FineWeb-Edu (ODC-By 1.0).

Text goes through the same preparation as exam questions (kinlogic.prepare), paragraph by paragraph, so the model
pretrains on the token stream it later reads: lowercase words, punctuation, and person placeholders @k. Only real
first names (the name table), single letters and descriptions ("a man") become placeholders; other capitalized words
(titles, places, organisations) stay words, else the model would learn that @k means any proper noun.

    python pretrain_data.py vocab --out data/subword_vocab.json            # word list + byte-pair merges
    python pretrain_data.py build --vocab data/subword_vocab.json --out DIR --tokens 400000000
    python pretrain_data.py build --mix domain=1 --tokens 60000000 --out ANNEAL_DIR     # kinship text for the decay phase
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import re
import sys
import urllib.request

import numpy as np

TINY = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt'
FINEWEB = 'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{i:03d}_00000.parquet'
PARA_WORDS = 160   # prepare() numbers at most 32 people; paragraphs keep well under that


def person_names(path='data/names/name_gender.tsv.gz', min_support=200) -> set[str]:
    out = set()
    with gzip.open(path, 'rt', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if int(r['support']) >= min_support:
                out.add(r['name'])
    return out


def paragraphs(text: str):
    """Chunks of at most PARA_WORDS words, split at blank lines, then at sentence ends."""
    for para in re.split(r'\n\s*\n', text):
        words = para.split()
        if not words:
            continue
        if len(words) <= PARA_WORDS:
            yield para
            continue
        cur = []
        for sent in re.split(r'(?<=[.?!])\s+', para):
            n = len(sent.split())
            if cur and sum(len(s.split()) for s in cur) + n > PARA_WORDS:
                yield ' '.join(cur)
                cur = []
            cur.append(sent)
        if cur:
            yield ' '.join(cur)


def tokens(text: str, names: set[str]) -> list[str]:
    """Prepared tokens of a document; placeholders renumbered per paragraph from @0."""
    from kinlogic.prepare import prepare
    out = []
    for para in paragraphs(text):
        p = prepare(para)
        keep = {e['id'] for e in p['ents'] if e['kind'] != 'name' or e['text'].lower() in names}
        word = {e['id']: e['text'].lower() for e in p['ents'] if e['id'] not in keep}
        renum = {}
        for t in p['tokens']:
            if t in word:
                out.append(word[t])
            elif t[0] == '@' and t in keep:
                out.append(renum.setdefault(t, f'@{len(renum)}'))
            else:
                out.append(t)
    return out


def tiny_stories(split='train', limit_bytes=None):
    req = urllib.request.Request(TINY.format(split=split), headers={'Range': f'bytes=0-{limit_bytes}'} if limit_bytes else {})
    buf = ''
    with urllib.request.urlopen(req) as r:
        for chunk in iter(lambda: r.read(1 << 20), b''):
            buf += chunk.decode('utf-8', 'ignore')
            *docs, buf = buf.split('<|endoftext|>')
            for d in docs:
                if d.strip():
                    yield d.strip()


def fineweb(shards=range(1), cache='data/pretrain_cache'):
    import pyarrow.parquet as pq
    os.makedirs(cache, exist_ok=True)
    for i in shards:
        path = f'{cache}/fineweb_{i:03d}.parquet'
        if not os.path.exists(path):
            urllib.request.urlretrieve(FINEWEB.format(i=i), path + '.part')
            os.replace(path + '.part', path)
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            for t in pf.read_row_group(g, columns=['text']).column('text').to_pylist():
                yield t


def domain(src='data/dataset_v2', extra=('data/reworded.jsonl.gz',)):
    """Kinship text already prepared (token lists): the verified pool's questions (never the locked test) and the
    verified rewrites, files interleaved line by line so any prefix mixes every kind and depth."""
    import glob
    files = [f for f in sorted(glob.glob(f'{src}/**/*.jsonl.gz', recursive=True)) if '/test/' not in f] + [e for e in extra if os.path.exists(e)]
    its = [gzip.open(f, 'rt') for f in files]
    while its:
        for it in list(its):
            line = it.readline()
            if not line:
                its.remove(it)
                continue
            yield json.loads(line)['tokens']


def build_vocab(a):
    """Words seen often enough keep one token each (plus every word the generator and the fact language use); rarer
    words get byte-pair pieces, so no word is unknown."""
    from reader import ENTS, NEWS, SPECIALS, build_vocab as task_vocab
    from kinlogic.facts import vocabulary
    from subword import alphabet, train
    names = person_names()
    counts = collections.Counter()
    for src, n in ((tiny_stories('train', a.tiny_bytes), a.tiny_docs), (fineweb(range(1)), a.web_docs)):
        for k, doc in enumerate(src):
            if k >= n:
                break
            counts.update(t for t in tokens(doc, names) if t[0] not in '@#' and t != '<unk>')
    task = set(task_vocab(extra=a.reworded).itos)
    fixed = SPECIALS + ENTS + NEWS + [t for t in vocabulary() if t != ';']
    common = [w for w, c in counts.most_common(a.words) if c >= a.min_count]
    words = (task | set(common)) - set(fixed)
    rare = {w: c for w, c in counts.items() if w not in words and w not in fixed and c >= 2}
    merges = train(rare, a.merges)
    pieces = set(alphabet(rare)) | {m[0] + m[1][2:] for m in merges}
    itos = fixed + sorted(words) + sorted(pieces - words - set(fixed))
    json.dump({'itos': itos, 'merges': merges, 'words': len(words), 'pieces': len(pieces), 'counted_tokens': sum(counts.values()),
               'sources': {'tinystories': 'roneneldan/TinyStories V2-GPT4 (CDLA-Sharing-1.0)',
                           'fineweb_edu': 'HuggingFaceFW/fineweb-edu sample-10BT (ODC-By 1.0)'}}, open(a.out, 'w'))
    print(f'{len(itos)} tokens: {len(words)} words, {len(pieces)} pieces, from {sum(counts.values())} counted tokens')


def _encode_docs(args):
    docs, vocab_path = args
    from reader import Vocab
    v = json.load(open(vocab_path))
    vocab = _encode_docs.vocab = getattr(_encode_docs, 'vocab', None) or Vocab(v['itos'], v['merges'])
    names = _encode_docs.names = getattr(_encode_docs, 'names', None) or person_names()
    bos = vocab.stoi['<bos>']
    out = []
    for d in docs:
        out.append(bos)
        out += vocab.encode(d if isinstance(d, list) else tokens(d, names))   # domain rows come prepared
    return np.asarray(out, dtype=np.uint16 if len(vocab) < 65536 else np.uint32)


def build(a):
    """Token shards (`shard_XXX.npy`, uint16 ids, <bos> between documents) mixing the sources by `--mix`."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(a.out, exist_ok=True)
    done = sorted(f for f in os.listdir(a.out) if f.startswith('shard_') and not f.endswith('.tmp.npy'))
    share = dict(kv.split('=') for kv in a.mix.split(','))
    srcs = {'tiny': tiny_stories('train'), 'web': fineweb(range(a.web_shards)), 'domain': domain()}
    srcs = {k: srcs[k] for k in share}
    plan = {k: int(a.tokens * float(v)) for k, v in share.items()}
    prog_path = os.path.join(a.out, 'progress.json')   # documents used and tokens written per source (exact resume)
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {'docs': {}, 'tokens': {}}
    got = collections.Counter(prog['tokens'])
    with ProcessPoolExecutor(a.workers) as pool:
        for k in plan:
            src = srcs[k]
            for _ in range(prog['docs'].get(k, 0)):
                next(src, None)
            while got[k] < plan[k]:
                batch = [[d for _, d in zip(range(200), src)] for _ in range(a.workers * 4)]
                batch = [b for b in batch if b]
                if not batch:
                    break
                arr = np.concatenate(list(pool.map(_encode_docs, [(b, a.vocab) for b in batch])))
                name = f'shard_{len(done):04d}_{k}.npy'
                np.save(os.path.join(a.out, name + '.tmp.npy'), arr)
                os.replace(os.path.join(a.out, name + '.tmp.npy'), os.path.join(a.out, name))
                done.append(name)
                got[k] += arr.size
                prog['docs'][k] = prog['docs'].get(k, 0) + sum(len(b) for b in batch)
                prog['tokens'] = dict(got)
                json.dump(prog, open(prog_path, 'w'))
                print(f'{k}: {got[k] / 1e6:.0f}M / {plan[k] / 1e6:.0f}M tokens', flush=True)
    json.dump({'tokens': dict(got), 'vocab': a.vocab, 'mix': a.mix}, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('vocab')
    v.add_argument('--out', default='data/subword_vocab.json')
    v.add_argument('--tiny_docs', type=int, default=200000)
    v.add_argument('--tiny_bytes', type=int, default=300_000_000, help='read only the start of the TinyStories file')
    v.add_argument('--web_docs', type=int, default=60000)
    v.add_argument('--words', type=int, default=20000)
    v.add_argument('--min_count', type=int, default=30)
    v.add_argument('--merges', type=int, default=6000)
    v.add_argument('--reworded', default='data/reworded.jsonl.gz')
    b = sub.add_parser('build')
    b.add_argument('--vocab', default='data/subword_vocab.json')
    b.add_argument('--out', default='data/pretrain')
    b.add_argument('--tokens', type=int, default=400_000_000)
    b.add_argument('--mix', default='tiny=0.5,web=0.5')
    b.add_argument('--web_shards', type=int, default=4)
    b.add_argument('--workers', type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    build_vocab(a) if a.cmd == 'vocab' else build(a)
