from __future__ import annotations
import argparse, gzip, hashlib, json, os, zlib

import numpy as np
import torch

from fast_qa import PERMS
from qa_data import EDGE_RELS, ENTS, NOFACT, RELATIONS, UNKNOWN

# Exports the QA model and the name tagger for the browser engine (web/engine.c): exact fp32 weights in the
# engine's order, byte-shuffled and gzipped (lossless; the browser undoes both), plus vocab and settings JSON.


def weights(state: dict, n_layer: int) -> np.ndarray:
    order = ['tok_emb.weight', 'pos_emb.weight']
    for l in range(n_layer):
        p = f'blocks.{l}.'
        order += [p + 'ln1.weight', p + 'ln1.bias', p + 'attn.qkv.weight', p + 'attn.proj.weight', p + 'ln2.weight',
                  p + 'ln2.bias', p + 'ffn.net.0.weight', p + 'ffn.net.0.bias', p + 'ffn.net.2.weight', p + 'ffn.net.2.bias']
    order += ['ln_f.weight', 'ln_f.bias', 'head.weight']
    assert set(order) == set(state), set(state) ^ set(order)
    return np.concatenate([state[k].float().cpu().numpy().ravel() for k in order]).astype('<f4')


def write(out: str, name: str, w: np.ndarray, meta: dict):
    raw = w.tobytes()
    shuffled = w.view(np.uint8).reshape(-1, 4).T.copy().tobytes()   # byte planes compress better; lossless
    with open(f'{out}/{name}.bin.gz', 'wb') as f:
        f.write(gzip.compress(shuffled, 9))
    with open(f'{out}/{name}.bin', 'wb') as f:   # for browsers without DecompressionStream
        f.write(shuffled)
    with open(f'{out}/{name}.json', 'w') as f:
        json.dump({**meta, 'floats': int(w.size), 'crc32': zlib.crc32(raw)}, f)   # checked by kinship.js at load
    print(f'{name}: {w.size / 1e6:.2f}M weights | raw {len(raw) / 1e6:.1f} MB | download {os.path.getsize(f"{out}/{name}.bin.gz") / 1e6:.1f} MB')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qa', default='runs/clutrr-qa-v8/best.pt')
    p.add_argument('--tagger', default='runs/name-tagger-v4/tagger.pt')
    p.add_argument('--out', default='web/model')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.qa, map_location='cpu')
    c = ck['config']
    write(args.out, 'qa', weights(ck['model'], c['n_layer']), {
        'config': {k: c[k] for k in ('vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd')},
        'vocab': ck['vocab'], 'relations': RELATIONS, 'edge_relations': EDGE_RELS, 'entities': ENTS,
        'unknown': UNKNOWN, 'nofact': NOFACT, 'perms': PERMS})

    tg = torch.load(args.tagger, map_location='cpu')
    c = tg['config']
    write(args.out, 'tagger', weights(tg['model'], c['n_layer']), {
        'config': {k: c[k] for k in ('vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd')}, 'vocab': tg['vocab']})
    write_manifest([args.qa, args.tagger] + sorted(f'{args.out}/{f}' for f in os.listdir(args.out)))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_manifest(paths: list[str], out: str = 'model_manifest.json'):
    """SHA-256 and size of every model file, for verify_model.py after copying or moving."""
    entries = {p: {'bytes': os.path.getsize(p), 'sha256': sha256(p)} for p in paths}
    with open(out, 'w') as f:
        json.dump(entries, f, indent=1)
    print(f'{out}: {len(entries)} files')


if __name__ == '__main__':
    main()
