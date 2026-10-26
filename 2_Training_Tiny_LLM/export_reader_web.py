"""Export a v10 reader as exact fp32 byte-shuffled weights for the browser runtime."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zlib
import shutil
from pathlib import Path

import numpy as np
import torch

from kinlogic import facts
from kinlogic.kin import KIN, NONKIN, ONLY, SIDED


def ordered_weights(state: dict, n_layer: int) -> np.ndarray:
    """Stable layout consumed by the v10 WebAssembly engine; tied output uses tok_emb.weight."""
    order = ['tok_emb.weight']
    for layer in range(n_layer):
        p = f'blocks.{layer}.'
        order += [p + 'ln1.weight', p + 'ln2.weight', p + 'attn.qkv.weight', p + 'attn.proj.weight',
                  p + 'attn.q_norm.weight', p + 'attn.k_norm.weight', p + 'ffn.w1.weight',
                  p + 'ffn.w3.weight', p + 'ffn.w2.weight']
    order += ['ln_f.weight', 'copy_q.weight', 'copy_k.weight', 'gate.weight', 'gate.bias']
    if set(order) != set(state) or len(order) != len(state):
        missing, extra = sorted(set(order) - set(state)), sorted(set(state) - set(order))
        raise ValueError(f'checkpoint parameters do not match the supported v10 layout; missing={missing}, extra={extra}')
    arrays = []
    for key in order:
        tensor = state[key]
        if tensor.dtype != torch.float32:
            raise ValueError(f'{key} is {tensor.dtype}; browser export requires exact fp32 weights')
        arrays.append(tensor.detach().cpu().contiguous().numpy().ravel())
    return np.concatenate(arrays).astype('<f4', copy=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default='web/v10')
    parser.add_argument('--verifier')
    parser.add_argument('--final')
    parser.add_argument('--name-table')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck['config']
    if cfg.get('arch') != 'v10' or not cfg.get('qk_norm') or not cfg.get('prefix_lm'):
        raise ValueError('expected the selected v10 reader with QK normalization and prefix-LM enabled')
    weights = ordered_weights(ck['model'], int(cfg['n_layer']))
    raw = weights.tobytes(order='C')
    shuffled = weights.view(np.uint8).reshape(-1, 4).T.copy().tobytes()
    (out / 'reader.bin.gz').write_bytes(gzip.compress(shuffled, compresslevel=9, mtime=0))
    (out / 'reader.bin').write_bytes(shuffled)  # fallback for browsers without streaming gzip support
    meta = {
        'arch': 'v10',
        'config': {k: cfg[k] for k in ('vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd', 'qk_norm', 'prefix_lm')},
        'rope_base': 10000.0,
        'rms_eps': 1e-6,
        'hidden_size': 1024,
        'copy': True,
        'd_copy': int(ck['model']['copy_q.weight'].shape[0]),
        'tied_output': True,
        'floats': int(weights.size),
        'crc32': zlib.crc32(raw),
        'vocab': ck['vocab'],
        'merges': ck.get('merges'),
        'special': {name: i for i, name in enumerate(ck['vocab'][:6])},
        'grammar': {
            'starters': facts._STARTERS, 'afterEntity': facts._AFTER_ENT, 'questions': facts.QUESTIONS,
            'puzzle': facts.PUZZLE, 'groups': facts.GROUPS, 'roles': facts.ROLES, 'profs': facts.PROFS,
            'attributes': facts.ATTRS, 'countable': facts.COUNTABLE, 'sides': facts.SIDES,
            'relations': facts.RELS, 'kin': list(KIN), 'nonkin': sorted(NONKIN), 'only': sorted(ONLY),
            'sided': sorted(SIDED), 'numbers': facts.NUMBERS, 'pair': facts._PAIR, 'maxNew': facts.MAX_NEW,
        },
        'checkpoint': {'step': int(ck.get('step', 0)), 'format': ck.get('format', 'v10-reader')},
    }
    (out / 'reader.json').write_text(json.dumps(meta, separators=(',', ':')) + '\n')
    names = ['reader.json', 'reader.bin.gz', 'reader.bin']
    for arg, name in ((args.verifier, 'verifier.json'), (args.final, 'FINAL.json'), (args.name_table, 'name_table.tsv.gz')):
        if arg:
            shutil.copyfile(arg, out / name)
            names.append(name)
    files = {name: {'bytes': (out / name).stat().st_size, 'sha256': sha256(out / name)} for name in names}
    (out / 'manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    print(f"reader: {weights.size / 1e6:.2f}M exact fp32 weights | raw {len(raw) / 1e6:.1f} MB | "
          f"gzip {(out / 'reader.bin.gz').stat().st_size / 1e6:.1f} MB | vocab {len(ck['vocab']):,} | "
          f"step {ck.get('step', 0)}")
    print(f'manifest: {out / "manifest.json"}')


if __name__ == '__main__':
    main()
