from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kinlogic.generate import sample
from model_reader import make_model
from reader import Vocab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', default='web/test/reader-parity')
    p.add_argument('--count', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261026)
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    vocab = Vocab.from_ck(ck)
    stoi = vocab.stoi
    model = make_model(ck['config']).eval()
    model.load_state_dict(ck['model'])
    model.c_id = stoi['<c>']
    rng = random.Random(a.seed)
    decodes = [stoi[w] for w in ('mother', 'father', 'sister', 'brother', ';') if w in stoi]
    cases, outputs = [], []
    attempts = 0
    with torch.no_grad():
        while len(cases) < a.count and attempts < a.count * 20:
            attempts += 1
            ex = sample(rng)
            ids = [stoi['<bos>'], *vocab.encode(ex['tokens']), stoi['<c>']]
            if len(ids) > ck['config']['block_size'] - 16 or len(ids) < 3:
                continue
            if len(cases) == a.count - 1 and len(ids) < 96:
                continue  # reserve the last case for a long prefix
            extra = rng.choice(decodes)
            prefix = torch.tensor([ids])
            h_prefix = model.hidden(prefix)[0]
            h_decode = model.hidden(torch.tensor([ids + [extra]]))[0, -1]
            logits = model.head(h_decode).float()
            copy_ids = ids[1:-1]
            keys = model.copy_k(h_prefix)[1:-1]
            logp = model._mix(torch.log_softmax(logits, -1)[None], h_decode[None], keys[None],
                              torch.tensor([copy_ids]), torch.ones((1, len(copy_ids)), dtype=torch.bool))[0]
            arrays = [h_prefix.numpy().ravel(), h_decode.numpy(), logits.numpy(), keys.numpy().ravel(), logp.numpy()]
            values = np.concatenate(arrays).astype('<f4', copy=False)
            cases.append({'ids': ids, 'extra': extra, 'P': len(ids), 'copy': len(copy_ids), 'count': len(values),
                          'kind': ex.get('kind', 'unknown')})
            outputs.append(values)
    if len(cases) != a.count:
        raise RuntimeError(f'only generated {len(cases)} of {a.count} prompts after {attempts} attempts')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'golden.bin').write_bytes(np.concatenate(outputs).astype('<f4', copy=False).tobytes())
    meta = {'vocab': len(vocab), 'C': ck['config']['n_embd'], 'DC': ck['model']['copy_q.weight'].shape[0],
            'cases': cases, 'seed': a.seed}
    (out / 'golden.json').write_text(json.dumps(meta, separators=(',', ':')) + '\n')
    print(f"wrote {len(cases)} deterministic prompts ({min(c['P'] for c in cases)}-{max(c['P'] for c in cases)} tokens) to {out}; seed {a.seed}")


if __name__ == '__main__':
    main()
