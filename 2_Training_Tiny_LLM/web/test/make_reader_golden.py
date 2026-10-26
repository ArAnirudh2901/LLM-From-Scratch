from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_reader import make_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', default='web/test/reader-golden')
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    vocab = ck['vocab']; stoi = {t: i for i, t in enumerate(vocab)}
    words = ['@0', 'is', 'the', 'father', 'of', '@1']
    ids = [stoi['<bos>']] + [stoi.get(w, stoi['<unk>']) for w in words] + [stoi['<c>']]
    token = stoi['mother']
    model = make_model(ck['config']).eval()
    model.load_state_dict(ck['model']); model.c_id = stoi['<c>']
    with torch.no_grad():
        prefix = torch.tensor([ids])
        h_prefix = model.hidden(prefix)[0]
        h_decode = model.hidden(torch.tensor([ids + [token]]))[0, -1]
        logits = model.head(h_decode).float()
        keys = model.copy_k(h_prefix)[1:-1]
        logp = model._mix(torch.log_softmax(logits, -1)[None], h_decode[None], keys[None],
                          prefix[:, 1:-1], torch.ones((1, len(words)), dtype=torch.bool))[0]
    arrays = [h_prefix.numpy().ravel(), h_decode.numpy(), logits.numpy(), keys.numpy().ravel(), logp.numpy()]
    values = np.concatenate(arrays).astype('<f4', copy=False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'golden.bin').write_bytes(values.tobytes())
    meta = {'ids': ids, 'extra': token, 'c': stoi['<c>'], 'vocab': len(vocab),
            'C': ck['config']['n_embd'], 'DC': ck['model']['copy_q.weight'].shape[0], 'P': len(ids),
            'words': words, 'count': len(values)}
    (out / 'golden.json').write_text(json.dumps(meta) + '\n')
    print(f"wrote {out}: prefix {len(ids)} tokens, {len(values)} reference floats")


if __name__ == '__main__':
    main()
