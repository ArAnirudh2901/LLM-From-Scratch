"""Name -> gender model: a small character CNN that reads a first name's spelling and gives P(female), so names never
seen in any list still get a gender (suffixes like -preet, -amma, -appa, -bhai/-ben, spelling variants). ~95k
parameters, exact fp32 (~0.4 MB): microseconds per name on any CPU, easy to run in the browser.

It decides only when sure: male if p <= t_male, female if p >= t_female, else the gender stays open (a wrong gender
flips an answer; an open one does not). Thresholds are picked on validation names so wrong genders stay <= `--max_wrong`.

    python name_model.py train --data data/names/name_gender.tsv.gz --out runs/names-v1   (Colab: --device cuda)
    python name_model.py eval  --ckpt runs/names-v1/name_model.pt
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = "^$-'abcdefghijklmnopqrstuvwxyz"
STOI = {c: i + 1 for i, c in enumerate(CHARS)}   # 0 = padding
MAX_LEN = 24


def encode(name: str) -> list[int]:
    s = '^' + ''.join(c for c in name.lower() if c in STOI and c not in '^$') + '$'
    if len(s) > MAX_LEN:   # keep the start and the ending, where gender usually shows
        s = s[:MAX_LEN // 2] + s[-MAX_LEN // 2:]
    return [STOI[c] for c in s]


class NameModel(nn.Module):
    def __init__(self, d: int = 32, ch: int = 64, widths=(2, 3, 4, 5), hidden: int = 128):
        super().__init__()
        self.cfg = dict(d=d, ch=ch, widths=list(widths), hidden=hidden)
        self.emb = nn.Embedding(len(CHARS) + 1, d, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(d, ch, w, padding=w - 1) for w in widths])
        self.fc = nn.Linear(2 * ch * len(widths), hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):   # x: (B, T) ids, 0 = padding -> logits (B,)
        mask = (x > 0).float()
        e = self.emb(x).transpose(1, 2)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(e))[:, :, :x.size(1)]
            m = mask[:, None, :h.size(2)]
            feats.append((h - 1e4 * (1 - m)).max(2).values)
            feats.append((h * m).sum(2) / m.sum(2).clamp(min=1))
        return self.out(F.relu(self.fc(torch.cat(feats, 1)))).squeeze(1)


def batch_ids(names, device):
    ids = [encode(n) for n in names]
    T = max(len(i) for i in ids)
    x = torch.zeros(len(ids), T, dtype=torch.long)
    for k, i in enumerate(ids):
        x[k, :len(i)] = torch.tensor(i)
    return x.to(device)


def load_table(path):
    rows = {'train': [], 'val': [], 'test': []}
    with gzip.open(path, 'rt', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows[r['split']].append((r['name'], float(r['p_female']), int(r['support'])))
    return rows


@torch.no_grad()
def predict(model, names, device, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(names), bs):
        out += torch.sigmoid(model(batch_ids(names[i:i + bs], device))).float().cpu().tolist()
    return out


def decide(p, t_male, t_female):
    return 'M' if p <= t_male else 'F' if p >= t_female else None


def pick_thresholds(ps, gold, max_wrong):
    """Most names decided with wrong genders <= max_wrong of the clear names; t_male < 0.5 < t_female, grid 0.01."""
    p = torch.tensor(ps)
    f = torch.tensor([g == 'F' for g in gold])
    grid = torch.arange(1, 50) / 100
    le = (p[None, :] <= grid[:, None])             # decided male at t_male = grid[i]
    ge = (p[None, :] >= (1 - grid)[:, None])       # decided female at t_female = 1 - grid[j]
    right = (le & ~f).sum(1)[:, None] + (ge & f).sum(1)[None, :]
    wrong = (le & f).sum(1)[:, None] + (ge & ~f).sum(1)[None, :]
    right = torch.where(wrong <= max_wrong * len(gold), right, torch.full_like(right, -1))
    i, j = divmod(int(right.argmax()), len(grid))
    return round(float(grid[i]), 2), round(float(1 - grid[j]), 2)   # float32 grid values would miss p == t


def clear(rows, lo=0.1, hi=0.9, min_support=20):
    """Names whose data gender is clear and well supported: (name, 'M'|'F')."""
    return [(n, 'F' if p >= hi else 'M') for n, p, s in rows if s >= min_support and (p >= hi or p <= lo)]


def report(model, rows, device, t_male, t_female):
    out = {}
    for name, items in rows.items():
        ps = predict(model, [n for n, _ in items], device)
        dec = [decide(p, t_male, t_female) for p in ps]
        g = [x for _, x in items]
        out[name] = {'n': len(items), 'right': round(sum(d == x for d, x in zip(dec, g)) / max(1, len(g)), 4),
                     'wrong': round(sum(d is not None and d != x for d, x in zip(dec, g)) / max(1, len(g)), 4)}
    return out


def curated_sets():
    from kinlogic.names import NAME_GENDER, UNISEX
    held = [(n, g) for n, g in NAME_GENDER.items() if zlib.crc32(n.encode()) % 2 == 1]   # never used for noise rates
    return held, sorted(UNISEX)


def train(a):
    torch.manual_seed(0)
    dev = torch.device(a.device)
    rows = load_table(a.data)
    tr = rows['train']
    w = torch.tensor([math.log1p(s) for _, _, s in tr])   # frequent names count more, rare ones still seen
    X = torch.zeros(len(tr), MAX_LEN, dtype=torch.uint8)   # encoded once
    for k, (n, _, _) in enumerate(tr):
        e = encode(n)
        X[k, :len(e)] = torch.tensor(e, dtype=torch.uint8)
    X, Y = X.to(dev), torch.tensor([p for _, p, _ in tr]).to(dev)
    model = NameModel().to(dev)
    print(f'{sum(p.numel() for p in model.parameters())} parameters; train {len(tr)}, val {len(rows["val"])}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(tr) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    val_clear = clear(rows['val'])
    best, t0, step = 1e9, time.time(), 0
    os.makedirs(a.out, exist_ok=True)
    for ep in range(a.epochs):
        idx = torch.multinomial(w, len(tr), replacement=True).to(dev)
        model.train()
        for i in range(0, len(idx) - a.batch + 1, a.batch):
            j = idx[i:i + a.batch]
            loss = F.binary_cross_entropy_with_logits(model(X[j].long()), Y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 2000 == 0:
                print(f'epoch {ep} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
        ps = predict(model, [n for n, _ in val_clear], dev)
        vloss = sum(-math.log(max(1e-6, p if g == 'F' else 1 - p)) for p, (_, g) in zip(ps, val_clear)) / len(ps)
        print(f'epoch {ep}: val log-loss on clear names {vloss:.4f}', flush=True)
        if vloss < best:
            best = vloss
            torch.save({'model': model.state_dict(), 'config': model.cfg, 'epoch': ep, 'val_logloss': vloss}, f'{a.out}/name_model.pt')
    finish(a.out, rows, dev, a.max_wrong)


def finish(out, rows, dev, max_wrong):
    ck = torch.load(f'{out}/name_model.pt', map_location='cpu', weights_only=False)
    model = NameModel(**ck['config']).to(dev)
    model.load_state_dict(ck['model'])
    val_clear = clear(rows['val'])
    ps = predict(model, [n for n, _ in val_clear], dev)
    t_male, t_female = pick_thresholds(ps, [g for _, g in val_clear], max_wrong)
    held, unisex = curated_sets()
    res = report(model, {'test (data)': clear(rows['test']), 'curated held-out': held}, dev, t_male, t_female)
    up = predict(model, unisex, dev)
    res['unisex left open'] = f'{sum(decide(p, t_male, t_female) is None for p in up)}/{len(unisex)}'
    ck.update({'t_male': t_male, 't_female': t_female, 'results': res})
    torch.save(ck, f'{out}/name_model.pt')
    print(json.dumps({'t_male': t_male, 't_female': t_female, **res}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'eval'])
    ap.add_argument('--data', default='data/names/name_gender.tsv.gz')
    ap.add_argument('--out', default='runs/names-v1')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--max_wrong', type=float, default=0.01)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a)
    else:
        finish(os.path.dirname(a.ckpt or f'{a.out}/name_model.pt'), load_table(a.data), torch.device(a.device), a.max_wrong)
