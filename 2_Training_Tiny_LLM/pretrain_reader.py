"""Pretrains our own v10 reader from scratch on English (pretrain_data.py shards): next-token prediction over the same
prepared token stream questions use. The result is the --init of train_reader.py, which then learns the fact language
on top of a model that already reads English (Furrer et al. 2020; TinyStories, Eldan & Li 2023).

    python pretrain_reader.py --data /content/drive/MyDrive/kinthread/pretrain --out .../runs/pre-v1 --amp --resume
Random windows of `block` tokens; warmup, constant rate, then a linear decay over the last 20% (warmup-stable-decay);
the last shard's tail is held out for validation loss. Resumable from latest.pt.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model_reader import make_model
from train import get_device


class Shards:
    def __init__(self, folder, hold=2_000_000):
        files = sorted(glob.glob(os.path.join(folder, 'shard_*.npy')))
        assert files, f'no shards in {folder}'
        self.arrs = [np.load(f, mmap_mode='r') for f in files]
        hold = min(hold, self.arrs[-1].size // 5)
        self.val = self.arrs[-1][-hold:]
        self.arrs[-1] = self.arrs[-1][:-hold]
        self.sizes = np.array([a.size for a in self.arrs], dtype=np.float64)
        self.total = int(self.sizes.sum())

    def batch(self, rng, n, block, val=False):
        out = np.empty((n, block + 1), dtype=np.int64)
        for k in range(n):
            a = self.val if val else self.arrs[rng.choice(len(self.arrs), p=self.sizes / self.sizes.sum())]
            i = rng.integers(0, a.size - block - 1)
            out[k] = a[i:i + block + 1]
        t = torch.from_numpy(out)
        return t[:, :-1], t[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--vocab', default='data/subword_vocab.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokens', type=float, default=4e8, help='training tokens (budget)')
    ap.add_argument('--n_layer', type=int, default=16, help='deep and thin beats wide and shallow below 1B (MobileLLM, 2024)')
    ap.add_argument('--n_embd', type=int, default=384)
    ap.add_argument('--n_head', type=int, default=6)
    ap.add_argument('--block', type=int, default=512)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--anneal', default=None, help='shards of kinship text mixed into the decay phase (SmolLM2, MiniCPM, OLMo 2)')
    ap.add_argument('--anneal_share', type=float, default=0.5, help='share of each batch from --anneal during the decay')
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = get_device(force_cpu=a.cpu)
    latest = os.path.join(a.out, 'latest.pt')
    ck = torch.load(latest, map_location='cpu', weights_only=False) if a.resume and os.path.exists(latest) else None
    v = json.load(open(a.vocab)) if not ck else {'itos': ck['vocab'], 'merges': ck['merges']}
    cfg = ck['config'] if ck else {'arch': 'v10', 'vocab_size': len(v['itos']), 'block_size': a.block, 'n_layer': a.n_layer,
                                   'n_head': a.n_head, 'n_embd': a.n_embd, 'dropout': 0.0}
    model = make_model(cfg).to(dev)
    if ck:
        model.load_state_dict(ck['model'])
    steps = int(a.tokens // (a.batch * a.block))
    decay_from = int(0.8 * steps)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and 'emb' not in n]   # no decay on embeddings and norms (OLMo 2, SmolLM3)
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and 'emb' not in n)]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.weight_decay}, {'params': rest, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=dev.type == 'cuda')
    amp = a.amp and dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    step = 0
    if ck:
        opt.load_state_dict(ck['opt'])
        if amp and 'scaler' in ck:
            scaler.load_state_dict(ck['scaler'])
        step = ck['step']
    data = Shards(a.data)
    anneal = Shards(a.anneal, hold=200_000) if a.anneal else None
    rng = np.random.default_rng(a.seed + step)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params / 1e6:.1f}M parameters, vocab {cfg["vocab_size"]}, {data.total / 1e6:.0f}M tokens on disk, '
          f'{steps} steps ({steps * a.batch * a.block / 1e6:.0f}M tokens), from step {step}', flush=True)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        if s < decay_from:
            return a.lr
        return a.lr * (0.1 + 0.9 * (1 - (s - decay_from) / max(1, steps - decay_from)))

    def save(path, extra=None, slim=False):
        tmp = path + '.tmp'
        state = {'model': model.state_dict(), 'config': cfg, 'vocab': v['itos'], 'merges': v['merges'], 'step': step,
                 'pretrain': vars(a), **(extra or {})}
        if not slim:   # final.pt (the --init of train_reader) needs no optimizer state
            state.update(opt=opt.state_dict(), scaler=scaler.state_dict())
        torch.save(state, tmp)
        os.replace(tmp, path)   # a session cut mid-write never corrupts the checkpoint

    @torch.no_grad()
    def val_loss(n=20, src=None):
        model.eval()
        g = np.random.default_rng(1234)
        tot = 0.0
        for _ in range(n):
            x, y = (src or data).batch(g, a.batch, a.block, val=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                tot += F.cross_entropy(model.head(model.hidden(x.to(dev))).float().flatten(0, 1), y.to(dev).flatten()).item()
        model.train()
        return tot / n

    model.train()
    t0, seen = time.time(), 0
    while step < steps:
        k = int(round(a.batch * a.anneal_share)) if anneal and step >= decay_from else 0
        x, y = data.batch(rng, a.batch - k, a.block)
        if k:   # the decay phase sees in-domain text (annealing / mid-training)
            xa, ya = anneal.batch(rng, k, a.block)
            x, y = torch.cat([x, xa]), torch.cat([y, ya])
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model.head(model.hidden(x)).float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += x.numel()
        if step % 100 == 0:
            el = time.time() - t0
            print(f'step {step:6d} | loss {loss.item():.3f} | lr {lr_at(step):.2e} | {seen / el:.0f} tok/s | {el / 60:.1f} min', flush=True)
        if step % a.save_every == 0:
            save(latest)
        if step % a.eval_every == 0 or step == steps:
            vl = val_loss()
            line = f'[eval {step}] held-out loss {vl:.3f} (perplexity {math.exp(vl):.1f})'
            if anneal:
                line += f' | kinship text {val_loss(src=anneal):.3f}'
            print(line, flush=True)
    save(latest)
    save(os.path.join(a.out, 'final.pt'), {'val_loss': val_loss()}, slim=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
