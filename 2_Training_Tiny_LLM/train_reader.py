"""Trains the reader model (question -> facts) on generated exam questions and CLUTRR stories.

Starts from the v8 relation model (same GPT size; shared words keep their weights). Questions are generated on
the fly by CPU workers, so no dataset needs uploading. Checkpoints (latest and best) go to --out, which can be a
Google Drive folder; --resume continues a cut-off run.

    python train_reader.py --out runs/reader-v9 --steps 30000 --batch 64          # Colab T4: add --amp
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from kinlogic import generate
from kinlogic.engine import solve
from kinlogic.facts import format as fmt, parse
from model_gpt import GPT
from qa_data import hidden_states
from reader import Vocab, build_vocab, collate, decode, encode_example
from train import get_device


class Stream(IterableDataset):
    """Endless training examples: generated questions, and CLUTRR stories with probability p_clutrr."""

    def __init__(self, vocab, block, seed, clutrr, p_clutrr, unk_p):
        self.vocab, self.block, self.seed, self.clutrr, self.p, self.unk_p = vocab, block, seed, clutrr, p_clutrr, unk_p

    def __iter__(self):
        w = get_worker_info()
        rng = random.Random(self.seed * 1000 + (w.id if w else 0))
        while True:
            ex = rng.choice(self.clutrr) if self.clutrr and rng.random() < self.p else generate.sample(rng)
            enc = encode_example(self.vocab, ex['tokens'], ex['lf'], self.block, rng, self.unk_p)
            if enc:
                yield enc


def as_list(batch):
    return batch


def load_jsonl(path, n=None):
    rows = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


def dev_set(n, seed=999):
    rng = random.Random(seed)
    return [generate.sample(rng) for _ in range(n)]


def answers(lf, people=()):
    """The main answer(s) a fact text leads to: one key per question, plus claim verdicts."""
    stmts, _ = parse(lf)
    try:
        r = solve(stmts, people=people, max_steps=5000)
    except RecursionError:
        return None
    keys = tuple(q['answers'][0]['key'] if q['answers'] else q['status'] for q in r['questions'])
    return keys + tuple(sorted(r['claims'].items()))


def canon(lf):
    return fmt(parse(lf)[0])


def evaluate(model, vocab, device, exs, bs=64):
    """Exact facts and same answers, overall and per kind."""
    preds = decode(model, vocab, [e['tokens'] for e in exs], device, bs=bs)
    by = {}
    for e, p in zip(exs, preds):
        people = sorted({t for t in e['tokens'] if t[0] == '@'})
        exact = canon(p) == canon(e['lf'])
        same = exact or answers(p, people) == answers(e['lf'], people)
        k = by.setdefault(e['kind'], [0, 0, 0])
        k[0] += exact; k[1] += same; k[2] += 1
    tot = [sum(v[i] for v in by.values()) for i in range(3)]
    model.train()
    return {'exact': tot[0] / tot[2], 'answer': tot[1] / tot[2],
            'kinds': {k: (round(v[0] / v[2], 3), round(v[1] / v[2], 3)) for k, v in sorted(by.items())}}, preds


def warm_start(model, vocab, path):
    """Copies v8 weights; words both vocabularies share keep their embeddings and output rows."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    old = ck['model']
    old_stoi = {t: i for i, t in enumerate(ck['vocab'])}
    sd = model.state_dict()
    shared = 0
    for k, v in old.items():
        if k in ('tok_emb.weight', 'head.weight'):
            for t, i in vocab.stoi.items():
                j = old_stoi.get(t)
                if j is not None:
                    sd[k][i] = v[j]
                    shared += k == 'tok_emb.weight'
        elif k in sd and sd[k].shape == v.shape:
            sd[k] = v
    model.load_state_dict(sd)
    print(f'warm start from {path}: {shared}/{len(vocab)} words shared', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/reader-v9')
    ap.add_argument('--init', default='runs/clutrr-qa-v8/best.pt', help="v8 checkpoint to start from ('' for scratch)")
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lm_weight', type=float, default=0.1)
    ap.add_argument('--p_clutrr', type=float, default=0.15)
    ap.add_argument('--unk_p', type=float, default=0.02)
    ap.add_argument('--block', type=int, default=320)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--dev', type=int, default=1500)
    ap.add_argument('--amp', action='store_true', help='fp16 mixed precision on CUDA (weights stay fp32)')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = get_device()
    torch.manual_seed(a.seed)
    latest, best_path = os.path.join(a.out, 'latest.pt'), os.path.join(a.out, 'best.pt')

    ck = torch.load(latest, map_location='cpu', weights_only=False) if a.resume and os.path.exists(latest) else None
    vocab = Vocab(ck['vocab']) if ck else build_vocab()
    cfg = ck['config'] if ck else {'vocab_size': len(vocab), 'block_size': a.block, 'n_layer': 6, 'n_head': 6,
                                   'n_embd': 384, 'dropout': a.dropout}
    model = GPT(**cfg).to(device)
    if ck:
        model.load_state_dict(ck['model'])
    elif a.init and os.path.exists(a.init):
        warm_start(model, vocab, a.init)
        model.to(device)
    print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, vocab {len(vocab)}, device {device}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)
    amp = a.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    step, best = 0, -1.0
    if ck:
        opt.load_state_dict(ck['opt'])
        if 'scaler' in ck and amp:
            scaler.load_state_dict(ck['scaler'])
        step, best = ck['step'], ck.get('best', -1.0)
        print(f'resumed at step {step} (best {best:.4f})', flush=True)
    body = torch.compile(model) if a.compile and device.type == 'cuda' else model

    clutrr = load_jsonl('data/clutrr_facts/train.jsonl.gz') if os.path.exists('data/clutrr_facts/train.jsonl.gz') else []
    dev = dev_set(a.dev)
    cval = load_jsonl('data/clutrr_facts/validation.jsonl.gz', 600) if os.path.exists('data/clutrr_facts/validation.jsonl.gz') else []
    loader = DataLoader(Stream(vocab, a.block, a.seed + step, clutrr, a.p_clutrr, a.unk_p), batch_size=a.batch,
                        num_workers=a.workers, collate_fn=as_list, persistent_workers=a.workers > 0)
    it = iter(loader)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        t = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    def save(path, extra=None):
        tmp = path + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'config': cfg,
                    'vocab': vocab.itos, 'step': step, 'best': best, 'format': 'reader-v9', **(extra or {})}, tmp)
        os.replace(tmp, path)   # never a half-written checkpoint, even if the session dies

    model.train()
    t0, seen = time.time(), 0
    pad = vocab.stoi['<pad>']
    while step < a.steps:
        idx, tgt, lm = collate(next(it), pad, device)
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            h = hidden_states(body, idx)
            mt, ml = tgt != -100, lm != -100
            loss_f = F.cross_entropy(model.head(h[mt]).float(), tgt[mt])
            loss_lm = F.cross_entropy(model.head(h[ml]).float(), lm[ml]) if ml.any() else torch.zeros((), device=device)
            loss = loss_f + a.lm_weight * loss_lm
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += idx.size(0)
        if step % 100 == 0:
            el = time.time() - t0
            print(f'step {step:6d} | facts {loss_f.item():.4f} | words {loss_lm.item():.3f} | lr {lr_at(step):.2e} | '
                  f'{seen / el:.0f} q/s | {el / 60:.1f} min', flush=True)
        if step % a.save_every == 0:
            save(latest)
        if step % a.eval_every == 0 or step == a.steps:
            r, _ = evaluate(model, vocab, device, dev)
            score = r['answer']
            line = f"[eval {step}] generated: exact {r['exact']:.3f} answer {r['answer']:.3f}"
            if cval:
                rc, _ = evaluate(model, vocab, device, cval)
                score = 0.75 * r['answer'] + 0.25 * rc['answer']
                line += f" | CLUTRR val: exact {rc['exact']:.3f} answer {rc['answer']:.3f}"
            print(line + f" | by kind {r['kinds']}", flush=True)
            if score > best:
                best = score
                save(best_path, {'eval': r})
                print(f'  new best {best:.4f}', flush=True)
            save(latest)
    print(f'done: {step} steps, best {best:.4f}', flush=True)


if __name__ == '__main__':
    main()
