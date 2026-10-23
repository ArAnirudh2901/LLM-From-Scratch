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
import re
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from kinlogic import generate
from kinlogic.engine import solve
from kinlogic.facts import format as fmt, parse
from model_gpt import GPT  # noqa: F401
from model_reader import make_model
from qa_data import hidden_states
from reader import fact_losses, Vocab, build_vocab, collate, decode, encode_example
from train import get_device


class Stream(IterableDataset):
    """Endless training examples: generated questions, CLUTRR stories with probability p_clutrr, and the hand-written
    training split (eval/authored_set.py) with probability p_authored."""

    def __init__(self, vocab, block, seed, clutrr, p_clutrr, unk_p, authored=(), p_authored=0.0):
        self.vocab, self.block, self.seed, self.clutrr, self.p, self.unk_p = vocab, block, seed, clutrr, p_clutrr, unk_p
        self.authored, self.pa = list(authored), p_authored

    def __iter__(self):
        w = get_worker_info()
        rng = random.Random(self.seed * 1000 + (w.id if w else 0))
        while True:
            u = rng.random()
            ex = (rng.choice(self.clutrr) if self.clutrr and u < self.p else
                  rng.choice(self.authored) if self.authored and u < self.p + self.pa else generate.sample(rng))
            enc = encode_example(self.vocab, ex['tokens'], ex['lf'], self.block, rng, self.unk_p)
            if enc:
                yield enc


class PoolStream(IterableDataset):
    """Endless examples from the verified pool (build_dataset.py): generator kinds by their mix weights, deep chains
    with a curriculum (their share grows from p_deep[0] to p_deep[1] and the longest allowed chain from 4 to 12 links
    over the run), CLUTRR stories and the hand-written training split."""

    def __init__(self, vocab, block, seed, pool, clutrr, p_clutrr, authored, p_authored, unk_p, total, start, p_deep):
        # total: examples over which the curriculum ramps (fixed, not the step budget: an early stop must still have
        # trained every depth)
        self.vocab, self.block, self.seed, self.pool = vocab, block, seed, pool
        self.clutrr, self.pc, self.authored, self.pa, self.unk_p = clutrr, p_clutrr, authored, p_authored, unk_p
        self.total, self.start, self.p_deep = total, start, p_deep

    def __iter__(self):
        w = get_worker_info()
        nw = w.num_workers if w else 1
        rng = random.Random(self.seed * 1000 + (w.id if w else 0))
        kinds = [k for k in self.pool['kinds'] if self.pool['kinds'][k]]
        weights = [generate.MIX.get(k, 1) for k in kinds]
        depths = sorted(self.pool['deep'])
        n = 0
        while True:
            t = min(1.0, (self.start + n * nw) / max(1, self.total))
            u = rng.random()
            if self.clutrr and u < self.pc:
                ex = rng.choice(self.clutrr)
            elif self.authored and u < self.pc + self.pa:
                ex = rng.choice(self.authored)
            elif self.pool.get('reworded') and u < self.pc + self.pa + self.pool['p_reworded']:
                ex = rng.choice(self.pool['reworded'])
            elif depths and rng.random() < self.p_deep[0] + (self.p_deep[1] - self.p_deep[0]) * t:
                top = 4 + round(8 * t)
                ds = [d for d in depths if d <= top]
                ex = rng.choice(self.pool['deep'][rng.choices(ds, [1 + d for d in ds])[0]])
            else:
                ex = rng.choice(self.pool['kinds'][rng.choices(kinds, weights)[0]])
            n += 1
            tokens, lf = (ex[0].split(' '), ex[1]) if isinstance(ex, tuple) else (ex['tokens'], ex['lf'])
            enc = encode_example(self.vocab, tokens, lf, self.block, rng, self.unk_p)
            if enc:
                yield enc


def load_pool(root):
    """{'kinds': {kind: rows}, 'deep': {depth: rows}, 'dev': rows, 'test': rows} from build_dataset.py's files."""
    import glob
    import zlib
    keep = lambda r: {'tokens': r['tokens'], 'lf': r['lf'], 'kind': r['kind'], 'depth': r.get('depth')}
    small = lambda r: (' '.join(r['tokens']), r['lf'])   # training rows: two strings each (1.4M rows fit in RAM)
    pool = {'kinds': {}, 'deep': {}, 'dev': [], 'test': []}
    for f in sorted(glob.glob(f'{root}/kinds/*.jsonl.gz')):
        pool['kinds'][os.path.basename(f).split('.')[0]] = [small(r) for r in load_jsonl(f)]
    for f in sorted(glob.glob(f'{root}/deep/*/d*.jsonl.gz')):
        for r in load_jsonl(f):
            pool['deep'].setdefault(r['depth'], []).append(small(r))
    for r in load_jsonl(f'{root}/test/deep.jsonl.gz') if os.path.exists(f'{root}/test/deep.jsonl.gz') else []:
        pool['dev' if zlib.crc32(r['text'].encode()) % 2 == 0 else 'test'].append(keep(r))
    for k in ('dev', 'test'):   # mixed formats and depths in any prefix (the file is ordered by format, then depth)
        random.Random(0).shuffle(pool[k])
    import gc
    gc.collect()
    gc.freeze()   # millions of long-lived objects: keep full collections from rescanning them (and from copying
    return pool   # pages into the forked data workers)


def by_depth(rows, preds):
    """Answer accuracy per chain length, from one decoding pass."""
    acc = {}
    for x, p in zip(rows, preds):
        people = sorted({t for t in x['tokens'] if t[0] == '@'})
        a = acc.setdefault(x['depth'], [0, 0])
        a[0] += answers(p, people) == answers(x['lf'], people)
        a[1] += 1
    return {d: round(a / b, 3) for d, (a, b) in sorted(acc.items())}


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
    """What a fact text leads to, as one comparable value: each question's first answer (for "which pair": every
    pair of members that qualifies), each claim's verdict, and for data sufficiency which statements suffice alone."""
    from kinlogic.engine import pair_holds
    from kinlogic.exam import settled
    stmts, _ = parse(lf)
    try:
        r = solve(stmts, people=people, max_steps=5000)
    except RecursionError:
        return None
    keys = []
    for q in r['questions']:
        if q['q']['t'] == 'pair':
            ids = [e for e in r['ids'] if e.startswith('@')]
            pairs = sorted((a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
                           if r['worlds'] and all(pair_holds(w, q['q']['what'], r['ids'][a], r['ids'][b]) for w in r['worlds']))
            keys.append(('pair', tuple(pairs)))
        else:
            keys.append(q['answers'][0]['key'] if q['answers'] else q['status'])
    parts = sorted({s['part'] for s in stmts if s.get('part')})
    for k in parts:   # data sufficiency: does each statement alone give one answer?
        rk = solve(stmts, parts={k}, people=people, max_steps=5000)
        keys.append(('alone', k, settled(rk)))
    return tuple(keys) + tuple(sorted(r['claims'].items()))


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
        if k in ('tok_emb.weight', 'head.weight') and k in sd:   # v10 ties head to tok_emb
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
    ap.add_argument('--steps', type=int, default=85000, help='budget (about 4 passes over the verified set, the most repetition still worth as much as new data: Muennighoff et al. 2023); the wsd schedule usually stops earlier')
    ap.add_argument('--schedule', default='wsd', choices=['wsd', 'cosine'],
                    help='wsd: warmup, constant rate while held-out scores improve, then a linear decay and stop')
    ap.add_argument('--patience', type=int, default=3, help='evaluations without improvement before the decay starts')
    ap.add_argument('--min_delta', type=float, default=0.002)
    ap.add_argument('--cooldown', type=float, default=0.2, help='decay length as a share of the steps so far (min 2000; 20%% matches cosine, Hägele et al. 2024)')
    ap.add_argument('--run', default=None, help='name of this run (stored in its checkpoints and the runs index)')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lm_weight', type=float, default=0.1)
    ap.add_argument('--p_clutrr', type=float, default=0.15)
    ap.add_argument('--p_authored', type=float, default=0.005, help='share of hand-written training questions')
    ap.add_argument('--unk_p', type=float, default=0.02)
    ap.add_argument('--block', type=int, default=320)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--eval_every', type=int, default=3000)
    ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--dev', type=int, default=1000)
    ap.add_argument('--amp', action='store_true', help='fp16 mixed precision on CUDA (weights stay fp32)')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--arch', default='v10', choices=['v10', 'gpt'], help='v10: RoPE, RMSNorm, SwiGLU, copy head')
    ap.add_argument('--n_layer', type=int, default=6)
    ap.add_argument('--n_loop', type=int, default=None, help='looped reader: run the layer stack this many times (shared weights)')
    ap.add_argument('--loop_curriculum', default='', help="loops by share of the budget, e.g. '1:0,2:0.25,3:0.5' (retrofitted recurrence, McLeish et al. 2025)")
    ap.add_argument('--neftune', type=float, default=0.0, help='noise on input embeddings while training (NEFTune; 5 is the paper default)')
    ap.add_argument('--llrd', type=float, default=1.0, help='layer-wise learning-rate decay: block i gets lr * llrd^(top - i); keeps pretrained features (LP-FT)')
    ap.add_argument('--ema', type=float, default=0.999, help='decay of the averaged weights used to evaluate and save (0: off)')
    ap.add_argument('--pool', default='data/dataset', help="verified questions from build_dataset.py ('' : generate on the fly)")
    ap.add_argument('--p_deep', default='0.08,0.25', help='share of deep chains at the start and after the ramp')
    ap.add_argument('--ramp_steps', type=int, default=20000, help='steps by which every depth (up to 12) is in the mix')
    ap.add_argument('--reworded', default='data/reworded.jsonl.gz', help='extra verified rows, comma-separated files: rewrites (kinlogic/reword.py), STaR rows (star.py), minimal pairs')
    ap.add_argument('--p_reworded', type=float, default=0.15)
    ap.add_argument('--reworded_dev', default='data/reworded_dev.jsonl.gz', help='held-out rewrites (kinlogic/reword_merge.py): scored each evaluation')
    ap.add_argument('--w_reworded', type=float, default=0.4, help="weight of the held-out rewrites in the pick score (closest to real phrasing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = get_device(force_cpu=a.cpu)
    torch.manual_seed(a.seed)
    latest, best_path = os.path.join(a.out, 'latest.pt'), os.path.join(a.out, 'best.pt')

    ck = torch.load(latest, map_location='cpu', weights_only=False) if a.resume and os.path.exists(latest) else None
    sched = (ck or {}).get('sched') or {'decay_start': None, 'decay_len': None, 'flat': 0}
    init = torch.load(a.init, map_location='cpu', weights_only=False) if not ck and a.init and os.path.exists(a.init) else None
    pre = init if init and init.get('merges') else None   # a pretrained model (pretrain_reader.py) brings its vocabulary and shape
    vocab = Vocab.from_ck(ck) if ck else Vocab.from_ck(pre) if pre else build_vocab(extra=(a.reworded or '').split(',')[0] or None)
    cfg = ck['config'] if ck else {**pre['config'], 'dropout': a.dropout, 'block_size': max(pre['config']['block_size'], a.block)} if pre else {'vocab_size': len(vocab), 'block_size': a.block, 'n_layer': a.n_layer, 'n_head': 6,
                                   'n_embd': 384, 'dropout': a.dropout, **({'arch': 'v10'} if a.arch == 'v10' else {})}
    if a.n_loop and not ck:
        cfg = {**cfg, 'n_loop': a.n_loop}
    model = make_model(cfg).to(device)
    if ck:
        model.load_state_dict(ck.get('raw') or ck['model'])
    elif a.init and os.path.exists(a.init):
        warm_start(model, vocab, a.init)
        model.to(device)
    print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, vocab {len(vocab)}, device {device}', flush=True)

    model.neft = a.neftune
    groups = {}
    nb = len(getattr(model, 'blocks', []))
    for n, p_ in model.named_parameters():   # layer-wise learning-rate decay: lower layers move less
        m = re.match(r'blocks\.(\d+)\.', n)
        sc = a.llrd ** (nb - 1 - int(m.group(1))) if m else a.llrd ** nb if n.startswith('tok_emb') else 1.0
        groups.setdefault(sc, []).append(p_)
    opt = torch.optim.AdamW([{'params': ps, 'scale': sc} for sc, ps in groups.items()], lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)
    curriculum = sorted((float(f), int(k)) for k, f in (x.split(':') for x in a.loop_curriculum.split(',') if x)) if a.loop_curriculum else []
    amp = a.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    step, best = 0, -1.0
    if ck:
        opt.load_state_dict(ck['opt'])
        if 'scaler' in ck and amp:
            scaler.load_state_dict(ck['scaler'])
        step, best = ck['step'], ck.get('best', -1.0)
        print(f'resumed at step {step} (best {best:.4f})', flush=True)
    ema = None
    if a.ema:   # averaged weights: evaluated and saved; usually better than the last step's
        import copy
        ema = copy.deepcopy(model).eval()
        if ck and ck.get('raw'):   # 'model' holds the averaged weights
            ema.load_state_dict(ck['model'])
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    body = torch.compile(model) if a.compile and device.type == 'cuda' else model

    clutrr = load_jsonl('data/clutrr_facts/train.jsonl.gz') if os.path.exists('data/clutrr_facts/train.jsonl.gz') else []
    dev = dev_set(a.dev)
    cval = load_jsonl('data/clutrr_facts/validation.jsonl.gz', 400) if os.path.exists('data/clutrr_facts/validation.jsonl.gz') else []
    from eval.authored_set import examples as authored_examples
    authored = authored_examples(('train',)) if a.p_authored > 0 else []
    pool = load_pool(a.pool) if a.pool and os.path.isdir(a.pool) else None
    rw_dev = load_jsonl(a.reworded_dev) if a.reworded_dev and os.path.exists(a.reworded_dev) else []
    extra = [(f.split('*')[0], int(f.split('*')[1]) if '*' in f else 1) for f in (a.reworded or '').split(',') if f]
    extra = [(f, k) for f, k in extra if os.path.exists(f)]   # rewrites, STaR rows, minimal pairs; 'file*3' repeats a file
    if pool and extra:
        pool['reworded'], pool['p_reworded'] = [r for f, k in extra for r in load_jsonl(f) * k], a.p_reworded
        print(f"reworded: {len(pool['reworded'])} verified rewrites at {a.p_reworded:.0%}", flush=True)
    if pool:
        import hashlib
        sig = hashlib.sha1(''.join(r['lf'] for r in pool['dev'][:600] + rw_dev).encode()).hexdigest()[:12]
        if ck is not None and sched.get('dev_sig') != sig:   # scores on another dev set (or an unsigned one) are not comparable
            print(f"deep dev set changed ({sched.get('dev_sig')} -> {sig}): best score and patience reset", flush=True)
            best, sched['flat'] = -1.0, 0
        sched['dev_sig'] = sig
        print(f"pool: {sum(map(len, pool['kinds'].values()))} kind questions, {sum(map(len, pool['deep'].values()))} deep, "
              f"deep dev {len(pool['dev'])} / test {len(pool['test'])}", flush=True)
        data = PoolStream(vocab, a.block, a.seed + step, pool, clutrr, a.p_clutrr, authored, a.p_authored, a.unk_p,
                          min(a.steps, a.ramp_steps) * a.batch, step * a.batch, tuple(map(float, a.p_deep.split(','))))
    else:
        data = Stream(vocab, a.block, a.seed + step, clutrr, a.p_clutrr, a.unk_p, authored, a.p_authored)
    loader = DataLoader(data, batch_size=a.batch, num_workers=a.workers, collate_fn=as_list, persistent_workers=a.workers > 0)
    it = iter(loader)


    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        if a.schedule == 'cosine':
            t = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
            return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
        if sched['decay_start'] is None:   # stable phase
            return a.lr
        t = min(1.0, (s - sched['decay_start']) / sched['decay_len'])
        return a.lr * (0.1 + 0.9 * (1 - math.sqrt(t)))   # 1-sqrt cooldown (Hägele et al. 2024)

    import subprocess
    meta = {'run': a.run or os.path.basename(a.out.rstrip('/')), 'args': vars(a), 'started': time.strftime('%Y-%m-%d %H:%M'),
            'git': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip(),
            'data': None}
    if a.pool and os.path.exists(f'{a.pool}/manifest.json'):
        import hashlib
        raw = open(f'{a.pool}/manifest.json', 'rb').read()
        meta['data'] = {'total': json.loads(raw)['total'], 'manifest_sha256': hashlib.sha256(raw).hexdigest()}

    def save(path, extra=None, resumable=True):
        """'model' holds the weights to use (averaged when EMA is on); a resumable checkpoint also keeps the raw
        weights and the optimizer."""
        tmp = path + '.tmp'
        ck_ = {'model': (ema or model).state_dict(), 'config': cfg, 'vocab': vocab.itos, 'merges': vocab.merges, 'step': step, 'best': best,
               'format': 'reader-v10' if cfg.get('arch') == 'v10' else 'reader-v9', 'meta': meta, 'sched': sched,
               **(extra or {})}
        if resumable:
            ck_.update({'raw': model.state_dict() if ema else None, 'opt': opt.state_dict(), 'scaler': scaler.state_dict()})
        torch.save(ck_, tmp)
        os.replace(tmp, path)   # never a half-written checkpoint, even if the session dies

    model.train()
    t0, seen = time.time(), 0
    pad = vocab.stoi['<pad>']
    while step < a.steps:
        idx, tgt, lm = collate(next(it), pad, device)
        for g in opt.param_groups:
            g['lr'] = lr_at(step) * g.get('scale', 1.0)
        if curriculum:   # loops in use grow with training (retrofitted recurrence)
            model.set_loops(max(k for f, k in curriculum if f <= step / a.steps))
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            loss_f, loss_lm = fact_losses(body if cfg.get('arch') != 'v10' else model, idx, tgt, lm, vocab)
            loss = loss_f + a.lm_weight * loss_lm
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            with torch.no_grad():
                for pe, p_ in zip(ema.parameters(), model.parameters()):
                    pe.mul_(a.ema).add_(p_.detach(), alpha=1 - a.ema)
        step += 1
        seen += idx.size(0)
        if step % 100 == 0:
            el = time.time() - t0
            print(f'step {step:6d} | facts {loss_f.item():.4f} | words {loss_lm.item():.3f} | lr {lr_at(step):.2e} | '
                  f'{seen / el:.0f} q/s | {el / 60:.1f} min', flush=True)
        if step % a.save_every == 0:
            save(latest)
        if step % a.eval_every == 0 or step == a.steps:
            judge = ema or model
            r, _ = evaluate(judge, vocab, device, dev)
            score = r['answer']
            line = f"[eval {step}] generated: exact {r['exact']:.3f} answer {r['answer']:.3f}"
            if cval:
                rc, _ = evaluate(judge, vocab, device, cval)
                score = 0.6 * r['answer'] + 0.2 * rc['answer']
                line += f" | CLUTRR val: exact {rc['exact']:.3f} answer {rc['answer']:.3f}"
            if pool and pool['dev']:
                rows = pool['dev'][:600]
                rd, preds = evaluate(judge, vocab, device, rows)
                score += 0.2 * rd['answer']
                line += f" | deep dev answer {rd['answer']:.3f} by depth {by_depth(rows, preds)}"
            if rw_dev:
                rr, _ = evaluate(judge, vocab, device, rw_dev)
                score += a.w_reworded * rr['answer']
                line += f" | reworded dev answer {rr['answer']:.3f}"
            print(line + f" | by kind {r['kinds']}", flush=True)
            if score > best + a.min_delta:
                sched['flat'] = 0
            else:
                sched['flat'] += 1
            if score > best:
                best = score
                save(best_path, {'eval': r, 'score': score}, resumable=False)
                print(f'  new best {best:.4f}', flush=True)
            ramped = step >= min(a.steps, a.ramp_steps)   # never stop before every depth has been trained
            if a.schedule == 'wsd' and sched['decay_start'] is None and ((ramped and sched['flat'] >= a.patience) or step >= 0.9 * a.steps):
                sched['decay_start'], sched['decay_len'] = step, max(2000, int(a.cooldown * step))
                print(f"  held-out scores flat for {sched['flat']} evaluations: decaying the rate over "
                      f"{sched['decay_len']} steps, then stopping", flush=True)
            save(latest)
            if sched['decay_start'] is not None and step >= sched['decay_start'] + sched['decay_len']:
                break
    print(f'done: {step} steps, best {best:.4f}', flush=True)
    index = os.path.join(os.path.dirname(a.out.rstrip('/')), 'index.jsonl')
    with open(index, 'a') as f:   # one line per finished run, so versions stay told apart
        f.write(json.dumps({**{k: v for k, v in meta.items() if k != 'args'}, 'out': a.out, 'steps': step, 'best': best,
                            'arch': cfg.get('arch', 'gpt'), 'finished': time.strftime('%Y-%m-%d %H:%M')}) + '\n')


if __name__ == '__main__':
    main()
