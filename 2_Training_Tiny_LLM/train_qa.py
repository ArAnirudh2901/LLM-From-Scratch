from __future__ import annotations
import argparse, math, os, random, time
from collections import defaultdict, deque

import torch
import torch.nn.functional as F

from model_gpt import GPT
from qa_data import (NOFACT, RELATIONS, TRAIN_TEMPLATES, UNKNOWN, StorySynth, Vocab, add_distractors, bad_chain,
                     beam_predict, hidden_states, load_splits, make_batch, mine_negatives, predict, unanswerable,
                     unknown_eval_sets, unlinked)
from train import get_device


def evaluate(model, vocab, data, device, cot: bool, block_size: int, beam: int = 0, views: int = 1, bs: int = 64,
             unk_threshold: float = 0.5, min_conf: float = 0.7) -> dict:
    """On clean examples (no name-swapped twin in raw training data): accuracy overall and per hop count
    ('unknown' is right only for unanswerable questions), share answered 'unknown', and share answered with
    a relation at agreement >= min_conf plus accuracy on those (beam only; greedy counts every relation)."""
    data = [e for e in data if not e.get('seen')]
    model.eval()
    rows = []   # (gold, prediction ignoring p_unknown, p_start, p_link, agreement, hops): rescorable at any threshold
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        if beam and cot:
            preds, _, info = beam_predict(model, vocab, chunk, device, beam=beam, block_size=block_size,
                                          views=views, unk_threshold=math.inf)
            share, p_start, p_link = info['share'].tolist(), info['p_start'].tolist(), info['p_link'].tolist()
        else:
            preds, _ = predict(model, vocab, chunk, device, cot, block_size)
            share, p_start, p_link = [1.0] * len(chunk), [0.0] * len(chunk), [1.0] * len(chunk)
        rows += [(ex['rel'], *x, ex['k']) for ex, x in zip(chunk, zip(preds, p_start, p_link, share))]
    model.train()
    if device.type == 'mps':
        torch.mps.empty_cache()
    return score_rows(rows, unk_threshold, min_conf)


def score_rows(rows, unk_threshold: float, min_conf: float) -> dict:
    per_k = defaultdict(lambda: [0, 0])
    unk = sure = sure_hit = 0
    for gold, pr, p_start, p_link, sh, k in rows:
        pr = UNKNOWN if max(p_start, 1 - p_link) >= unk_threshold else pr
        hit = pr == gold
        per_k[k][0] += hit; per_k[k][1] += 1
        unk += pr == UNKNOWN
        if pr != UNKNOWN and sh >= min_conf:
            sure += 1; sure_hit += hit
    n = max(1, len(rows))
    return {'acc': sum(h for h, _ in per_k.values()) / n, 'unknown': unk / n, 'sure': sure / n,
            'sure_acc': sure_hit / max(1, sure), 'per_k': {k: round(h / c, 3) for k, (h, c) in sorted(per_k.items())},
            'rows': rows}


def report(tag: str, name: str, r: dict, min_conf: float) -> str:
    return (f"[{tag}] {name}: correct {r['acc']:.3f} | said unknown {r['unknown']:.3f} | "
            f"answered with agreement>={min_conf}: {r['sure']:.3f} (correct {r['sure_acc']:.3f}) | by hops {r['per_k']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='runs/clutrr-qa')
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=6)
    p.add_argument('--n_embd', type=int, default=384)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--lm_weight', type=float, default=0.1)   # auxiliary next-word loss on the story
    p.add_argument('--sub_prob', type=float, default=0.3)    # share of exact sub-questions from proofs and facts
    p.add_argument('--other_tests', action='store_true')     # also train on other CLUTRR configs' test splits
    p.add_argument('--syn_frac', type=float, default=0.0)    # share of synthesized stories per batch
    p.add_argument('--unk_frac', type=float, default=0.0)    # share of unanswerable questions (answer 'unknown')
    p.add_argument('--distract_frac', type=float, default=0.0)  # share of answerable questions with distractors added
    p.add_argument('--neg_frac', type=float, default=0.0)    # share of chains with one invented step (target <nofact>)
    p.add_argument('--mine_frac', type=float, default=0.0)   # share of mined hard negatives (model's own false steps)
    p.add_argument('--mine_every', type=int, default=200)    # mine a fresh set every this many steps
    p.add_argument('--mine_n', type=int, default=512)        # unanswerable questions searched per mining round
    p.add_argument('--eval_beam', type=int, default=0)       # in-training eval: beam width, 1 view, on dev subsets (0 = greedy)
    p.add_argument('--unk_eval', action='store_true')        # final eval: also unanswerable and distractor sets
    p.add_argument('--unk_threshold', type=float, default=0.5)  # answer 'unknown' when p_unknown >= this
    p.add_argument('--min_conf', type=float, default=0.7)    # report: answers with vote agreement >= this
    p.add_argument('--init_from', default=None)              # continue from a checkpoint (same vocab)
    p.add_argument('--cot', action='store_true')             # write path facts before the answer
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--beam', type=int, default=8)             # final eval: constrained beam width (0 = greedy)
    p.add_argument('--views', type=int, default=4)           # final eval: random person-id views to vote over
    p.add_argument('--eval_only', action='store_true')       # skip training; evaluate --init_from
    p.add_argument('--block_size', type=int, default=320)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = get_device(args.cpu)
    train, dev, val, test = load_splits(other_tests=args.other_tests)
    init = torch.load(args.init_from) if args.init_from else None
    vocab = Vocab(init['vocab']) if init else Vocab.build(train)
    new = [tok for tok, on in ((UNKNOWN, args.unk_frac > 0), (NOFACT, args.neg_frac > 0)) if on and tok not in vocab.stoi]
    if new:   # new tokens; their rows start as the mean relation row (P(<nofact>) starts low: steps look valid)
        vocab = Vocab(vocab.itos + new)
        if init:
            rel = torch.tensor(vocab.encode(RELATIONS))
            for k in ('tok_emb.weight', 'head.weight'):
                w = init['model'][k]
                init['model'][k] = torch.cat([w, w[rel.to(w.device)].mean(0, keepdim=True).expand(len(new), -1)])
    synth = StorySynth(train) if args.syn_frac > 0 else None
    clean = lambda d: sum(not e['seen'] for e in d)
    print(f"device {device} | train {len(train)} | dev {clean(dev)} | val {clean(val)} | test {clean(test)} (clean) | vocab {len(vocab)} | cot {args.cot}")

    config = dict(vocab_size=len(vocab), block_size=args.block_size, n_layer=args.n_layer,
                  n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(**config).to(device)
    if init:
        model.load_state_dict(init['model'])
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    mined = deque(maxlen=4 * args.mine_n)

    def mine():
        cands = [unanswerable(rng.choice(train), rng, TRAIN_TEMPLATES, rng.choice(train), rng.choice(['two', 'friend']))
                 for _ in range(args.mine_n)]
        model.eval()
        new = mine_negatives(model, vocab, [c for c in cands if c], device, synth.table)
        model.train()
        rng.shuffle(new)
        mined.extend(new)
        if device.type == 'mps':
            torch.mps.empty_cache()
        return len(new)

    def draw():
        if mined and rng.random() < args.mine_frac:
            return mined.popleft()
        e = synth.sample(rng) if synth and rng.random() < args.syn_frac else rng.choice(train)
        r = rng.random()
        if r < args.unk_frac:   # CLUTRR's disconnected-noise stories give real unlinked people
            kind = 'disc' if unlinked(e) and rng.random() < 0.5 else None
            return unanswerable(e, rng, TRAIN_TEMPLATES, rng.choice(train), kind) or e
        if r < args.unk_frac + args.neg_frac:
            return bad_chain(e, rng, TRAIN_TEMPLATES, rng.choice(train), synth.table if synth else None) or e
        if r < args.unk_frac + args.neg_frac + args.distract_frac:
            return add_distractors(e, rng, TRAIN_TEMPLATES, rng.choice(train)) or e
        return e

    def batches():
        # length buckets with a token budget: similar lengths per batch (little padding) and
        # rows x padded length <= batch_size x 96 (~190 KB activations per token on 6 layers)
        size = lambda e: -(-(len(e['story'].split()) + 4 * e['k'] + 12) // 16) * 16
        while True:
            pool = [draw() for _ in range(args.batch_size * 16)]
            pool.sort(key=size)
            chunks, cur = [], []
            for e in pool:
                if cur and (len(cur) + 1) * size(e) > args.batch_size * 96:
                    chunks.append(cur); cur = []
                cur.append(e)
            chunks.append(cur)
            rng.shuffle(chunks)
            yield from chunks

    os.makedirs(args.out_dir, exist_ok=True)
    dev_unk = unknown_eval_sets(dev, 100 if args.eval_beam else 150, seed=1) if args.unk_frac > 0 and not args.eval_only else {}
    dev_sel = dev[:300] if args.eval_beam else dev
    best, t0, stream = -1.0, time.time(), batches()
    for step in range(1, 0 if args.eval_only else args.steps + 1):
        if args.mine_frac > 0 and step % args.mine_every == 1:
            t1 = time.time()
            print(f"[mine] step {step} | {mine()} hard negatives from {args.mine_n} questions | {time.time() - t1:.0f}s", flush=True)
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        idx, tgt, lm_tgt, ans, pos = make_batch(vocab, next(stream), device, rng, args.sub_prob, args.cot)
        h = hidden_states(model, idx)
        mc, ml = tgt != -100, lm_tgt != -100   # vocab head only where a target exists: no padding, no B*T*V tensor
        chain_loss = F.cross_entropy(model.head(h[mc]), tgt[mc])
        ans_loss = F.cross_entropy(model.head(h[torch.arange(len(ans), device=device), pos]), ans) \
            if (ans != -100).any() else h.new_zeros(())
        lm_loss = F.cross_entropy(model.head(h[ml]), lm_tgt[ml])
        loss = chain_loss + ans_loss + args.lm_weight * lm_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == 'mps' and step % 50 == 0:
            torch.mps.empty_cache()   # MPS keeps a buffer per batch shape; without this it grew to 6 GB and swapped

        if step % 200 == 0:
            print(f"step {step:6d} | chain {chain_loss.item():.3f} | answer {ans_loss.item():.3f} | lm {lm_loss.item():.3f} | lr {lr_at(step):.2e} | {time.time() - t0:.0f}s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            ev = dict(beam=args.eval_beam, unk_threshold=args.unk_threshold)
            r = evaluate(model, vocab, dev_sel, device, args.cot, args.block_size, **ev)
            line, score = f"dev acc {r['acc']:.3f} (said unknown {r['unknown']:.3f}) | by hops {r['per_k']}", r['acc']
            if dev_unk:   # select on answerable, distractor and unanswerable questions alike
                acc = {k: evaluate(model, vocab, v, device, args.cot, args.block_size, **ev)['acc'] for k, v in dev_unk.items()}
                line += ' | ' + ' | '.join(f'{k} {a:.3f}' for k, a in acc.items())
                unk = [a for k, a in acc.items() if k.startswith('unknown')]
                score = (r['acc'] + acc['answerable + distractors'] + sum(unk) / len(unk)) / 3
            print(f"[eval] step {step} | {line} | score {score:.3f} | {time.time() - t0:.0f}s", flush=True)
            if score > best:
                best = score
                torch.save({'model': model.state_dict(), 'config': config, 'vocab': vocab.itos,
                            'cot': args.cot, 'step': step}, f"{args.out_dir}/best.pt")

    ck = init if args.eval_only else torch.load(f"{args.out_dir}/best.pt")
    model.load_state_dict(ck['model'])
    sets = [('val (clean new stories, 2-4 hops)', val), ('test (new stories, 2-10 hops)', test)]
    if args.unk_eval or args.unk_frac > 0:
        sets += [(f'val {k}', v) for k, v in unknown_eval_sets(val, 400, seed=2).items()]
        sets += [(f'test {k}', v) for k, v in unknown_eval_sets(test, 300, seed=3).items()]
    tag, rows = f"final best@{ck['step']} beam {args.beam} views {args.views}", {}
    for name, data in sets:
        r = evaluate(model, vocab, data, device, args.cot, args.block_size, args.beam, args.views,
                     unk_threshold=args.unk_threshold, min_conf=args.min_conf)
        rows[name] = r['rows']
        print(report(tag, name, r, args.min_conf), flush=True)
    torch.save(rows, f"{args.out_dir}/final_rows.pt")   # per-example scores: rescore any threshold with score_rows


if __name__ == '__main__':
    main()
