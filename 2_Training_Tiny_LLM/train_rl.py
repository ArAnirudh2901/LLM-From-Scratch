"""Reinforcement learning after supervised training (GRPO, as in DeepSeek-R1), with the exact engine as the judge.

For each question the reader samples G fact readings; each gets a reward from the engine:
    +1.0  the readings' answers equal the gold answers (any order or wording of facts that means the same counts)
    +0.2  the facts match the gold facts exactly
    -0.2  the facts contradict each other or ask nothing
Advantages are rewards normalised within each question's group (no value network). The loss pushes up readings
better than their group's average, with a KL penalty to the supervised model and a little supervised loss so the
reader does not drift. Syntax errors are impossible (grammar-constrained sampling), so there is no format reward.

    python train_rl.py --init runs/reader-v9/best.pt --out runs/reader-v9-rl --steps 800        # GPU: add --amp
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from kinlogic import generate as gen
from model_gpt import GPT
from qa_data import hidden_states
from reader import Vocab, collate, encode_example, generate
from train import get_device
from train_reader import answers, canon, dev_set, evaluate, load_jsonl


def reward(pred, gold, gold_ans, people):
    if not pred:
        return -0.2
    ans = answers(pred, people)
    if ans is None or not ans or any(a in ('contradiction', 'none') for a in ans if isinstance(a, str)):
        return -0.2
    r = 1.0 if ans == gold_ans else 0.0
    return r + (0.2 if canon(pred) == canon(gold) else 0.0)


def seq_logprobs(model, vocab, batch, device):
    """Mean log-probability of each sequence's facts (positions after <c>) under `model`, per token too."""
    idx, tgt, _ = collate(batch, vocab.stoi['<pad>'], device)
    h = hidden_states(model, idx)
    mask = tgt != -100
    lp = torch.zeros(tgt.shape, device=device)
    logits = model.head(h[mask]).float()
    lp[mask] = torch.log_softmax(logits, -1).gather(1, tgt[mask][:, None])[:, 0]
    return lp, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='runs/reader-v9/best.pt')
    ap.add_argument('--out', default='runs/reader-v9-rl')
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--questions', type=int, default=16, help='questions per step')
    ap.add_argument('--group', type=int, default=8, help='readings sampled per question')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--beta', type=float, default=0.04, help='KL penalty to the supervised model')
    ap.add_argument('--sft_weight', type=float, default=0.1)
    ap.add_argument('--p_clutrr', type=float, default=0.15)
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--dev', type=int, default=800)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = get_device()
    rng = random.Random(a.seed)
    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    vocab, cfg = Vocab(ck['vocab']), ck['config']
    policy = GPT(**cfg).to(device)
    policy.load_state_dict(ck['model'])
    ref = copy.deepcopy(policy).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    amp = a.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    clutrr = load_jsonl('data/clutrr_facts/train.jsonl.gz') if os.path.exists('data/clutrr_facts/train.jsonl.gz') else []
    dev = dev_set(a.dev, seed=777)

    def score():
        r, _ = evaluate(policy, vocab, device, dev)
        return r['answer'], r

    best, r0 = score()
    print(f"[start] answer {best:.4f} exact {r0['exact']:.4f}", flush=True)
    torch.save({**ck, 'step_rl': 0}, os.path.join(a.out, 'best.pt'))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        qs = [rng.choice(clutrr) if clutrr and rng.random() < a.p_clutrr else gen.sample(rng) for _ in range(a.questions)]
        samples = generate(policy, vocab, [q['tokens'] for q in qs], device, k=a.group, sample=True, temperature=a.temperature)
        batch, adv, rewards = [], [], []
        for q, group in zip(qs, samples):
            people = sorted({t for t in q['tokens'] if t[0] == '@'})
            gold_ans = answers(q['lf'], people)
            rs = [reward(f, q['lf'], gold_ans, people) for f, _ in group]
            rewards += rs
            m = sum(rs) / len(rs)
            sd = math.sqrt(sum((x - m) ** 2 for x in rs) / len(rs))
            if sd < 1e-6:
                continue   # every reading equally good or bad: nothing to learn from this question
            for (f, _), x in zip(group, rs):
                enc = encode_example(vocab, q['tokens'], f, cfg['block_size'])
                if enc:
                    batch.append(enc)
                    adv.append((x - m) / (sd + 1e-4))
        policy.train()
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            loss = torch.zeros((), device=device)
            if batch:
                lp, mask = seq_logprobs(policy, vocab, batch, device)
                with torch.no_grad():
                    ref_lp, _ = seq_logprobs(ref, vocab, batch, device)
                n = mask.sum(1).clamp(min=1)
                A = torch.tensor(adv, device=device)
                pg = -(A * (lp * mask).sum(1) / n).mean()
                d = (ref_lp - lp) * mask
                kl = ((torch.exp(d) - d - 1) * mask).sum(1).div(n).mean()
                loss = pg + a.beta * kl
            if a.sft_weight:   # keep the supervised skill: a little loss on the gold facts
                gold = [e for e in (encode_example(vocab, q['tokens'], q['lf'], cfg['block_size']) for q in qs) if e]
                idx, tgt, _ = collate(gold, vocab.stoi['<pad>'], device)
                h = hidden_states(policy, idx)
                mt = tgt != -100
                loss = loss + a.sft_weight * F.cross_entropy(policy.head(h[mt]).float(), tgt[mt])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 10 == 0:
            print(f'step {step:5d} | reward {sum(rewards) / len(rewards):.3f} | learning from {len(batch)} readings | '
                  f'loss {loss.item():.4f} | {(time.time() - t0) / 60:.1f} min', flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            sc, r = score()
            print(f"[eval {step}] answer {sc:.4f} exact {r['exact']:.4f} by kind {r['kinds']}", flush=True)
            if sc > best:
                best = sc
                torch.save({**ck, 'model': policy.state_dict(), 'step_rl': step, 'eval_rl': r}, os.path.join(a.out, 'best.pt'))
                print(f'  new best {best:.4f}', flush=True)
    print(f'done; best answer accuracy {best:.4f} (the supervised model is kept if RL never beat it)', flush=True)


if __name__ == '__main__':
    main()
