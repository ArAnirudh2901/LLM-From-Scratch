"""Reinforcement learning with verifiable rewards on new, unseen keyed questions (r6 stage 3), built so the reader
cannot memorize its way to reward (SFT memorizes, RL generalizes: Chu et al. 2025).

Questions: keyed real-style questions the reader never trained on (--pool: question, options, answer letter), mixed
with generated questions (gold facts). Questions the exact readers already read are skipped: RL works where the model
reads. Every time a question is drawn it is altered afresh (eval.perturb: renamed people, shuffled options, new code
symbols, page noise), so no surface ever repeats (transform-augmented GRPO, 2026).

Per question V altered versions x G sampled readings form ONE group. Reward: +1 when the engine's answer is the key,
0 when wrong, -0.2 when no family fits or nothing is asked. The advantage is the reward minus the group mean over all
versions (Dr. GRPO: no division by the spread), so readings that work whatever the wording rise. Groups whose readings
all score the same are dropped (DAPO dynamic sampling). On-policy: one update per batch, loss averaged per token
(DAPO), a KL anchor (k3) to a reference that is reset to the policy whenever real dev improves (ProRL), and a little
supervised loss on generated gold facts. The best checkpoint and early stop are decided by REAL dev questions only.

    python train_rlvr.py --init runs/v10-r6a/best.pt --pool data/real_train2/pool.jsonl --out runs/v10-r6-rl --amp
"""
from __future__ import annotations

import argparse
import copy
import glob
import gzip
import json
import os
import random
import time

import torch

from kinlogic import generate as gen
from model_reader import make_model
from reader import Vocab, collate, encode_example, fact_losses, fact_token_logprobs, generate
from train import get_device
from train_reader import answers, load_jsonl


def load_pool(patterns, exact_skip=True):
    """Keyed questions (answer letter among the options) that the exact readers do not read."""
    from eval.run_real import as_text
    from kinlogic.codes import _fits, read as read_code
    from kinlogic.prepare import prepare
    from kinlogic.rules import read as read_rules
    from kinlogic.speech import read as read_speech
    rows = []
    for pat in patterns:
        for f in glob.glob(pat):
            for line in (gzip.open(f, 'rt') if f.endswith('.gz') else open(f)):
                r = json.loads(line)
                if r.get('options') and isinstance(r['options'][0], str):   # symgen format: option texts, key index
                    r = {**r, 'options': [[l, t] for l, t in zip('abcde', r['options'])],
                         'answer': 'abcde'[int(r['answer'])] if str(r.get('answer', '')).isdigit() else r.get('answer')}
                if not r.get('answer') or not r.get('options') or r['answer'] not in [l for l, _ in r['options']]:
                    continue
                if r.get('key_status', 'confirmed') not in ('confirmed-exact', 'confirmed', 'corrected', 'agreed'):
                    continue   # only keys checked by build_real_train.py
                if exact_skip and 'exact' in r:
                    if not r['exact']:
                        rows.append(r)
                    continue
                if exact_skip:
                    p = prepare(as_text(r))
                    try:
                        if read_code(p['tokens'], p['options'], p['ents']) or _fits(read_rules(p['tokens']) or read_speech(p['tokens']), p['ents']):
                            continue
                    except Exception:
                        pass
                rows.append(r)
    return rows


def real_reward(p, lf, key):
    from kinlogic.exam import _answer_one
    try:
        res = _answer_one(p, lf, lambda b: [''] * len(b), None)
    except Exception:
        return -0.2
    if res.get('status') in ('none', 'contradiction', 'error'):
        return -0.2
    return 1.0 if res.get('option') == key else 0.0


def main():
    from eval.perturb import variant
    from eval.run_real import as_text, load_real, run, split_of
    from eval.real_clean import broken
    from kinlogic.prepare import prepare
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--pool', nargs='+', required=True, help='keyed question files (jsonl: question, options, answer)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--questions', type=int, default=24, help='questions per step')
    ap.add_argument('--versions', type=int, default=2, help='altered versions of each question in its group')
    ap.add_argument('--group', type=int, default=4, help='readings sampled per version')
    ap.add_argument('--p_generated', type=float, default=0.1, help='share of generated questions (gold facts) in each step')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=2e-6)
    ap.add_argument('--beta', type=float, default=0.02, help='KL anchor to the reference')
    ap.add_argument('--sft_weight', type=float, default=0.05)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--real_dev', type=int, default=300)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = get_device()
    rng = random.Random(a.seed)
    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    vocab, cfg = Vocab.from_ck(ck), ck['config']
    policy = make_model(cfg).to(device)
    policy.load_state_dict(ck['model'])
    policy.c_id = vocab.stoi['<c>']
    ref = copy.deepcopy(policy).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    amp = a.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    names = json.load(open('kinlogic/data/perturb_names.json'))
    pool = load_pool(a.pool)
    dev_rows = [r for r in load_real() if split_of(r) == 'dev' and not broken(r)]
    random.Random(7).shuffle(dev_rows)
    dev_rows = dev_rows[:a.real_dev]
    pool_keys = {r['question'] for r in pool}
    assert not pool_keys & {r['question'] for r in load_real()}, 'an evaluation question is in the RL pool'
    print(f'RL pool: {len(pool)} keyed questions the model must read | real dev for selection: {len(dev_rows)}', flush=True)

    def real_dev():
        policy.eval()
        with torch.no_grad():
            sr, _ = run(lambda b: generate(policy, vocab, b, device, k=4), dev_rows)
        return sum(sr['ok']) / max(1, len(sr['ok']))

    best, flat, order, pos = real_dev(), 0, [], 0
    hist, win = [], []   # (mean reward, real dev) per evaluation: reward rising while real dev falls = overfitting
    flat_q = {}   # pool index -> draws in a row whose readings all scored the same
    print(f'[start] REAL dev {best:.3f}', flush=True)
    torch.save({**ck, 'step_rl': 0, 'real_dev': best}, os.path.join(a.out, 'best.pt'))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        items, drawn = [], {}   # (group, tokens, judge) ; judge: ('real', prepared, key) or ('gen', gold lf, people)
        for qi in range(a.questions):
            if pool and rng.random() >= a.p_generated:
                while True:
                    if pos >= len(order):
                        order, pos = rng.sample(range(len(pool)), len(pool)), 0   # each question once per pass
                    j = order[pos]
                    pos += 1
                    if flat_q.get(j, 0) < 2 or rng.random() < 0.15:
                        break   # questions whose readings all scored the same twice are mostly skipped (no signal)
                r = pool[j]
                drawn[qi] = j
                for _ in range(a.versions):   # a fresh surface every time
                    rv = variant(r, rng, names)
                    p = prepare(as_text(rv))
                    items.append((qi, p['tokens'], ('real', p, rv['answer'])))
            else:
                ex = gen.sample(rng)
                people = sorted({t for t in ex['tokens'] if t[0] == '@'})
                items.append((qi, ex['tokens'], ('gen', ex['lf'], people)))
        policy.eval()
        with torch.no_grad():
            samples = generate(policy, vocab, [it[1] for it in items], device, k=a.group, sample=True, temperature=a.temperature)
        groups = {}
        for (qi, toks, judge), got in zip(items, samples):
            for lf, _ in got:
                if judge[0] == 'real':
                    rw = real_reward(judge[1], lf, judge[2])
                else:
                    gold = answers(judge[1], judge[2])
                    mine = answers(lf, judge[2])
                    rw = 1.0 if mine == gold else (-0.2 if mine is None else 0.0)
                groups.setdefault(qi, []).append((toks, lf, rw))
        batch, adv, rewards, kept = [], [], [], 0
        for qi, g in groups.items():
            rs = [x for _, _, x in g]
            rewards += rs
            if qi in drawn:
                flat_q[drawn[qi]] = flat_q.get(drawn[qi], 0) + 1 if max(rs) - min(rs) < 1e-6 else 0
            if max(rs) - min(rs) < 1e-6:
                continue   # all readings equally good or bad: no signal (DAPO dynamic sampling)
            kept += 1
            m = sum(rs) / len(rs)
            for toks, lf, x in g:
                enc = encode_example(vocab, toks, lf, cfg['block_size'])
                if enc:
                    batch.append(enc)
                    adv.append(x - m)   # Dr. GRPO: no division by the group's spread
        policy.train()
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            loss = torch.zeros((), device=device)
            if batch:
                idx, tgt, _ = collate(batch, vocab.stoi['<pad>'], device)
                lp, mask = fact_token_logprobs(policy, idx, tgt, vocab)
                with torch.no_grad():
                    ref_lp, _ = fact_token_logprobs(ref, idx, tgt, vocab)
                A = torch.tensor(adv, device=device)[:, None]
                mask = mask.float()
                pg = -(A * lp.float() * mask).sum() / mask.sum()   # token-level average (DAPO)
                d = ((ref_lp - lp) * mask).float().clamp(-10, 10)   # k3 estimator explodes when one token's probability collapses
                kl = ((torch.exp(d) - d - 1) * mask).sum() / mask.sum()
                loss = pg + a.beta * kl
            if a.sft_weight:
                gold = [e for e in (encode_example(vocab, ex['tokens'], ex['lf'], cfg['block_size']) for ex in (gen.sample(rng) for _ in range(8))) if e]
                gi, gt, gl = collate(gold, vocab.stoi['<pad>'], device)
                loss_f, _ = fact_losses(policy, gi, gt, gl, vocab)
                loss = loss + a.sft_weight * loss_f
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        win += rewards
        if step % 10 == 0:
            print(f'step {step:5d} | reward {sum(rewards) / max(1, len(rewards)):.3f} | groups with signal {kept}/{len(groups)} | '
                  f'loss {loss.item():.4f} | {(time.time() - t0) / 60:.1f} min', flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            acc = real_dev()
            hist.append((sum(win) / max(1, len(win)), acc))
            win = []
            warn = ' | health: reward up, real dev down (reward overfitting)' if len(hist) >= 3 and hist[-1][0] > hist[-3][0] and hist[-1][1] < best - 0.02 and hist[-2][1] < best - 0.02 else ''
            print(f'[eval {step}] REAL dev {acc:.3f} (best {best:.3f}) | mean reward {hist[-1][0]:.3f}{warn}', flush=True)
            if acc > best + 1e-9:
                best, flat = acc, 0
                torch.save({**ck, 'model': policy.state_dict(), 'step_rl': step, 'real_dev': acc}, os.path.join(a.out, 'best.pt'))
                ref.load_state_dict(policy.state_dict())   # ProRL: re-anchor at the new best
                print('  new best; reference reset', flush=True)
            else:
                flat += 1
                if flat >= a.patience:
                    print('real dev flat: stopping', flush=True)
                    break
    print(f'done; best REAL dev {best:.3f} (the starting model is kept if RL never beat it)', flush=True)


if __name__ == '__main__':
    main()
