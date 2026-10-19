from __future__ import annotations
import argparse, os, time
import torch
from tokenizer import ByteTokenizer
from dataset import ByteDataset
from model_gpt import GPT

def get_device(force_cpu: bool = False) -> torch.device:
    """Detects and returns the best available device (CUDA, Apple MPS, Intel XPU, CPU)."""
    if force_cpu:
        return torch.device("cpu")
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.device(torch.accelerator.current_accelerator().type)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def get_grad_scaler(device: torch.device, enabled: bool):
    """Scaler only on cuda, mps (tested). No-op elsewhere."""
    use_scaler = enabled and device.type in {"cuda", "mps"}
    return torch.amp.GradScaler(device.type, enabled=use_scaler)


@torch.no_grad()
def estimate_loss(model: GPT, ds: ByteDataset, args) -> dict:
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            xb, yb = ds.get_batch(split, args.batch_size, args.device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    
    p = argparse.ArgumentParser()
    
    p.add_argument("--data", type=str, default="data/clutrr.txt")
    p.add_argument("--out_dir", type=str, default="runs/min-gpt")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=400)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=400)
    p.add_argument("--sample_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--amp", action="store_true")
    
    args = p.parse_args()

    args.device = get_device(args.cpu)
    print(f"Using device: {args.device}")

    tokenizer = ByteTokenizer()
    ds = ByteDataset(args.data, block_size=args.block_size)
    model = GPT(tokenizer.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout).to(args.device)

    raw_model = model   # compiled model prefixes state_dict keys with "_orig_mod."
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = get_grad_scaler(args.device, args.amp)

    config = {
        'vocab_size': tokenizer.vocab_size,
        'block_size': args.block_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
        'dropout': args.dropout,
    }

    if args.device.type in {"cuda", "mps"}:
        amp_dtype = torch.float16
    elif args.device.type == "cpu":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32
    amp_enabled = args.amp and args.device.type in {"cuda", "mps", "cpu"}

    best_val = float('inf')
    t0 = time.time()

    for step in range(1, args.steps + 1):
        xb, yb = ds.get_batch('train', args.batch_size, args.device)

        with torch.autocast(
            device_type=args.device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            _, loss = model(xb, yb)
        
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(opt)        # unscale first so clip sees true grad norms
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)      # This clips the gradients to prevent the exploding gradient problem 

        scaler.step(opt)
        scaler.update()

        if step % 50 == 0:
            print(f"Step: {step:5d}  |  Loss: {loss.item():.4f}  |  {(time.time()-t0):.1f}s")
            t0 = time.time()

        if step % args.eval_interval == 0:
            losses = estimate_loss(model, ds, args)
            print(f"Eval  |  Train {losses['train']:.4f}  |  val {losses['val']:.4f}")

            if losses['val'] < best_val:
                best_val = losses['val']
                ckpt_path = f"{args.out_dir}/model_best.pt"
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save({'model': raw_model.state_dict(), 'config': config}, ckpt_path)
                print(f"Saved Checkpoint: {ckpt_path}")

        if args.sample_every > 0 and step % args.sample_every == 0:
            start = torch.randint(low=0, high=len(ds.train) - args.block_size - 1, size=(1,)).item()
            seed = ds.train[start:start + args.block_size].unsqueeze(0).to(args.device)
            raw_model.eval()
            out = raw_model.generate(seed, max_new_tokens=args.sample_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            raw_model.train()
            txt = tokenizer.decode(out[0].cpu())

            print("\n====================SAMPLE====================\n" + txt[-(args.block_size + args.sample_tokens):] + "\n====================")

    # Saving the final model
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({'model': raw_model.state_dict(), 'config': config}, f"{args.out_dir}/model_final.pt")

if __name__ == '__main__':
    main()