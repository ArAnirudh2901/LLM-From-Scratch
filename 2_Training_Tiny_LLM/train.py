from __future__ import annotations
import argparse, time
import torch
from tokenizer import ByteTokenizer
from dataset import ByteDataset
from model_gpt import GPT

def get_device(force_cpu: bool = False) -> torch.device:
    """Detects and returns the best available device (CUDA, Apple MPS, Intel XPU, CPU)."""
    if force_cpu:
        return torch.device("cpu")
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def estimate_loss(model: GPT, ds: ByteDataset, args) -> dict:
    pass


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
    model = GPT(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout).to(args.device)