# Repository layout (Part 2)
#
# part_2/
#   orchestrator.py        # runs quick smoke-train + eval + sample
#   tokenizer.py           # 2.1 byte-level tokenizer (0..255)
#   dataset.py             # 2.2 dataset + batching + shift
#   utils.py               # 2.5 sampling helpers (top-k/top-p)
#   model_gpt.py           # tiny GPT: tok/pos emb + blocks + head
#   train.py               # 2.3/2.4 training loop w/ val eval & ckpt
#   sample.py              # 2.5 text generation from a checkpoint
#   eval_loss.py           # 2.6 evaluate loss on a file/ckpt
#   tests/
#     test_tokenizer.py    # round-trip encode/decode
#     test_dataset_shift.py # label shift sanity
#   runs/                  # (created at runtime) checkpoints & logs
#
# NOTE ON IMPORTS
# ---------------
# All imports are LOCAL. Run from inside `part_2/`.
# Example quickstart (CPU ok):
#   cd part_2
#   python train.py --data tiny.txt --steps 300 --sample_every 100
#   python sample.py --ckpt runs/min-gpt/model_best.pt --tokens 200 --prompt 'Once upon a time '

import pathlib, subprocess, sys, shlex, torch

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / 'runs' / 'min-gpt'

def run(cmd: str):
    print(f"\n >>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

def get_hardware_config() -> dict:
    """Detects available hardware (CUDA, MPS, XPU, CPU) and returns optimized hyperparameters."""
    gpu_config = {
        "batch_size": 64,
        "block_size": 256,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "steps": 4000,
        "dropout": 0.1,
    }
    # MPS shares system RAM: batch 64 peaks ~4.2 GB, batch 32 ~2.2 GB (8 GB M-series, measured)
    mps_config = {**gpu_config, "batch_size": 32}
    cpu_config = {
        "batch_size": 16,
        "block_size": 128,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "steps": 1000,
        "dropout": 0.1,
    }

    # Unified PyTorch Accelerator API (PyTorch 2.6+), else explicit backend checks
    acc_type = None
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        acc_type = torch.accelerator.current_accelerator().type
    elif torch.cuda.is_available():
        acc_type = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        acc_type = "mps"
    elif getattr(torch, "xpu", None) and torch.xpu.is_available():
        acc_type = "xpu"

    if acc_type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        dev_name = torch.cuda.get_device_name(0)
        print(f"[Orchestrator] Detected CUDA GPU: {dev_name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb >= 16:
            return {**gpu_config, "batch_size": 128, "steps": 5000}
        return gpu_config
    elif acc_type == "mps":
        print("[Orchestrator] Detected Apple Silicon GPU (MPS)")
        return mps_config
    elif acc_type is not None:
        print(f"[Orchestrator] Detected Accelerator Backend: {acc_type.upper()}")
        return gpu_config

    print("[Orchestrator] CPU Mode: Using lightweight configuration for fast execution")
    return cpu_config

if __name__ == '__main__':
    cfg = get_hardware_config()
    cmd = (
        f"{sys.executable} train.py --data data/clutrr.txt "
        f"--steps {cfg['steps']} --sample_every 400 --eval_interval 400 "
        f"--batch_size {cfg['batch_size']} --block_size {cfg['block_size']} "
        f"--n_layer {cfg['n_layer']} --n_head {cfg['n_head']} --n_embd {cfg['n_embd']} "
        f"--dropout {cfg['dropout']} --lr 5e-4"
    )
    run(cmd)