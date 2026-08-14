import argparse, subprocess, sys, pathlib, shlex

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out" 

def run(cmd: str):
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)

    if res.returncode != 0:
        sys.exit(res.returncode)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visualize", action="store_true", help="run visualization scripts and save PNGs to ./out")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)

    # Sanity Checks using a small Numpy example
    run("python attn_numpy_demo.py")

    # Performing unit tests
    run("python -m pytest -q test_attn_math.py")
    run("python -m pytest -q test_causal_mask.py")

    # The Matrix math walkthrough for Multi-head Attention Mechanism
    run("python demo_mha_shapes.py")

    if args.visualize:
        run("python demo_visualize_multi_head.py")
        print(f"\nVisualization images saved to: {OUT}")

    print("\nTesting the Transformer Architecture implementation completed ✅ ")

if __name__ == "__main__":
    main()