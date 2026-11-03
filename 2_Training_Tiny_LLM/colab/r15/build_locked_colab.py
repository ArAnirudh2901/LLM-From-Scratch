"""Writes colab/r15/r15_locked_colab.ipynb: r15's locked test (owner approval "approve locked", 2026-11-03), scored once
on a Colab T4 with the dev FINAL setting and r15's own verifier. Reads r15 and the private questions from the Drive
folder the r15 run wrote (MyDrive/kinthread); writes r15_locked.json there. A finished result is never recomputed.
    python colab/r15/build_locked_colab.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    ('markdown', '# r15 locked test (once)\nRuntime > Change runtime type > T4 GPU, then Runtime > Run all. Allow Drive.'),
    ('code', '''import os, subprocess, json
def sh(cmd):
    print('$', cmd, flush=True)
    if subprocess.run(cmd, shell=True).returncode:
        raise SystemExit(f'STOP: command failed: {cmd}')
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/kinthread/runs/v10-r15'
R = '/content/km'
if not os.path.isdir(R):
    sh(f'git clone -q --depth 1 https://github.com/ArAnirudh2901/Kinthread-llm-model {R}')
os.chdir(R)
sh('tar -xzf /content/drive/MyDrive/kinthread/private_r15.tgz -C /content/km')
sh(f'echo "9ec3886a9c15b14f7598d27ee2821197cf423d494645ab76863dc2e5444102f3  {OUT}/best.pt" | sha256sum -c')'''),
    ('code', '''res = f'{OUT}/r15_locked.json'
def finished(path):
    try:
        json.load(open(path))
        return True
    except (OSError, ValueError):
        return False
if not finished(res):   # scored once: a finished result is never recomputed
    sh(f'python -m eval.run_real --ckpt {OUT}/best.pt --split test --adaptive 1.0 --hybrid --cover --split_search --para 2 '
       f'--k 16 --verifier {OUT}/r15_verifier.json --out /content/r15_locked.json')
    sh(f'cp /content/r15_locked.json {res}')
print(json.dumps(json.load(open(res))['score']['splits'], indent=1))
print('LOCKED DONE')'''),
]

if __name__ == '__main__':
    nb = {'nbformat': 4, 'nbformat_minor': 0, 'cells': [],
          'metadata': {'colab': {'provenance': []}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'accelerator': 'GPU'}}
    for kind, src in CELLS:
        cell = {'cell_type': kind, 'metadata': {}, 'source': src}
        if kind == 'code':
            cell.update({'execution_count': None, 'outputs': []})
        nb['cells'].append(cell)
    json.dump(nb, open(os.path.join(HERE, 'r15_locked_colab.ipynb'), 'w'), indent=1)
