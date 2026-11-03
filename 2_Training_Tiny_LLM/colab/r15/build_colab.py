"""Writes colab/r15/r15_colab.ipynb: the final run (r15) on a free Colab T4 when Kaggle's weekly quota is spent. Same
recipe as colab/r15 step C (r13 + hard constructions + teacher rows, twins on), then step D (own verifier, gauntlet,
consistency, dev FINAL setting vs r8p). Outputs go to Drive (MyDrive/kinthread/runs/v10-r15), so a dropped session
resumes (training --resume; finished steps are skipped). r13's weights come from the repo (two parts, SHA checked);
the private dev questions are uploaded once from the owner's machine (private_r15.tgz, never in git).
    python colab/r15/build_colab.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    ('markdown', '# r15 (final model) on Colab\nRuntime > Change runtime type > T4 GPU, then Runtime > Run all. Allow Drive; '
                 'upload `private_r15.tgz` when asked (first run only).'),
    ('code', '''import os, subprocess, json
def sh(cmd):
    print('$', cmd, flush=True)
    if subprocess.run(cmd, shell=True).returncode:
        raise SystemExit(f'STOP: command failed: {cmd}')
sh('nvidia-smi -L')
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/kinthread/runs/v10-r15'
os.makedirs(OUT, exist_ok=True)
R = '/content/km'
if not os.path.isdir(R):
    sh(f'git clone -q --depth 1 https://github.com/ArAnirudh2901/Kinthread-llm-model {R}')
os.chdir(R)
if subprocess.run('cd runs/v10-r13-final && sha256sum -c --quiet best.pt.sha256', shell=True).returncode:   # missing or partial
    sh('cat runs/v10-r13-final/best.pt.part-* > runs/v10-r13-final/best.pt')
    sh('cd runs/v10-r13-final && sha256sum -c best.pt.sha256')
sh('pip -q install -U gdown')
S = '/content/share'
os.makedirs(S, exist_ok=True)
if not os.path.isdir(f'{S}/dataset_v7'):
    sh(f'gdown -q --folder https://drive.google.com/drive/folders/1MNiQuramSX5ImytxEdvNEIN7SgFtT6k8 -O {S}/dataset_v7')
if not os.path.exists(f'{S}/star_r3.jsonl.gz'):
    sh(f'wget -q -O {S}/star_r3.jsonl.gz "https://drive.usercontent.google.com/download?id=1_qk0JL_lkzKBdv2vwHxWijI4yNrNPRtr&export=download&confirm=t"')
sh('cd data && sha256sum -c SHA256SUMS --ignore-missing --quiet')
sh('python -m pytest -q tests/test_model_reader.py tests/test_twins.py tests/test_build_hard.py')'''),
    ('code', '''P = '/content/drive/MyDrive/kinthread/private_r15.tgz'   # dev questions + r8p dev baseline: evaluation only
if not os.path.exists(P):
    from google.colab import files
    up = files.upload()
    open(P, 'wb').write(next(iter(up.values())))
sh(f'tar -xzf {P} -C {R}')
print(len(os.listdir('data/real_questions')), 'real-question files')'''),
    ('code', '''EXTRA = [f'{S}/dataset_v7/reworded.jsonl.gz*2', 'data/mix_r8.jsonl.gz', f'{S}/star_r3.jsonl.gz*2',
         'data/balanced_mix_v1.jsonl.gz*2', 'data/detour_v1.jsonl.gz', 'data/teach_sg3.jsonl.gz*4', 'data/hard_v1.jsonl.gz']
missing = [e for e in EXTRA if not os.path.exists(e.split('*')[0])]
if missing:
    raise SystemExit(f'STOP: training files missing: {missing}')
if not os.path.exists(f'{OUT}/done.txt'):
    sh('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_reader.py --arch v10 --run v10-r15 '
       f'--out {OUT} --pool {S}/dataset_v7 --init runs/v10-r13-final/best.pt --prefix_lm --of_form --style_p 0.8 '
       '--steps 6000 --batch 64 --lr 1e-4 --warmup 300 --optim muon --amp --workers 2 --resume --compile '
       '--eval_every 1000 --save_every 500 --reworded ' + ','.join(EXTRA) + ' --reworded_dev data/reworded_dev.jsonl.gz '
       '--p_reworded 0.75 --block 512 --real_dev 575 --select_real --cons_p 0.5 --cons_w 0.5')
    open(f'{OUT}/done.txt', 'w').write('trained')
sh(f'sha256sum {OUT}/best.pt')'''),
    ('code', '''B = f'{OUT}/best.pt'
steps = [(f'{OUT}/r15_verifier.json', f'python train_verifier.py --ckpt {B} --n 1500 --out {OUT}/r15_verifier.json'),
         (f'{OUT}/r15_dev_final.json', f'python -m eval.run_real --ckpt {B} --split dev --adaptive 1.0 --hybrid --cover --split_search '
                                       f'--para 2 --k 16 --verifier {OUT}/r15_verifier.json --out {OUT}/r15_dev_final.json'),
         (f'{OUT}/compare_final.json', f'python -m eval.compare_runs private/r8p_dev_final.json {OUT}/r15_dev_final.json --out {OUT}/compare_final.json'),
         (f'{OUT}/r15_consistency.json', f'python -m eval.consistency --groups 300 --ckpt {B} --k 16 --out {OUT}/r15_consistency.json'),
         (f'{OUT}/r15_gauntlet.json', f'python -m eval.gauntlet --ckpt {B} --k 16 --out {OUT}/r15_gauntlet.json')]
def finished(path):   # a file cut off by a dropped session is redone, not skipped
    try:
        json.load(open(path))
        return True
    except (OSError, ValueError):
        return False
for out, cmd in steps:
    if not finished(out):
        sh(cmd)
cmp = json.load(open(f'{OUT}/compare_final.json'))
g, c = json.load(open(f'{OUT}/r15_gauntlet.json')), json.load(open(f'{OUT}/r15_consistency.json'))
summary = {'dev': cmp, 'gauntlet': {'r8p': 0.9495, 'r13': 0.9443, 'r15': g['correct']},
           'consistency_all_agree': {'r8p': 0.94, 'r13': 0.98, 'r15': c['all variants agree']}}
json.dump(summary, open(f'{OUT}/D_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('R15 DONE')'''),
]


def main():
    nb = {'nbformat': 4, 'nbformat_minor': 0, 'metadata': {'colab': {'provenance': []}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
          'accelerator': 'GPU'}, 'cells': []}
    for kind, src in CELLS:
        cell = {'cell_type': kind, 'metadata': {}, 'source': src}
        if kind == 'code':
            cell.update({'execution_count': None, 'outputs': []})
        nb['cells'].append(cell)
    path = os.path.join(HERE, 'r15_colab.ipynb')
    json.dump(nb, open(path, 'w'), indent=1)
    print(path)


if __name__ == '__main__':
    main()
