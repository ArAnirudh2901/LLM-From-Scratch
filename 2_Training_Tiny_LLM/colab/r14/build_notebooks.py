"""Writes the r14 Kaggle pipeline (same helpers and rules as colab/r9p): r14 = r13 (r8p + twin training; consistency 0.98)
+ teacher rows for the generated exam questions r8p failed on (teach2.py: Gemini writes the facts, kept only when our
engine reaches the answer key; data/teach_sg3.jsonl.gz), twins still on.
    python colab/r14/build_notebooks.py ; PIPE=r14 colab/r9p/run_step.sh C   (then D; E only with owner approval)
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r9p'))
import build_notebooks as r9   # noqa: E402

USER = r9.USER
r9.HERE = os.path.dirname(os.path.abspath(__file__))
r9.SLUG = {'C': 'kinthread-r14-c-teach', 'D': 'kinthread-r14-d-select', 'E': 'kinthread-r14-e-locked'}
code, COMMON, R8P = r9.code, r9.COMMON, r9.R8P

C = [code(COMMON + '''
R13 = one('runs/v10-r13/best.pt', 'the output of kinthread-r13-c-twins')
sh('pip -q install -U gdown')
S = '/kaggle/temp/share'
os.makedirs(S, exist_ok=True)
if not os.path.isdir(f'{S}/dataset_v7'):
    sh(f'gdown -q --folder https://drive.google.com/drive/folders/1MNiQuramSX5ImytxEdvNEIN7SgFtT6k8 -O {S}/dataset_v7')
if not os.path.exists(f'{S}/star_r3.jsonl.gz'):
    sh(f'wget -q -O {S}/star_r3.jsonl.gz "https://drive.usercontent.google.com/download?id=1_qk0JL_lkzKBdv2vwHxWijI4yNrNPRtr&export=download&confirm=t"')
DATA, OUT = f'{S}/dataset_v7', f'{W}/runs/v10-r14'
os.makedirs(OUT, exist_ok=True)
sh('cd data && sha256sum -c SHA256SUMS --ignore-missing --quiet')
sh('python -m pytest -q tests/test_model_reader.py tests/test_twins.py')
'''), code('''
EXTRA = [f'{DATA}/reworded.jsonl.gz*2', 'data/mix_r8.jsonl.gz', f'{S}/star_r3.jsonl.gz*2',
         'data/balanced_mix_v1.jsonl.gz*2', 'data/detour_v1.jsonl.gz', 'data/teach_sg3.jsonl.gz*4']
missing = [e for e in EXTRA if not os.path.exists(e.split('*')[0])]
if missing:
    stop(f'training files missing: {missing}')
sh('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_reader.py --arch v10 --run v10-r14 '
   f'--out {OUT} --pool {DATA} --init {R13} --prefix_lm --of_form --style_p 0.8 '
   '--steps 4000 --batch 64 --lr 1e-4 --warmup 300 --optim muon --amp --workers 2 --resume --compile '
   '--eval_every 1000 --save_every 500 --reworded ' + ','.join(EXTRA) + ' --reworded_dev data/reworded_dev.jsonl.gz '
   '--p_reworded 0.75 --block 512 --real_dev 575 --select_real --cons_p 0.5 --cons_w 0.5')
'''), code('''
sh(f'rm -f {OUT}/latest.pt')
best = f'{OUT}/best.pt'
if not os.path.exists(best):
    stop('training wrote no best.pt')
sh(f'python -m eval.run_real --ckpt {best} --split dev --adaptive 1.0 --hybrid --out {PRIV}/r14_dev_quick.json')
quick = json.load(open(f'{PRIV}/r14_dev_quick.json'))['score']['splits']
summary = {'step': 'C', 'r14_sha256': sha(best), 'r14_dev_quick_adaptive_hybrid': quick,
           'r13_same_setting': '73.4% fair (quick, adaptive + hybrid)', 'r8p_same_setting': '73.0%'}
json.dump(summary, open(f'{PUB}/C_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('C DONE')
''')]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r12'))
import importlib.util   # noqa: E402
_spec = importlib.util.spec_from_file_location('r12b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r12', 'build_notebooks.py'))
_r12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r12)
r9.HERE = os.path.dirname(os.path.abspath(__file__))   # loading r12's builder pointed the helpers at its folder
r9.SLUG = {'C': 'kinthread-r14-c-teach', 'D': 'kinthread-r14-d-select', 'E': 'kinthread-r14-e-locked'}


def _rename(cells):
    out = copy.deepcopy(cells)
    for cell in out:
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        src = src.replace(_r12.COMMON, COMMON)   # no grammar encoder: no transformers install
        src = src.replace('runs/v10-r12t/best.pt', 'runs/v10-r14/best.pt').replace('kinthread-r12-t-twins', 'kinthread-r14-c-teach')
        cell['source'] = src.replace('kinthread-r12-', 'kinthread-r14-').replace('r12', 'r14')
    return out


D, E = _rename(_r12.D), _rename(_r12.E)

if __name__ == '__main__':
    print(r9.write('C', '# r14 step C: r13 + teacher rows, twins on\nAbout 1.5 h.', C, [f'{USER}/kinthread-r13-c-twins']))
    print(r9.write('D', '# r14 step D: verifier, suites, dev FINAL setting vs r8p, adopt or keep\nAbout 2-3 h.', D,
                   [f'{USER}/kinthread-r9p-a-baseline', f'{USER}/kinthread-r9p-b1-suites-r8p', f"{USER}/{r9.SLUG['C']}"]))
    print(r9.write('E', '# r14 step E: locked test, once, only with the owner approval\nAbout 1-2 h.', E,
                   [f"{USER}/{r9.SLUG[k]}" for k in ('C', 'D')]))
