"""Writes the r12 Kaggle pipeline (same helpers and rules as colab/r9p): r12 = r9p's recipe (r8p warm start, same
data and steps) plus a grammar encoder (genc.py: Ettin-68M, MIT, open data) whose view of the question enters through
a zero-started projection, so training starts exactly at r8p. r9p is the matched control: any gain over it is the
encoder's.
    python colab/r12/build_notebooks.py ; PIPE=r12 colab/r9p/run_step.sh C   (then D; E only with owner approval)
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r9p'))
import build_notebooks as r9   # noqa: E402

USER = r9.USER
r9.HERE = os.path.dirname(os.path.abspath(__file__))
r9.SLUG = {'C': 'kinthread-r12-c-train', 'T': 'kinthread-r12-t-twins', 'D': 'kinthread-r12-d-select', 'E': 'kinthread-r12-e-locked'}
code, R8P = r9.code, r9.R8P
COMMON = r9.COMMON + '''sh('pip -q install "transformers>=4.48"')   # ModernBERT (the grammar encoder)
'''
GENC = '--genc jhu-clsp/ettin-encoder-68m --genc_rev ac19ae4bc51093b31c475665ac872a936d056cc2 --genc_lr 5e-5 --genc_freeze 1000'

C = [code(COMMON + R8P + '''
sh('pip -q install -U gdown')
S = '/kaggle/temp/share'
os.makedirs(S, exist_ok=True)
if not os.path.isdir(f'{S}/dataset_v7'):
    sh(f'gdown -q --folder https://drive.google.com/drive/folders/1MNiQuramSX5ImytxEdvNEIN7SgFtT6k8 -O {S}/dataset_v7')
if not os.path.exists(f'{S}/star_r3.jsonl.gz'):
    sh(f'wget -q -O {S}/star_r3.jsonl.gz "https://drive.usercontent.google.com/download?id=1_qk0JL_lkzKBdv2vwHxWijI4yNrNPRtr&export=download&confirm=t"')
DATA, OUT = f'{S}/dataset_v7', f'{W}/runs/v10-r12'
os.makedirs(OUT, exist_ok=True)
sh('cd data && sha256sum -c SHA256SUMS --ignore-missing --quiet')
sh('python -m pytest -q tests/test_model_reader.py tests/test_genc.py')
'''), code('''
EXTRA = [f'{DATA}/reworded.jsonl.gz*2', 'data/mix_r8.jsonl.gz', f'{S}/star_r3.jsonl.gz*2',
         'data/balanced_mix_v1.jsonl.gz*2', 'data/detour_v1.jsonl.gz']
missing = [e for e in EXTRA if not os.path.exists(e.split('*')[0])]
if missing:
    stop(f'training files missing: {missing}')
sh('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_reader.py --arch v10 --run v10-r12 '
   f'--out {OUT} --pool {DATA} --init runs/v10-r8p-final/best.pt --prefix_lm --of_form --style_p 0.8 '
   '--steps 9000 --batch 64 --lr 3e-4 --warmup 500 --optim muon --amp --workers 2 --resume --compile '
   '--eval_every 1500 --save_every 1000 --reworded ' + ','.join(EXTRA) + ' --reworded_dev data/reworded_dev.jsonl.gz '
   '--p_reworded 0.75 --ramp_steps 2000 --block 512 --real_dev 575 --select_real ''' + GENC + '''')
'''), code('''
sh(f'rm -f {OUT}/latest.pt')
best = f'{OUT}/best.pt'
if not os.path.exists(best):
    stop('training wrote no best.pt')
sh(f'python -m eval.run_real --ckpt {best} --split dev --adaptive 1.0 --hybrid --out {PRIV}/r12_dev_quick.json')
quick = json.load(open(f'{PRIV}/r12_dev_quick.json'))['score']['splits']
summary = {'step': 'C', 'r12_sha256': sha(best), 'r12_dev_quick_adaptive_hybrid': quick,
           'r9p_same_setting': '520/714 = 72.8% fair (results/r9p/C; the matched control without the encoder)'}
json.dump(summary, open(f'{PUB}/C_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('C DONE')
''')]

# T: r12 continues from its own checkpoint with twin training (twins.py): same question, another sentence order or
# sentence breaks, same facts; a symmetric KL makes the readings match (repeatability). The encoder is already trained.
T = [code(COMMON + """
sh('pip -q install -U gdown')
R12C = one('runs/v10-r12/best.pt', 'the output of kinthread-r12-c-train')
S = '/kaggle/temp/share'
os.makedirs(S, exist_ok=True)
if not os.path.isdir(f'{S}/dataset_v7'):
    sh(f'gdown -q --folder https://drive.google.com/drive/folders/1MNiQuramSX5ImytxEdvNEIN7SgFtT6k8 -O {S}/dataset_v7')
if not os.path.exists(f'{S}/star_r3.jsonl.gz'):
    sh(f'wget -q -O {S}/star_r3.jsonl.gz "https://drive.usercontent.google.com/download?id=1_qk0JL_lkzKBdv2vwHxWijI4yNrNPRtr&export=download&confirm=t"')
DATA, OUT = f'{S}/dataset_v7', f'{W}/runs/v10-r12t'
os.makedirs(OUT, exist_ok=True)
sh('cd data && sha256sum -c SHA256SUMS --ignore-missing --quiet')
sh('python -m pytest -q tests/test_genc.py tests/test_twins.py')
"""), code("""
EXTRA = [f'{DATA}/reworded.jsonl.gz*2', 'data/mix_r8.jsonl.gz', f'{S}/star_r3.jsonl.gz*2',
         'data/balanced_mix_v1.jsonl.gz*2', 'data/detour_v1.jsonl.gz']
sh('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_reader.py --arch v10 --run v10-r12t '
   f'--out {OUT} --pool {DATA} --init {R12C} --prefix_lm --of_form --style_p 0.8 '
   '--steps 4000 --batch 64 --lr 1e-4 --warmup 200 --optim muon --amp --workers 2 --resume --compile '
   '--eval_every 1000 --save_every 1000 --reworded ' + ','.join(EXTRA) + ' --reworded_dev data/reworded_dev.jsonl.gz '
   '--p_reworded 0.75 --block 512 --real_dev 575 --select_real --genc_freeze 0 --genc_lr 5e-5 --cons_p 0.4 --cons_w 0.5')
"""), code("""
sh(f'rm -f {OUT}/latest.pt')
best = f'{OUT}/best.pt'
if not os.path.exists(best):
    stop('training wrote no best.pt')
sh(f'python -m eval.run_real --ckpt {best} --split dev --adaptive 1.0 --hybrid --out {PRIV}/r12t_dev_quick.json')
quick = json.load(open(f'{PRIV}/r12t_dev_quick.json'))['score']['splits']
summary = {'step': 'T', 'r12t_sha256': sha(best), 'from': sha(R12C), 'r12t_dev_quick_adaptive_hybrid': quick}
json.dump(summary, open(f'{PUB}/T_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('T DONE')
""")]

D = [code(COMMON + '''
R12 = one('runs/v10-r12t/best.pt', 'the output of kinthread-r12-t-twins')   # r12 after twin training
A_DEV = one('private/r8p_dev_final.json', 'the output of kinthread-r9p-a-baseline')
B1 = json.load(open(one('public/B_r8p_summary.json', 'the output of kinthread-r9p-b1-suites-r8p')))
print('r12', sha(R12)[:12])
'''), code('''
par(f'python train_verifier.py --ckpt {R12} --n 1500 --out {PUB}/r12_verifier.json',
    f'python -m eval.gauntlet --ckpt {R12} --k 16 --out {PUB}/r12_gauntlet.json && '
    f'python -m eval.consistency --groups 300 --ckpt {R12} --k 16 --out {PUB}/r12_consistency.json && '
    f'python -m eval.evaluate --ckpt {R12} --sets unseen --out {PUB}/r12_unseen.json')
BASE = '--adaptive 1.0 --hybrid --cover --split_search --para 2 --k 16'
sh(f'python -m eval.run_real --ckpt {R12} --split dev {BASE} --verifier {PUB}/r12_verifier.json --out {PRIV}/r12_dev_final.json')
sh(f'python -m eval.compare_runs {A_DEV} {PRIV}/r12_dev_final.json --out {PUB}/compare_final.json')
'''), code('''
cmp = json.load(open(f'{PUB}/compare_final.json'))
g, c = (json.load(open(f'{PUB}/r12_{x}.json')) for x in ('gauntlet', 'consistency'))
suites_ok = (g['correct'] >= B1['gauntlet_correct'] - 0.01 and c['all variants agree'] >= B1['consistency_all_agree'] - 0.01)
decision = {'step': 'D', 'adopt': bool(cmp['adopt'] and suites_ok), 'config': 'final', 'extra_flags': '',
            'verifier': 'r12_verifier.json', 'dev': cmp,
            'gauntlet': {'r8p': B1['gauntlet_correct'], 'r12': g['correct']},
            'consistency_all_agree': {'r8p': B1['consistency_all_agree'], 'r12': c['all variants agree']},
            'suites_not_worse': suites_ok, 'r12_sha256': sha(R12),
            'rule': 'adopt = dev fair gain >= 2 points, wins > losses, McNemar p < 0.05, gauntlet and consistency not lower by more than 1 point'}
json.dump(decision, open(f'{PUB}/decision.json', 'w'), indent=1)
print(json.dumps(decision, indent=1))
print('D DONE: ADOPT r12' if decision['adopt'] else 'D DONE: KEEP r8p')
''')]

E = copy.deepcopy(r9.E)   # locked test: same checks, r12 names
for i, cell in enumerate(E):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if i == 0:
        src = src.replace(r9.COMMON, COMMON, 1)
    src = src.replace('kinthread-r9p-', 'kinthread-r12-').replace('r9p', 'r12')
    cell['source'] = src.replace('runs/v10-r12/', 'runs/v10-r12t/').replace('kinthread-r12-c-train', 'kinthread-r12-t-twins')   # the model after twins

if __name__ == '__main__':
    print(r9.write('C', '# r12 step C: r12 = r8p + grammar encoder (Ettin-68M), r9p recipe\nAbout 4-5 h.', C, []))
    print(r9.write('T', '# r12 step T: twin training (repeatability) from r12 step C\nAbout 2 h.', T, [f"{USER}/{r9.SLUG['C']}"]))
    print(r9.write('D', '# r12 step D: verifier, suites, dev FINAL setting vs r8p, adopt or keep\nAbout 2-3 h.', D,
                   [f'{USER}/kinthread-r9p-a-baseline', f'{USER}/kinthread-r9p-b1-suites-r8p', f"{USER}/{r9.SLUG['T']}"]))
    print(r9.write('E', '# r12 step E: locked test, once, only with the owner approval\nAbout 1-2 h.', E,
                   [f"{USER}/{r9.SLUG[k]}" for k in ('T', 'D')]))
