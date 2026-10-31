"""Writes the r10 Kaggle pipeline (same helpers and rules as colab/r9p): r10 = r8p + gated attention + Canon layers
(both start as identity) + engine-feedback correction rows, then the step-D comparison with r8p.
    python colab/r10/build_notebooks.py ; PIPE=r10 colab/r9p/run_step.sh C   (then D; E only with owner approval)
r8p baselines come from the r9p runs: step A (dev, FINAL setting) and step B1 (suites).
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r9p'))
import build_notebooks as r9   # noqa: E402

USER = r9.USER
r9.HERE = os.path.dirname(os.path.abspath(__file__))
r9.SLUG = {'C': 'kinthread-r10-c-train', 'D': 'kinthread-r10-d-select', 'E': 'kinthread-r10-e-locked'}
code, COMMON, R8P = r9.code, r9.COMMON, r9.R8P

C = [code(COMMON + R8P + '''
sh('pip -q install -U gdown')
S = '/kaggle/temp/share'
os.makedirs(S, exist_ok=True)
if not os.path.isdir(f'{S}/dataset_v7'):
    sh(f'gdown -q --folder https://drive.google.com/drive/folders/1MNiQuramSX5ImytxEdvNEIN7SgFtT6k8 -O {S}/dataset_v7')
if not os.path.exists(f'{S}/star_r3.jsonl.gz'):
    sh(f'wget -q -O {S}/star_r3.jsonl.gz "https://drive.usercontent.google.com/download?id=1_qk0JL_lkzKBdv2vwHxWijI4yNrNPRtr&export=download&confirm=t"')
DATA, OUT = f'{S}/dataset_v7', f'{W}/runs/v10-r10'
os.makedirs(OUT, exist_ok=True)
sh('cd data && sha256sum -c SHA256SUMS --ignore-missing --quiet')
sh('python -m pytest -q tests/test_model_reader.py')
'''), code('''
import gzip
# engine-feedback rows: r8p drafts on verified questions, the engine's verdict, the true facts (build_corrections.py)
CORR = f'{S}/corrections_r8p.jsonl.gz'
sh(f'timeout 5400 python build_corrections.py --ckpt runs/v10-r8p-final/best.pt --out {CORR} --pool {DATA} '
   f'--per_kind 1500 --extra {DATA}/reworded.jsonl.gz {S}/star_r3.jsonl.gz data/balanced_mix_v1.jsonl.gz '
   f'data/detour_v1.jsonl.gz --n 20000 || test $? -eq 124')   # 124: the 90-minute cap; rows so far are kept
n_corr = sum(1 for _ in gzip.open(CORR, 'rt')) if os.path.exists(CORR) else 0
print('correction rows', n_corr)
if n_corr < 1000:
    stop(f'only {n_corr} correction rows')
shutil.copy(CORR, f'{W}/corrections_r8p.jsonl.gz')
'''), code('''
EXTRA = [f'{DATA}/reworded.jsonl.gz*2', 'data/mix_r8.jsonl.gz', f'{S}/star_r3.jsonl.gz*2',
         'data/balanced_mix_v1.jsonl.gz*2', 'data/detour_v1.jsonl.gz', f'{CORR}*2']
missing = [e for e in EXTRA if not os.path.exists(e.split('*')[0])]
if missing:
    stop(f'training files missing: {missing}')
sh('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_reader.py --arch v10 --run v10-r10 '
   f'--out {OUT} --pool {DATA} --init runs/v10-r8p-final/best.pt --prefix_lm --of_form --gate_attn --canon --style_p 0.8 '
   '--steps 15000 --batch 64 --lr 3e-4 --warmup 500 --optim muon --amp --workers 2 --resume '
   '--eval_every 1500 --save_every 1000 --reworded ' + ','.join(EXTRA) + ' --reworded_dev data/reworded_dev.jsonl.gz '
   '--p_reworded 0.75 --ramp_steps 2000 --block 512 --real_dev 575 --select_real')
'''), code('''
sh(f'rm -f {OUT}/latest.pt')
best = f'{OUT}/best.pt'
if not os.path.exists(best):
    stop('training wrote no best.pt')
par(f'python -m eval.run_real --ckpt {best} --split dev --adaptive 1.0 --hybrid --out {PRIV}/r10_dev_quick.json',
    f'python -m eval.run_real --ckpt {best} --split dev --adaptive 1.0 --hybrid --correct --out {PRIV}/r10_dev_quick_correct.json')
quick = {n: json.load(open(f'{PRIV}/r10_dev_quick{n}.json'))['score']['splits'] for n in ('', '_correct')}
summary = {'step': 'C', 'r10_sha256': sha(best), 'correction_rows': n_corr, 'r10_dev_quick_adaptive_hybrid': quick[''],
           'r10_dev_quick_correct': quick['_correct'], 'r9p_same_setting': '520/714 = 72.8% fair (results/r9p/C)'}
json.dump(summary, open(f'{PUB}/C_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('C DONE')
''')]

D = [code(COMMON + '''
R10 = one('runs/v10-r10/best.pt', 'the output of kinthread-r10-c-train')
A_DEV = one('private/r8p_dev_final.json', 'the output of kinthread-r9p-a-baseline')
B1 = json.load(open(one('public/B_r8p_summary.json', 'the output of kinthread-r9p-b1-suites-r8p')))
print('r10', sha(R10)[:12])
'''), code('''
par(f'python train_verifier.py --ckpt {R10} --n 1500 --out {PUB}/r10_verifier.json',   # suites need no verifier: other T4
    f'python -m eval.gauntlet --ckpt {R10} --k 16 --out {PUB}/r10_gauntlet.json && '
    f'python -m eval.consistency --groups 300 --ckpt {R10} --k 16 --out {PUB}/r10_consistency.json && '
    f'python -m eval.evaluate --ckpt {R10} --sets unseen --out {PUB}/r10_unseen.json')
BASE = f'--adaptive 1.0 --hybrid --verifier {PUB}/r10_verifier.json --cover --split_search --para 2 --k 16'
EXTRA_FLAGS = {'final': '', 'final_correct': '--correct'}
par(*[f'python -m eval.run_real --ckpt {R10} --split dev {BASE} {extra} --out {PRIV}/r10_dev_{name}.json'
      for name, extra in EXTRA_FLAGS.items()])
for name in EXTRA_FLAGS:
    sh(f'python -m eval.compare_runs {A_DEV} {PRIV}/r10_dev_{name}.json --out {PUB}/compare_{name}.json')
'''), code('''
g, c = json.load(open(f'{PUB}/r10_gauntlet.json')), json.load(open(f'{PUB}/r10_consistency.json'))
cmp = {n: json.load(open(f'{PUB}/compare_{n}.json')) for n in EXTRA_FLAGS}
passing = [n for n in cmp if cmp[n]['adopt']]   # a config meeting the rule beats a bigger non-significant gain
best = max(passing or cmp, key=lambda n: cmp[n]['gain_points'])
suites_ok = (g['correct'] >= B1['gauntlet_correct'] - 0.01 and c['all variants agree'] >= B1['consistency_all_agree'] - 0.01)
decision = {'step': 'D', 'adopt': bool(cmp[best]['adopt'] and suites_ok), 'config': best,
            'extra_flags': EXTRA_FLAGS[best], 'dev': cmp[best], 'all_configs': cmp,
            'gauntlet': {'r8p': B1['gauntlet_correct'], 'r10': g['correct']},
            'consistency_all_agree': {'r8p': B1['consistency_all_agree'], 'r10': c['all variants agree']},
            'suites_not_worse': suites_ok, 'r10_sha256': sha(R10),
            'rule': 'adopt = dev fair gain >= 2 points, wins > losses, McNemar p < 0.05, gauntlet and consistency not lower by more than 1 point'}
json.dump(decision, open(f'{PUB}/decision.json', 'w'), indent=1)
print(json.dumps(decision, indent=1))
print('D DONE: ADOPT r10' if decision['adopt'] else 'D DONE: KEEP r8p')
''')]

E = copy.deepcopy(r9.E)   # locked test: same checks, r10 names
for cell in E:
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    cell['source'] = src.replace('kinthread-r9p-', 'kinthread-r10-').replace('r9p', 'r10')

if __name__ == '__main__':
    print(r9.write('C', '# r10 step C: correction rows from r8p, then r10 = r8p + gated attention + Canon layers\nAbout 5-7 h.', C, []))
    print(r9.write('D', '# r10 step D: verifier, dev with and without re-reading vs r8p, suites, adopt or keep\nAbout 4-6 h.', D,
                   [f'{USER}/kinthread-r9p-a-baseline', f'{USER}/kinthread-r9p-b1-suites-r8p', f"{USER}/{r9.SLUG['C']}"]))
    print(r9.write('E', '# r10 step E: locked test, once, only with the owner approval\nAbout 1-2 h.', E,
                   [f"{USER}/{r9.SLUG[k]}" for k in ('C', 'D')]))
