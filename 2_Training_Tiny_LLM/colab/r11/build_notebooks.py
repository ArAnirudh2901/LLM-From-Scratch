"""Writes the r11 Kaggle pipeline (same helpers and rules as colab/r9p): r11 = r8p + focus rows (build_focus.py:
the whole question, then one sentence -> that sentence's facts), read at test time with --focus_search (each
sentence in context, joined under the engine). --focus_mask keeps the question blind to the sentence, so at test
time one question's keys/values serve all its sentences (exact; tests/test_focus.py). r10's gate, Canon layers and
correction rows are left out (no gain; Canon would also block the shared question).
    python colab/r11/build_notebooks.py ; PIPE=r11 colab/r9p/run_step.sh C   (then D; E only with owner approval)
r8p baselines come from the r9p runs: step A (dev, FINAL setting) and step B1 (suites).
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r9p'))
import build_notebooks as r9   # noqa: E402

USER = r9.USER
r9.HERE = os.path.dirname(os.path.abspath(__file__))
r9.SLUG = {'C': 'kinthread-r11-c-train', 'D': 'kinthread-r11-d-select', 'E': 'kinthread-r11-e-locked'}
code, COMMON, R8P = r9.code, r9.COMMON, r9.R8P

C = [code(COMMON + R8P + '''
sh('pip -q install -U gdown')
S = '/kaggle/temp/share'
os.makedirs(S, exist_ok=True)
if not os.path.isdir(f'{S}/dataset_v7'):
    sh(f'gdown -q --folder https://drive.google.com/drive/folders/1MNiQuramSX5ImytxEdvNEIN7SgFtT6k8 -O {S}/dataset_v7')
if not os.path.exists(f'{S}/star_r3.jsonl.gz'):
    sh(f'wget -q -O {S}/star_r3.jsonl.gz "https://drive.usercontent.google.com/download?id=1_qk0JL_lkzKBdv2vwHxWijI4yNrNPRtr&export=download&confirm=t"')
DATA, OUT = f'{S}/dataset_v7', f'{W}/runs/v10-r11'
os.makedirs(OUT, exist_ok=True)
sh('cd data && sha256sum -c SHA256SUMS --ignore-missing --quiet')
sh('python -m pytest -q tests/test_model_reader.py tests/test_focus.py')
'''), code('''
import gzip
FOCUS = f'{W}/focus_v1.jsonl.gz'   # in the outputs: the rows that trained r11 stay on record
sh(f'python build_focus.py --n 60000 --workers 4 --out {FOCUS}')
n_focus = sum(1 for _ in gzip.open(FOCUS, 'rt'))
if n_focus < 54000:
    stop(f'only {n_focus} focus rows (asked 60000)')
EXTRA = [f'{DATA}/reworded.jsonl.gz*2', 'data/mix_r8.jsonl.gz', f'{S}/star_r3.jsonl.gz*2',
         'data/balanced_mix_v1.jsonl.gz*2', 'data/detour_v1.jsonl.gz']
missing = [e for e in EXTRA if not os.path.exists(e.split('*')[0])]
if missing:
    stop(f'training files missing: {missing}')
ENV = 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '
TRAIN = ('python train_reader.py --arch v10 --run v10-r11 '
         f'--pool {DATA} --init runs/v10-r8p-final/best.pt --prefix_lm --of_form --style_p 0.8 '
         '--batch 64 --lr 3e-4 --warmup 500 --optim muon --amp --workers 2 '
         '--save_every 1000 --reworded ' + ','.join(EXTRA) + ' --reworded_dev data/reworded_dev.jsonl.gz '
         f'--p_reworded 0.55 --focus {FOCUS} --p_focus 0.2 --focus_mask --ramp_steps 2000 --block 512')
'''), code('''
import re
# compile smoke test (first run with it on a T4): 20 min each way, one T4 each; keep compile only if faster and sane
par(f'{ENV}timeout 1200 {TRAIN} --compile --steps 100000 --eval_every 100000 --save_every 100000 --out {W}/smoke_compile > {W}/smoke_compile.log 2>&1 || true',
    f'{ENV}timeout 1200 {TRAIN} --no-compile --steps 100000 --eval_every 100000 --save_every 100000 --out {W}/smoke_no-compile > {W}/smoke_no-compile.log 2>&1 '
    '|| test $? -eq 124')   # a compile failure falls back to eager; an eager failure stops here
def rate(m):
    rows = [(int(a), float(b), float(c), float(d)) for a, b, c, d in
            re.findall(r'step +(\\d+) \\| facts ([\\d.]+) .*? (\\d+) q/s \\| ([\\d.]+) min', open(f'{W}/smoke_{m}.log').read())]
    if len(rows) < 3:
        return 0.0, float('nan')
    (_, _, q0, t0), (_, loss, q1, t1) = rows[1], rows[-1]   # from step 200: compile warm-up left out
    return (q1 * t1 - q0 * t0) / max(0.1, t1 - t0), loss   # questions per second between them, last loss
(rc, lc), (re_, le) = rate('compile'), rate('no-compile')
if re_ == 0.0:
    stop('eager smoke run logged under 3 steps lines: check smoke_no-compile.log')
COMPILE = '--compile' if rc > 1.05 * re_ and lc == lc and lc < 1.5 * le + 0.05 else '--no-compile'
num = lambda x: x if x == x else None   # JSON has no NaN
smoke = {'compile_q_per_s': round(rc, 1), 'eager_q_per_s': round(re_, 1), 'compile_loss': num(lc), 'eager_loss': num(le), 'chosen': COMPILE}
print(smoke)
sh(f'rm -rf {W}/smoke_compile {W}/smoke_no-compile')
'''), code('''
sh(f'{ENV}{TRAIN} {COMPILE} --out {OUT} --steps 15000 --resume --eval_every 1500 --real_dev 575 --select_real')
'''), code('''
sh(f'rm -f {OUT}/latest.pt')
best = f'{OUT}/best.pt'
if not os.path.exists(best):
    stop('training wrote no best.pt')
par(f'python -m eval.run_real --ckpt {best} --split dev --adaptive 1.0 --hybrid --out {PRIV}/r11_dev_quick.json',
    f'python -m eval.run_real --ckpt {best} --split dev --adaptive 1.0 --hybrid --focus_search --out {PRIV}/r11_dev_quick_focus.json')
quick = {n: json.load(open(f'{PRIV}/r11_dev_quick{n}.json'))['score']['splits'] for n in ('', '_focus')}
summary = {'step': 'C', 'r11_sha256': sha(best), 'focus_rows': n_focus, 'focus_rows_sha256': sha(FOCUS), 'compile': smoke,
           'r11_dev_quick_adaptive_hybrid': quick[''], 'r11_dev_quick_focus': quick['_focus'],
           'r9p_same_setting': '520/714 = 72.8% fair (results/r9p/C)'}
json.dump(summary, open(f'{PUB}/C_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('C DONE')
''')]

D = [code(COMMON + '''
R11 = one('runs/v10-r11/best.pt', 'the output of kinthread-r11-c-train')
A_DEV = one('private/r8p_dev_final.json', 'the output of kinthread-r9p-a-baseline')
B1 = json.load(open(one('public/B_r8p_summary.json', 'the output of kinthread-r9p-b1-suites-r8p')))
print('r11', sha(R11)[:12])
'''), code('''
# each config is judged with the verifier and suites of its own reading path (focus joins are new to a verifier)
par(f'python train_verifier.py --ckpt {R11} --n 1500 --out {PUB}/r11_verifier.json && '
    f'python train_verifier.py --ckpt {R11} --n 1500 --split_search --focus_search --out {PUB}/r11_verifier_focus.json',
    f'python -m eval.gauntlet --ckpt {R11} --k 16 --out {PUB}/r11_gauntlet.json && '
    f'python -m eval.consistency --groups 300 --ckpt {R11} --k 16 --out {PUB}/r11_consistency.json && '
    f'python -m eval.evaluate --ckpt {R11} --sets unseen --out {PUB}/r11_unseen.json && '
    f'python -m eval.gauntlet --ckpt {R11} --k 16 --focus_search --out {PUB}/r11_gauntlet_focus.json && '
    f'python -m eval.consistency --groups 300 --ckpt {R11} --k 16 --focus_search --out {PUB}/r11_consistency_focus.json')
BASE = '--adaptive 1.0 --hybrid --cover --split_search --para 2 --k 16'
EXTRA_FLAGS = {'final': '', 'focus': '--focus_search'}   # focus: split-search joins plus focus joins
VERS = {'final': 'r11_verifier.json', 'focus': 'r11_verifier_focus.json'}
SUITES = {'final': '', 'focus': '_focus'}
par(*[f'python -m eval.run_real --ckpt {R11} --split dev {BASE} --verifier {PUB}/{VERS[name]} {extra} --out {PRIV}/r11_dev_{name}.json'
      for name, extra in EXTRA_FLAGS.items()])
for name in EXTRA_FLAGS:
    sh(f'python -m eval.compare_runs {A_DEV} {PRIV}/r11_dev_{name}.json --out {PUB}/compare_{name}.json')
'''), code('''
cmp = {n: json.load(open(f'{PUB}/compare_{n}.json')) for n in EXTRA_FLAGS}
passing = [n for n in cmp if cmp[n]['adopt']]   # a config meeting the rule beats a bigger non-significant gain
best = max(passing or cmp, key=lambda n: cmp[n]['gain_points'])
g, c = (json.load(open(f'{PUB}/r11_{x}{SUITES[best]}.json')) for x in ('gauntlet', 'consistency'))   # the adopted path's suites
suites_ok = (g['correct'] >= B1['gauntlet_correct'] - 0.01 and c['all variants agree'] >= B1['consistency_all_agree'] - 0.01)
decision = {'step': 'D', 'adopt': bool(cmp[best]['adopt'] and suites_ok), 'config': best,
            'extra_flags': EXTRA_FLAGS[best], 'verifier': VERS[best], 'dev': cmp[best], 'all_configs': cmp,
            'gauntlet': {'r8p': B1['gauntlet_correct'], 'r11': g['correct']},
            'consistency_all_agree': {'r8p': B1['consistency_all_agree'], 'r11': c['all variants agree']},
            'suites_not_worse': suites_ok, 'r11_sha256': sha(R11),
            'rule': 'adopt = dev fair gain >= 2 points, wins > losses, McNemar p < 0.05, gauntlet and consistency not lower by more than 1 point'}
json.dump(decision, open(f'{PUB}/decision.json', 'w'), indent=1)
print(json.dumps(decision, indent=1))
print('D DONE: ADOPT r11' if decision['adopt'] else 'D DONE: KEEP r8p')
''')]

E = copy.deepcopy(r9.E)   # locked test: same checks, r11 names
for cell in E:
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    src = src.replace('kinthread-r9p-', 'kinthread-r11-').replace('r9p', 'r11')
    cell['source'] = src.replace("VER = one('public/r11_verifier.json'", "VER = one('public/' + dec.get('verifier', 'r11_verifier.json')")

if __name__ == '__main__':
    print(r9.write('C', '# r11 step C: focus rows, compile smoke test, then r11 = r8p fine-tuned with focus rows\nAbout 6-8 h.', C, []))
    print(r9.write('D', '# r11 step D: verifiers and suites per reading path, dev with split search and with split + focus search vs r8p, suites, adopt or keep\nAbout 4-6 h.', D,
                   [f'{USER}/kinthread-r9p-a-baseline', f'{USER}/kinthread-r9p-b1-suites-r8p', f"{USER}/{r9.SLUG['C']}"]))
    print(r9.write('E', '# r11 step E: locked test, once, only with the owner approval\nAbout 1-2 h.', E,
                   [f"{USER}/{r9.SLUG[k]}" for k in ('C', 'D')]))
