"""Writes the r9p Kaggle pipeline: one folder per step with notebook.ipynb and kernel-metadata.json, pushed with
`kaggle kernels push -p colab/r9p/<folder>`. Every cell stops the run with a 'STOP: ...' message when something is
wrong, and outputs go to /kaggle/working/public (counts only: safe to commit) or /kaggle/working/private (holds real
question text: never commit). Re-run this script after editing it:  python colab/r9p/build_notebooks.py
"""
import json
import os

USER = 'aranirudh'
# private dataset: real_questions_full.tgz, the r8p best.pt and FINAL.json + verifier.json (the r8p and
# final-selection notebooks belong to account anirudharavalli and cannot be attached from USER)
REAL = f'{USER}/kinthread-real-questions'
SLUG = {'A': 'kinthread-r9p-a-baseline', 'B1': 'kinthread-r9p-b1-suites-r8p', 'C': 'kinthread-r9p-c-train',
        'B2': 'kinthread-r9p-b2-suites-r9p', 'D': 'kinthread-r9p-d-select', 'E': 'kinthread-r9p-e-locked'}
HERE = os.path.dirname(os.path.abspath(__file__))

COMMON = '''import os, glob, json, hashlib, shutil, subprocess
W = '/kaggle/working'
PUB, PRIV = f'{W}/public', f'{W}/private'   # public: counts only, safe to commit; private: real question text, never commit
os.makedirs(PUB, exist_ok=True)
os.makedirs(PRIV, exist_ok=True)
REPO, BRANCH = '/kaggle/temp/Kinthread-llm-model', 'main'
R8P_SHA = '38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52'


def sh(cmd):
    """Runs one shell command; stops the notebook if it fails."""
    print('$', cmd, flush=True)
    if subprocess.run(cmd, shell=True).returncode:
        raise SystemExit(f'STOP: command failed: {cmd}')


def stop(msg):
    raise SystemExit('STOP: ' + msg)


def find(pattern):
    return sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))


def one(pattern, what):
    got = find(pattern)
    if not got:
        stop(f'attach {what} (no {pattern} under /kaggle/input)')
    return got[0]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


os.makedirs('/kaggle/temp', exist_ok=True)
if not os.path.exists(REPO):
    sh(f'git clone --depth 1 -b {BRANCH} https://github.com/ArAnirudh2901/Kinthread-llm-model.git {REPO}')
os.chdir(REPO)
tgz = find('real_questions_full.tgz')
if not tgz:
    stop('attach the private dataset kinthread-real-questions (it holds real_questions_full.tgz)')
sh(f'rm -rf data/real_questions && tar -xzf {tgz[0]}')
if not glob.glob('data/real_questions/*.jsonl'):
    stop('no real question files after unpacking real_questions_full.tgz')
print('repo and real questions ready')
'''

R8P = '''

def r8p_checkpoint():
    cands = find('best.pt')
    try:
        import kagglehub
        for h in ('anirudharavalli/notebook99ae0586bd/versions/7', 'anirudharavalli/notebook99ae0586bd/versions/5'):
            try:
                cands += glob.glob(f'{kagglehub.notebook_output_download(h)}/**/best.pt', recursive=True)
            except Exception as e:
                print(h, e)
    except Exception as e:
        print('kagglehub:', e)
    for p in cands:
        if sha(p) == R8P_SHA:
            return p
    stop(f'no best.pt with SHA-256 {R8P_SHA} among {cands}')


os.makedirs('runs/v10-r8p-final', exist_ok=True)
if not os.path.exists('runs/v10-r8p-final/best.pt'):
    shutil.copy(r8p_checkpoint(), 'runs/v10-r8p-final/best.pt')
print('r8p checkpoint ready', sha('runs/v10-r8p-final/best.pt')[:12])
'''

VERIFIER = '''
finals = [p for p in find('FINAL.json') if os.path.exists(os.path.join(os.path.dirname(p), 'verifier.json'))]
if not finals:
    stop('attach the final-selection notebook output (notebookf8798397a0: FINAL.json and verifier.json)')
shutil.copy(finals[0], 'runs/v10-r8p-final/FINAL.json')
shutil.copy(os.path.join(os.path.dirname(finals[0]), 'verifier.json'), 'runs/v10-r8p-final/verifier.json')
R8P_FINAL = '--adaptive 1.0 --hybrid --verifier runs/v10-r8p-final/verifier.json --cover --split_search --para 2 --k 16'
print('r8p FINAL setting:', R8P_FINAL)
'''

TESTS = '''sh('python -m pytest -q tests/test_answer_options.py tests/test_trace.py tests/test_locked.py '
   'tests/test_mine_patterns.py tests/test_compare_runs.py tests/test_gauntlet.py')
'''


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip().splitlines(keepends=True)}


def write(key, title, cells, kernels, gpu=True):
    folder = os.path.join(HERE, f'{key}_' + '-'.join(SLUG[key].split('-')[3:]))   # e.g. A_baseline
    os.makedirs(folder, exist_ok=True)
    nb = {'cells': [md(title)] + cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
          'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    with open(os.path.join(folder, 'notebook.ipynb'), 'w') as f:
        json.dump(nb, f, indent=1)
    meta = {'id': f'{USER}/{SLUG[key]}', 'title': SLUG[key], 'code_file': 'notebook.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': gpu, 'enable_internet': True,
            'dataset_sources': [REAL], 'kernel_sources': kernels, 'competition_sources': []}
    with open(os.path.join(folder, 'kernel-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    return folder


A = [code(COMMON + R8P + VERIFIER), code(TESTS), code('''
sh(f'python -m eval.run_real --ckpt runs/v10-r8p-final/best.pt --split dev {R8P_FINAL} '
   f'--trace {PRIV}/trace_dev.jsonl --trace_shadow --out {PRIV}/r8p_dev_final.json')
splits = json.load(open(f'{PRIV}/r8p_dev_final.json'))['score']['splits']
print(splits)
got = int(splits['dev fair'].split('/')[0])
if abs(got - 528) > 6:
    stop(f'r8p dev fair is {got}/714 but FINAL.json says 528/714: checkpoint, verifier or questions differ')
'''), code('''
sh(f'python -m eval.trace_report {PRIV}/trace_dev.jsonl --out {PRIV}/trace_dev_report.json')
sh(f'python -m eval.rescore_trace {PRIV}/trace_dev.jsonl --verifier runs/v10-r8p-final/verifier.json --cover '
   f'--out {PUB}/rescore_dev.json')
sh(f'python -m eval.mine_patterns --trace {PRIV}/trace_dev.jsonl --out {PUB}/patterns_dev.json')
sh(f'python -m eval.mine_patterns --trace {PRIV}/trace_dev.jsonl --gen_files data/mix_r8.jsonl.gz '
   f'data/balanced_mix_v1.jsonl.gz data/detour_v1.jsonl.gz --gen_n 30000 --out {PUB}/patterns_vs_training.json')
'''), code('''
rescore = json.load(open(f'{PUB}/rescore_dev.json'))
pat = json.load(open(f'{PUB}/patterns_dev.json'))
summary = {'step': 'A', 'r8p_dev': splits, 'flags_adopted': rescore['adopted_cli'], 'rescore': rescore['settings'],
           'top_gaps': [f"{g['family']}: {g['feature']} (real {g['real']:.1%}, generated {g['gen']:.1%})" for g in pat['gaps'][:15]],
           'top_misses': [f"{g['family']}: {g['feature']} (missed {g['missed']}, accuracy {g['model_acc']:.0%})" for g in pat['misses'][:15]]}
json.dump(summary, open(f'{PUB}/A_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('A DONE')
''')]


def suites(model):
    pick = ("CKPT = 'runs/v10-r8p-final/best.pt'" if model == 'r8p' else
            "CKPT = one('runs/v10-r9p/best.pt', 'the output of kinthread-r9p-c-train')")
    cells = [code(COMMON + R8P + VERIFIER + f"\nMODEL = '{model}'\n{pick}\nprint(MODEL, CKPT, sha(CKPT)[:12])\n"), code('''
sh(f'python -m eval.gauntlet --ckpt {CKPT} --k 16 --out {PUB}/{MODEL}_gauntlet.json')
sh(f'python -m eval.consistency --groups 300 --ckpt {CKPT} --k 16 --out {PUB}/{MODEL}_consistency.json')
sh(f'python -m eval.evaluate --ckpt {CKPT} --sets unseen --out {PUB}/{MODEL}_unseen.json')
''')]
    if model == 'r8p':
        cells.append(code('''
sh(f'python -m eval.run_real --ckpt {CKPT} --split dev {R8P_FINAL} --max_new 192 --reserve 8 --out {PRIV}/r8p_dev_final_192.json')
s192 = json.load(open(f'{PRIV}/r8p_dev_final_192.json'))['score']['splits']
json.dump(s192, open(f'{PUB}/r8p_dev_final_192_splits.json', 'w'), indent=1)
print(s192)
'''))
    cells.append(code('''
g = json.load(open(f'{PUB}/{MODEL}_gauntlet.json'))
c = json.load(open(f'{PUB}/{MODEL}_consistency.json'))
summary = {'step': f'B-{MODEL}', 'checkpoint_sha256': sha(CKPT), 'gauntlet_correct': g['correct'],
           'gauntlet_by_category': {k: v['correct'] for k, v in g['by category'].items()},
           'consistency_all_agree': c['all variants agree'], 'consistency_by_transform': c['by transform']}
json.dump(summary, open(f'{PUB}/B_{MODEL}_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('B DONE')
'''))
    return cells


C = [code(COMMON + R8P + '''
sh('pip -q install -U gdown')
S = '/kaggle/temp/share'
os.makedirs(S, exist_ok=True)
if not os.path.isdir(f'{S}/dataset_v7'):
    sh(f'gdown -q --folder https://drive.google.com/drive/folders/1MNiQuramSX5ImytxEdvNEIN7SgFtT6k8 -O {S}/dataset_v7')
if not os.path.exists(f'{S}/star_r3.jsonl.gz'):
    sh(f'wget -q -O {S}/star_r3.jsonl.gz "https://drive.usercontent.google.com/download?id=1_qk0JL_lkzKBdv2vwHxWijI4yNrNPRtr&export=download&confirm=t"')
DATA, OUT = f'{S}/dataset_v7', f'{W}/runs/v10-r9p'
os.makedirs(OUT, exist_ok=True)
sh('cd data && sha256sum -c SHA256SUMS --ignore-missing --quiet')
'''), code('''
EXTRA = [f'{DATA}/reworded.jsonl.gz*2', 'data/mix_r8.jsonl.gz', f'{S}/star_r3.jsonl.gz*2',
         'data/balanced_mix_v1.jsonl.gz*2', 'data/detour_v1.jsonl.gz']
missing = [e for e in EXTRA if not os.path.exists(e.split('*')[0])]
if missing:
    stop(f'training files missing: {missing}')
sh('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_reader.py --arch v10 --run v10-r9p '
   f'--out {OUT} --pool {DATA} --init runs/v10-r8p-final/best.pt --prefix_lm --of_form --style_p 0.8 '
   '--steps 15000 --batch 64 --lr 3e-4 --warmup 500 --optim muon --amp --workers 2 --resume '
   '--eval_every 1500 --save_every 1000 --reworded ' + ','.join(EXTRA) + ' --reworded_dev data/reworded_dev.jsonl.gz '
   '--p_reworded 0.75 --ramp_steps 2000 --block 512 --real_dev 575 --select_real')
'''), code('''
sh(f'rm -f {OUT}/latest.pt')
best = f'{OUT}/best.pt'
if not os.path.exists(best):
    stop('training wrote no best.pt')
sh(f'python -m eval.run_real --ckpt {best} --split dev --adaptive 1.0 --hybrid --out {PRIV}/r9p_dev_quick.json')
quick = json.load(open(f'{PRIV}/r9p_dev_quick.json'))['score']['splits']
summary = {'step': 'C', 'r9p_sha256': sha(best), 'r9p_dev_quick_adaptive_hybrid': quick,
           'r8p_dev_same_setting': 'see colab/r8p_kaggle.ipynb output (dev, --adaptive 1.0 --hybrid)'}
json.dump(summary, open(f'{PUB}/C_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('C DONE')
''')]

D = [code(COMMON + '''
R9P = one('runs/v10-r9p/best.pt', 'the output of kinthread-r9p-c-train')
A_DEV = one('private/r8p_dev_final.json', 'the output of kinthread-r9p-a-baseline')
FLAGS = json.load(open(one('public/rescore_dev.json', 'the output of kinthread-r9p-a-baseline')))['adopted_cli']
B1 = json.load(open(one('public/B_r8p_summary.json', 'the output of kinthread-r9p-b1-suites-r8p')))
B2 = json.load(open(one('public/B_r9p_summary.json', 'the output of kinthread-r9p-b2-suites-r9p')))
print('r9p', sha(R9P)[:12], '| flags adopted in step A:', FLAGS or 'none')
'''), code('''
sh(f'python train_verifier.py --ckpt {R9P} --n 1500 --out {PUB}/r9p_verifier.json')
BASE = f'--adaptive 1.0 --hybrid --verifier {PUB}/r9p_verifier.json --cover --split_search --para 2 --k 16'
EXTRA_FLAGS = {'final': '', 'final_plus': f'{FLAGS} --max_new 192 --reserve 8'.strip()}
for name, extra in EXTRA_FLAGS.items():
    sh(f'python -m eval.run_real --ckpt {R9P} --split dev {BASE} {extra} --out {PRIV}/r9p_dev_{name}.json')
    sh(f'python -m eval.compare_runs {A_DEV} {PRIV}/r9p_dev_{name}.json --out {PUB}/compare_{name}.json')
'''), code('''
cmp = {n: json.load(open(f'{PUB}/compare_{n}.json')) for n in EXTRA_FLAGS}
best = max(cmp, key=lambda n: cmp[n]['gain_points'])
suites_ok = (B2['gauntlet_correct'] >= B1['gauntlet_correct'] - 0.01
             and B2['consistency_all_agree'] >= B1['consistency_all_agree'] - 0.01)
decision = {'step': 'D', 'adopt': bool(cmp[best]['adopt'] and suites_ok), 'config': best,
            'extra_flags': EXTRA_FLAGS[best], 'dev': cmp[best], 'all_configs': cmp,
            'gauntlet': {'r8p': B1['gauntlet_correct'], 'r9p': B2['gauntlet_correct']},
            'consistency_all_agree': {'r8p': B1['consistency_all_agree'], 'r9p': B2['consistency_all_agree']},
            'suites_not_worse': suites_ok, 'r9p_sha256': sha(R9P),
            'rule': 'adopt = dev fair gain >= 2 points, wins > losses, McNemar p < 0.05, gauntlet and consistency not lower by more than 1 point'}
json.dump(decision, open(f'{PUB}/decision.json', 'w'), indent=1)
print(json.dumps(decision, indent=1))
print('D DONE: ADOPT r9p' if decision['adopt'] else 'D DONE: KEEP r8p')
''')]

E = [code(COMMON + '''
ALLOW_LOCKED = False   # the owner sets this to True, once, after reading public/decision.json
dec = json.load(open(one('public/decision.json', 'the output of kinthread-r9p-d-select')))
if not dec['adopt']:
    stop('decision.json says keep r8p: there is nothing to score (r8p was already scored once)')
if not ALLOW_LOCKED:
    stop('the locked test is scored once and only with the owner approval: set ALLOW_LOCKED = True')
R9P = one('runs/v10-r9p/best.pt', 'the output of kinthread-r9p-c-train')
if sha(R9P) != dec['r9p_sha256']:
    stop('the attached r9p checkpoint is not the one step D selected')
VER = one('public/r9p_verifier.json', 'the output of kinthread-r9p-d-select')
FLAGS = f"--adaptive 1.0 --hybrid --verifier {VER} --cover --split_search --para 2 --k 16 {dec['extra_flags']}"
'''), code('''
sh(f'python -m eval.run_real --ckpt {R9P} --split test {FLAGS} --out {PRIV}/r9p_test.json')
test = json.load(open(f'{PRIV}/r9p_test.json'))['score']['splits']
summary = {'step': 'E', 'r9p_test': test, 'r8p_test': {'test fair': '474/689 = 68.8%', 'test verified': '462/640 = 72.2%'},
           'setting': dec['config'], 'extra_flags': dec['extra_flags'], 'r9p_sha256': dec['r9p_sha256']}
json.dump(summary, open(f'{PUB}/E_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('E DONE')
''')]

if __name__ == '__main__':
    out = [
        write('A', '# r9p step A: r8p baseline on real dev, flag replay, exam-pattern mining\nNo training. About 2-4 h.', A, []),
        write('B1', '# r9p step B1: r8p on the gauntlet, consistency and unseen suites, and the 192-token dev run\nAbout 3-5 h.', suites('r8p'), []),
        write('C', '# r9p step C: train r9p from r8p\nAbout 3-6 h.', C, []),
        write('B2', '# r9p step B2: r9p on the gauntlet, consistency and unseen suites\nAbout 2-4 h.', suites('r9p'), [f"{USER}/{SLUG['C']}"]),
        write('D', '# r9p step D: verifier for r9p, dev comparison with r8p, adopt or keep\nAbout 3-5 h.', D,
              [f"{USER}/{SLUG[k]}" for k in ('A', 'B1', 'C', 'B2')]),
        write('E', '# r9p step E: locked test, once, only with the owner approval\nAbout 1-2 h.', E,
              [f"{USER}/{SLUG[k]}" for k in ('C', 'D')]),
    ]
    print('\n'.join(out))
