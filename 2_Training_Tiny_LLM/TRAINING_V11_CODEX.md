# r9p training run-book (for Codex)

This file is the complete task. Everything is already written: six Kaggle notebooks, a runner script and the
decision rules. Your job is to run them in order, check each result, and write the report. Do not design, tune or
improve anything.

---

## 1. Prompt to paste into Codex

Copy everything inside the box into Codex as the task.

```text
You are running a fixed, already-written machine-learning pipeline. Do not design or change anything.

Repository: https://github.com/ArAnirudh2901/Kinthread-llm-model
Branch: main (work and push only on this branch)
Instructions: the file TRAINING_V11_CODEX.md in that branch. Follow sections 3 to 8 exactly.

How to work:
1. Do the setup in section 4 once.
2. Run the steps in this order and one at a time: A, B1, C, B2, D. Run E only if the owner writes "approve locked".
3. Each step is one command: colab/r9p/run_step.sh <STEP>. Wait for it to finish (it can take hours).
4. After each step, do the check listed for that step in section 5. Copy the numbers into the report (section 7).
5. Commit only the folder results/r9p/<STEP>/ and the report, with the commit message given in section 5.
6. If any output contains "STOP:" or a check fails: stop the whole run. Do not retry with changes.
   Report the step, the exact STOP line, and what you checked (section 6 says what each message means).

Never:
- edit any file except results/r9p/**, HANDOFF.md and AGENTS.md (two exceptions, both with the exact commands
  given: section 4 step 4, and the ALLOW_LOCKED line in step E);
- change numbers, flags or commands in the notebooks or the runner;
- commit anything from ~/kaggle_r9p, any .log file, any file named private/*, or the Kaggle token;
- run step E without the words "approve locked" from the owner;
- add "Co-Authored-By" or any AI attribution lines to commits.

When finished (or stopped), reply with the filled report from section 7 and nothing else.
```

---

## 2. What this run does, in one paragraph

The current model, v10-r8p, scores 73.9% on real dev exam questions and 68.8% on the locked real test. This run
measures r8p in detail (step A, B1), trains the next model r9p from r8p with new balanced training data (step C),
measures r9p the same way (step B2), and decides with fixed rules whether r9p is better (step D). Only if it is, and
only with the owner's approval, r9p is scored once on the locked test (step E). Nothing is deployed.

## 3. Rules that are never broken

1. Real exam questions are for evaluation only. They are never training data and never committed to git.
2. The locked test (step E) is scored at most once, and only after the owner writes "approve locked".
3. Never quantize, distil or shrink the model. Never change the architecture, tokenizer or vocabulary.
4. Commits are authored as `Anirudh Aravalli <anirudh30012005@gmail.com>`, with no AI attribution lines.
5. Push only to `main`. Do not open pull requests. Do not deploy anything.
6. Do not print, log or commit the Kaggle token (`~/.kaggle/kaggle.json`).

## 4. One-time setup

Run these in the repository root, in order.

1. Get the branch:
   ```bash
   git clone -b main https://github.com/ArAnirudh2901/Kinthread-llm-model.git
   cd Kinthread-llm-model
   git config user.name "Anirudh Aravalli"
   git config user.email "anirudh30012005@gmail.com"
   ```
2. Install the Kaggle tool. The owner gives you the token file; put it at `~/.kaggle/kaggle.json`:
   ```bash
   pip install kaggle
   chmod 600 ~/.kaggle/kaggle.json
   kaggle kernels list --mine --page-size 5
   ```
   The last command must list the owner's notebooks (account `anirudharavalli`). If it fails, STOP: the token is
   missing or wrong.
3. Check that the private real-questions dataset exists:
   ```bash
   kaggle datasets list --mine | grep -i real
   ```
   Expected: a dataset named `anirudharavalli/kinthread-real-questions`.
4. Only if step 3 shows a different name: open `colab/r9p/build_notebooks.py`, change the one line
   `REAL = f'{USER}/kinthread-real-questions'` to the name shown, then run `python3 colab/r9p/build_notebooks.py`.
   This is the only code edit you may make. If no real-questions dataset exists at all, STOP and ask the owner.
5. Check the data files:
   ```bash
   cd data && sha256sum -c SHA256SUMS --quiet && cd ..
   ```
   Expected: no output. Any line means a corrupted file: STOP.

## 5. The steps

Each step runs on Kaggle with a GPU. Kaggle allows about 30 GPU hours per week and 12 hours per run. The runner pushes
the notebook, checks Kaggle every 10 minutes, downloads the outputs to `~/kaggle_r9p/<STEP>` (outside the repo) and
copies only the public results to `results/r9p/<STEP>/`. It prints `step <STEP> passed` at the end, or a `STOP:` line.

| Order | Step | What it does | Time | Needs |
| --- | --- | --- | --- | --- |
| 1 | A | r8p on real dev with tracing; replays answer flags; mines exam patterns | 2-4 h | nothing |
| 2 | B1 | r8p on the gauntlet, consistency and unseen sets; dev run with a 192-token reading | 3-5 h | nothing |
| 3 | C | trains r9p from r8p (15,000 steps) | 3-6 h | nothing |
| 4 | B2 | r9p on the gauntlet, consistency and unseen sets | 2-4 h | C passed |
| 5 | D | trains r9p's verifier; compares r9p with r8p on dev; writes `decision.json` | 3-5 h | A, B1, C, B2 passed |
| 6 | E | locked test, once (owner approval only) | 1-2 h | D says adopt |

### Step A

```bash
colab/r9p/run_step.sh A
```

Check: open `results/r9p/A/A_summary.json`.
- `r8p_dev` must contain `"dev fair"`. The first number must be between 522 and 534 (FINAL.json says 528/714). The
  notebook already stops if it is not; just copy the value.
- Copy `flags_adopted` (it may be empty: that is a valid result).
- Copy the first 5 lines of `top_gaps` and of `top_misses`.

Commit:
```bash
git add results/r9p/A && git commit -m "r9p step A: r8p dev baseline, flag replay, exam patterns" && git push origin main
```

### Step B1

```bash
colab/r9p/run_step.sh B1
```

Check: open `results/r9p/B1/B_r8p_summary.json` and `results/r9p/B1/r8p_dev_final_192_splits.json`. Copy
`gauntlet_correct`, `consistency_all_agree`, and the `"dev fair"` value of the 192-token run.

Commit: `git add results/r9p/B1 && git commit -m "r9p step B1: r8p suites" && git push origin main`

### Step C (training)

```bash
colab/r9p/run_step.sh C
```

Check: open `results/r9p/C/C_summary.json`. Copy `r9p_sha256` and `r9p_dev_quick_adaptive_hybrid` (`"dev fair"`).
This quick score uses a lighter setting than FINAL, so it is lower than 73.9% and is not the decision.

Commit: `git add results/r9p/C && git commit -m "r9p step C: training" && git push origin main`

### Step B2

```bash
colab/r9p/run_step.sh B2
```

Check: open `results/r9p/B2/B_r9p_summary.json`. Copy `gauntlet_correct` and `consistency_all_agree`.

Commit: `git add results/r9p/B2 && git commit -m "r9p step B2: r9p suites" && git push origin main`

### Step D (decision)

```bash
colab/r9p/run_step.sh D
```

Check: open `results/r9p/D/decision.json`. Copy `adopt`, `config`, `extra_flags`, `dev.old_fair`, `dev.new_fair`,
`dev.gain_points`, `dev.wins`, `dev.losses`, `dev.p`, `gauntlet`, `consistency_all_agree`. The rule is fixed:

> adopt r9p only if its dev fair score is at least 2 points higher than r8p's, it wins more dev questions than it
> loses, McNemar p < 0.05, and its gauntlet and consistency scores are not lower by more than 1 point.

Commit: `git add results/r9p/D && git commit -m "r9p step D: decision" && git push origin main`

- If `adopt` is `false`: the run is finished. Write the report. Do not run E.
- If `adopt` is `true`: write the report and ask the owner: "Step D adopts r9p. Approve the one-time locked test
  (step E)?" Wait.

### Step E (only after the owner writes "approve locked")

1. Allow the one locked run, run it, and lock it again (three commands, exactly):
   ```bash
   sed -i 's/ALLOW_LOCKED = False/ALLOW_LOCKED = True/' colab/r9p/E_locked/notebook.ipynb
   colab/r9p/run_step.sh E
   sed -i 's/ALLOW_LOCKED = True/ALLOW_LOCKED = False/' colab/r9p/E_locked/notebook.ipynb
   ```
2. Check `results/r9p/E/E_summary.json`: copy `r9p_test` (`"test fair"` and `"test verified"`).
3. Check that `git diff colab/r9p/E_locked/notebook.ipynb` prints nothing (the lock is back on).
4. Commit: `git add results/r9p/E && git commit -m "r9p step E: locked test, scored once" && git push origin main`

## 6. If something stops

| Message | Meaning | What you do |
| --- | --- | --- |
| `STOP: attach the private dataset kinthread-real-questions` | wrong dataset name | redo section 4 steps 3-4 |
| `STOP: no best.pt with SHA-256 38c289...` | r8p checkpoint not found | stop; tell the owner the r8p notebook output (`notebook99ae0586bd`) is not reachable |
| `STOP: attach the final-selection notebook output` | FINAL.json / verifier.json missing | stop; tell the owner `notebookf8798397a0` output is not reachable |
| `STOP: r8p dev fair is N/714 but FINAL.json says 528/714` | setup differs from the final selection | stop; report N |
| `STOP: training files missing` | Google Drive download failed | run step C once more; if it stops again, report it |
| `STOP: attach the output of kinthread-r9p-...` | an earlier step did not pass | run that earlier step first |
| `STOP: command failed: ...pytest...` | a test failed on Kaggle | stop; report the failing test name from the log in `~/kaggle_r9p/<STEP>` |
| `CUDA out of memory` in the step C log | GPU too small for block 512 | stop; report it (the owner decides about `--accum 2`) |
| `non-finite loss` in the step C log | training diverged | stop; report it (the owner decides about lr 2e-4) |
| Kaggle quota or "no GPU available" | weekly GPU hours used up | wait for the weekly reset, then run the same step again |
| any other `STOP:` line | unexpected | stop; report the step and the exact line |

Never fix a problem by editing code, tests or notebooks. Report it.

## 7. Report (fill in, append to HANDOFF.md, and reply with it)

```markdown
## r9p run (YYYY-MM-DD)

| Measure | r8p | r9p |
| --- | --- | --- |
| Real dev fair, FINAL setting | <A: r8p_dev "dev fair"> | <D: dev.new_fair> |
| Gain (points), wins / losses, McNemar p | | <D: gain_points>, <wins> / <losses>, p=<p> |
| Dev fair with 192-token reading | <B1: "dev fair"> | (in D config final_plus) |
| Gauntlet correct | <B1: gauntlet_correct> | <B2: gauntlet_correct> |
| Consistency, all variants agree | <B1: consistency_all_agree> | <B2: consistency_all_agree> |
| Locked test fair / verified | 68.8% / 72.2% (scored 2026-10-26) | <E: test fair> / <test verified>, or "not run" |

- Flags adopted in step A: <flags_adopted or "none">
- Step D decision: <adopt true/false>, config <config>, extra flags <extra_flags>
- r9p checkpoint SHA-256: <C: r9p_sha256>
- Largest exam-pattern gaps (A): <5 lines>
- Most missed patterns (A): <5 lines>
- Problems: <none, or the STOP line and step>
```

Also add one line to the handoff log at the end of `AGENTS.md`:
`- YYYY-MM-DD (r9p run): step D <adopted r9p / kept r8p>; dev fair <r8p> -> <r9p>; locked <result or not run>. Results in results/r9p/.`

Commit: `git add HANDOFF.md AGENTS.md && git commit -m "r9p run report" && git push origin main`

## 8. Not part of this run (the owner decides later)

- Changing the generator from the pattern-mining results (step A writes them; Claude uses them in a later session).
- The story supplement (stories of 240-400 words with at most 32 people, and two-family stories).
- A larger model (v11), exporting r9p to the website, or any deploy.

---

## Background for humans (Codex does not need this section)

- **r8p:** v10-r8p, step 6,000, SHA-256 `38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52`, 39.5M
  exact fp32 parameters. Real dev 73.9% fair / 77.2% verified; locked test 68.8% / 72.2%.
- **Weak spots measured on 2026-10-28:** gauntlet (browser pipeline, 667 questions) 92.1% overall but claims 63%,
  family puzzles 70%, two detours 77%, data sufficiency 77%; renamed twins agree 100%, two-detour versions only 73%.
  Generated sets score 92-99%, real exams 69%: the main gap is real-exam wording.
- **New r9p training data:** `data/balanced_mix_v1` (22,500 rows, 9 formats x 2,500) x2 and `data/detour_v1` (4,000)
  on top of r8p's data; `--block 512` (keeps 146 mix_r8 rows of 449-512 tokens that r8p skipped). Rows with more than
  32 people (2.2% of mix_r8, mostly joined long questions) are skipped: the model has 32 person tokens.
- **Answer-stage flags** (`--closure`, `--core_repair`, `--lever`, `--trace_pow`) are replayed from the step A trace
  without the model (`eval.rescore_trace`), adopted only with McNemar p < 0.05 on dev.
- **Pipeline code:** `colab/r9p/build_notebooks.py` writes the six notebooks and their `kernel-metadata.json`;
  `colab/r9p/run_step.sh` runs one step; `eval/compare_runs.py` applies the adoption rule; public outputs hold counts
  only, private outputs (real question text) never leave Kaggle and `~/kaggle_r9p`.
- **Hypotheses after r9p, in order** (each kept only if dev says so): consistency-filtered self-training on generated
  questions (arXiv 2608.30258); minimal-pair rows (`flip_pair`, arXiv 2605.04330); generator changes from the
  pattern-mining gaps; `--para 4`; v11 (a larger reader) only if all of these plateau below 85% on dev.
- **Open data worth a look:** KinshipQA (github.com/TiandaSun/Kinship-Data-Benchmark-for-Multi-hop-Reasoning),
  LogiQAte (huggingface.co/datasets/abhilekhborah/LogiQAte), CLUTRR (github.com/facebookresearch/clutrr).
