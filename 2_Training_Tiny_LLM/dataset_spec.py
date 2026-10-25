"""Dataset v8 specification: the one place that says what the training data must contain. To cover something new,
add an entry here (and its rewrite function in boost_phrases.py if it is a new surface form), then run
`python build_dataset_v8.py`; the manifest it writes shows every count against these targets.

Three balances, each measured against real exam questions (phrase_audit.py, answer_audit.py):
  1. answers   - which relation is asked for: real exam shares blended with a uniform floor over every relation the
                 engine can name, so rare answers (great-grand-niece, co-brother, cousin's husband ...) are practised;
  2. surfaces  - how the facts are written (constructions below): each gets a target share of the rows;
  3. kinds     - question kinds (statements, speech, puzzles, coded, claims, sufficiency, deep chains) keep the
                 generator's verified mix (build_dataset.py) through the base pool.
Every row keeps verified facts: rewrites either leave the facts unchanged or are re-checked by the engine.
"""

# answer balance: target share = BLEND * real share + (1 - BLEND) * uniform share over all answers seen; weight caps
ANSWER_BLEND = 0.6
ANSWER_MAX_WEIGHT = 12.0
ANSWER_MIN_WEIGHT = 0.15

# surface constructions: name -> (target share of rewritten rows, how it is produced, real-exam example)
CONSTRUCTIONS = {
    'exam copula (Y is the R of X)': (0.17, 'restyle: X\'s R is Y / X has a R named Y / the R of X is Y -> Y is the R of X', 'A is the father of B.'),
    'relative clause, comma (, who is)': (0.12, 'boost: merge two sentences on a shared person', 'A is the sister of B, who is the son of C.'),
    'relative clause, no comma (who is)': (0.05, 'boost: subject relative without commas', 'A who is the wife of B has a son C.'),
    'whose clause': (0.06, 'boost: X , whose R is Y , ...', 'P, whose husband is Q, is the mother of R.'),
    'coordination (and the S of)': (0.08, 'restyle: X is the R of Y and the S of Z', 'A is the son of B and the brother of C.'),
    'married to': (0.15, 'restyle: wife/husband -> married to (engine re-checks the answer)', 'A is married to B.'),
    'if-conditional': (0.12, 'restyle: if A , B and C , how is ...?', 'If A is the brother of B and B is the mother of C, how ...?'),
    'exam question forms': (0.10, 'restyle: how is X related to Y / who is X to Y / what is X to Y', 'How is A related to C?'),
    'plain (generator wording kept)': (0.11, 'no rewrite', ''),
}

# meaning types real exams use more than the generator (phrase_audit.py): extra rows drawn from the verified pool
RARE_MEANING_QUOTAS = {'paternal / maternal': 25000, 'of-chain (R of the S of)': 15000, 'negation (not / no)': 20000,
                       'count (has two sons)': 8000, "possessive chain x2 (X 's R 's S)": 13000, 'only daughter / son of my': 8000,
                       'grand-/great-': 8000, 'only (only son / only child)': 8000, 'profession': 5000, 'code symbols (means)': 12000}

ROWS = 300000          # rows in data/dataset_v8_extra.jsonl.gz
LABEL_SAMPLE = 160000  # base rows labelled by the engine to compute answer weights
