"""Name -> gender table for the name model, from two CC0 sources (Harvard Dataverse):
  naampy  in_rolls_state_year_fn_naampy_x100.csv.gz   naampy v2, Indian electoral rolls, 31 states: female/male
          counts per state, birth year and first name; label noise corrected per state and direction (`noise_rates`)
  SSA     ssa/yob*.txt (names.zip)               US Social Security baby names 1880-: female/male counts (public domain)
  WGND    wgnd_2_0_name-gender-code.tab          WIPO World Gender Name Dictionary 2.0: gender share per name and country
                                                 (shares only; a light gap filler for names the counted sources lack)

    python build_names.py [--src data/names] [--out data/names/name_gender.tsv.gz]

Output rows: name, p_female, support, sources, split. p_female mixes the sources by evidence: counts (naampy, SSA),
each capped at `CAP` people so no source swamps another, and WGND countries worth up to `WGND_COUNT` people; support
is that evidence. Splits by name hash (train 90 / val 5 / test 5).
Common English and question words are dropped unless the evidence is strong (WGND has entries like "baby", "true").
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import re
import unicodedata
import zlib

WGND_COUNT = 60       # people the largest WGND country counts for (WGND has shares, not counts)
CAP = 5000            # evidence one counted source can give a name
MIN_SUPPORT = 3       # evidence below this is dropped
STRONG = 200          # evidence that keeps a name even when it is also a common word
NAME = re.compile(r"[a-z]+(?:[-'][a-z]+)?")


def norm(raw: str) -> str | None:
    """Lowercase ASCII first name, or None (several words, digits, symbols, one letter)."""
    s = unicodedata.normalize('NFKD', raw.strip().strip('"').strip("'")).encode('ascii', 'ignore').decode().lower()
    s = s.strip(" .-'")
    return s if len(s) >= 2 and NAME.fullmatch(s) else None


def load_ssa(folder):
    """{name: [female, male]} summed over every year file (lines: Name,F|M,count)."""
    import glob
    c = collections.defaultdict(lambda: [0, 0])
    for path in glob.glob(f'{folder}/yob*.txt'):
        for line in open(path, encoding='ascii', errors='ignore'):
            parts = line.strip().split(',')
            if len(parts) == 3 and parts[1] in ('F', 'M'):
                n = norm(parts[0])
                if n:
                    c[n][0 if parts[1] == 'F' else 1] += int(parts[2])
    return c


def calib_names():
    """Half of the curated names (by hash) measure label noise; the other half stays for evaluation."""
    from kinlogic.names import NAME_GENDER
    return {n: g for n, g in NAME_GENDER.items() if zlib.crc32(n.encode()) % 2 == 0}


def load_naampy(path):
    """{name: [female, male]} summed over states and years after correcting each state's label noise."""
    by = collections.defaultdict(lambda: [0, 0])   # (state, name) -> counts
    with gzip.open(path, 'rt', newline='') as f:
        for r in csv.DictReader(f):
            n = norm(r['first_name'])
            if n:
                c = by[(r['state'], n)]
                c[0] += int(float(r['n_female'] or 0))
                c[1] += int(float(r['n_male'] or 0))
    rates = noise_rates(by, calib_names())
    out = collections.defaultdict(lambda: [0.0, 0.0])
    for (st, n), (fo, mo) in by.items():
        e_fm, e_mf = rates.get(st, (0.0, 0.0))
        det = 1 - e_fm - e_mf   # observed = A @ true, A = [[1-e_fm, e_mf], [e_fm, 1-e_mf]]
        F = max(0.0, ((1 - e_mf) * fo - e_mf * mo) / det)
        M = max(0.0, ((1 - e_fm) * mo - e_fm * fo) / det)
        w = det * det   # noisy states keep their names but count for less (Gujarat 0.16, UP 0.92)
        out[n][0] += w * F
        out[n][1] += w * M
    return out, rates


def noise_rates(by, calib):
    """{state: (share of women's names recorded male, share of men's names recorded female)} on calibration names."""
    acc = collections.defaultdict(lambda: [0, 0, 0, 0])   # f-names: male recs, all; m-names: female recs, all
    for (st, n), (f, m) in by.items():
        g = calib.get(n)
        a = acc[st]
        if g == 'F':
            a[0] += m
            a[1] += f + m
        elif g == 'M':
            a[2] += f
            a[3] += f + m
    return {st: (min(0.3, a[0] / a[1]) if a[1] >= 200 else 0.05, min(0.3, a[2] / a[3]) if a[3] >= 200 else 0.05)
            for st, a in acc.items()}


def load_wgnd(path):
    """{name: {country: p_female}} from the shares (rows per name, country and gender)."""
    by = collections.defaultdict(dict)
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd)
        for row in rd:
            if len(row) < 4:
                continue
            n = norm(row[0])
            if not n or row[2].strip('"') not in ('F', 'M'):
                continue
            try:
                w = float(row[3])
            except ValueError:
                continue
            d = by[n].setdefault(row[1].strip('"'), [0.0, 0.0])
            d[0 if row[2].strip('"') == 'F' else 1] += w
    return {n: {k: f / (f + m) for k, (f, m) in cs.items() if f + m > 0} for n, cs in by.items()}


DEFAULT_REL = 0.25   # weight of a WGND country with too few calibration names to measure


def country_reliability(wg, calib, min_n=25):
    """{country: weight}: (2 * accuracy - 1)^2 on the calibration names (clear labels only) times the square root of
    the country's share of names: WGND has shares, not counts, and many small codes disagree with the large sources
    (e.g. Oliver female in ~25 small codes, male in US/GB/AU/CA)."""
    acc = collections.defaultdict(lambda: [0, 0])
    for n, g in calib.items():
        for c, p in wg.get(n, {}).items():
            if p >= 0.9 or p <= 0.1:
                acc[c][0] += ('F' if p >= 0.9 else 'M') == g
                acc[c][1] += 1
    size = collections.Counter(c for cs in wg.values() for c in cs)
    top = max(size.values())
    rel = {c: (max(0.0, 2 * r / t - 1) ** 2 if t >= min_n else DEFAULT_REL) for c, (r, t) in acc.items()}
    return {c: rel.get(c, DEFAULT_REL) * (n / top) ** 0.5 for c, n in size.items()}   # bigger sources count for more


def common_words():
    """Question vocabulary and frequent English words that are not names in this task."""
    words = set()
    lex = os.path.join(os.path.dirname(__file__), 'kinlogic', 'data', 'lexicon.txt')
    if os.path.exists(lex):
        words |= {w.strip().lower() for w in open(lex) if w.strip().isalpha()}
    return words


def name_table_label(p, support, min_support=100, clear=0.9):
    """'M'/'F' for well-evidenced names with a clear majority, 'U' for well-evidenced names used by both (0.25-0.75),
    else None (the model decides). On held-out curated names: 397/437 decided, 0 wrong."""
    if support < min_support:
        return None
    return 'F' if p >= clear else 'M' if p <= 1 - clear else 'U' if 0.25 <= p <= 0.75 else None


def split_of(name):
    h = zlib.crc32(name.encode()) % 100
    return 'test' if h < 5 else 'val' if h < 10 else 'train'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/names')
    ap.add_argument('--out', default='data/names/name_gender.tsv.gz')
    a = ap.parse_args()
    na, rates = load_naampy(f'{a.src}/in_rolls_state_year_fn_naampy_x100.csv.gz')
    ss = load_ssa(f'{a.src}/ssa')
    wg = load_wgnd(f'{a.src}/wgnd_2_0_name-gender-code.tab')
    words = common_words()
    crel = country_reliability(wg, calib_names())
    rows, dropped = [], collections.Counter()
    for n in sorted(set(na) | set(ss) | set(wg)):
        counted = []   # (p_female, capped evidence) per counted source
        for src in (na, ss):
            f, m = src.get(n, (0, 0))
            if f + m > 0:
                counted.append((f / (f + m), min(CAP, f + m)))
        cs = {c: p for c, p in wg.get(n, {}).items() if crel.get(c, DEFAULT_REL) > 0}
        wsum = sum(crel.get(c, DEFAULT_REL) for c in cs)
        ev_ct, ev_wg = sum(e for _, e in counted), WGND_COUNT * wsum
        support = ev_ct + ev_wg
        if support < MIN_SUPPORT:
            dropped['little evidence'] += 1
            continue
        if n in words and support < STRONG:
            dropped['common word'] += 1
            continue
        p_wg = sum(p * crel.get(c, DEFAULT_REL) for c, p in cs.items()) / wsum if wsum else 0.0
        p = (sum(pf * e for pf, e in counted) + p_wg * ev_wg) / support
        src = '+'.join(s for s, ok in (('naampy', n in na), ('ssa', n in ss), ('wgnd', ev_wg > 0)) if ok)
        rows.append((n, round(p, 4), round(support), src, split_of(n)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with gzip.open(a.out, 'wt', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(('name', 'p_female', 'support', 'sources', 'split'))
        w.writerows(rows)
    table = os.path.join(os.path.dirname(a.out), 'name_table.tsv.gz')   # exact lookup shipped with the name model
    with gzip.open(table, 'wt') as f:
        for n, p, sup, _, _ in rows:
            g = name_table_label(p, sup)
            if g:
                f.write(f'{n}\t{g}\n')
    sha = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
    sure = sum(1 for r in rows if r[1] <= 0.05 or r[1] >= 0.95)
    man = {'names': len(rows), 'by_split': dict(collections.Counter(r[4] for r in rows)),
           'by_source': dict(collections.Counter(r[3] for r in rows)), 'clear_gender': sure, 'dropped': dict(dropped),
           'naampy_names': len(na), 'ssa_names': len(ss), 'wgnd_names': len(wg), 'cap': CAP, 'wgnd_count': WGND_COUNT, 'min_support': MIN_SUPPORT,
           'noise_rates': {k: [round(x, 4) for x in v] for k, v in sorted(rates.items())},
           'country_weights': {k: round(v, 3) for k, v in sorted(crel.items())}, 'sha256': sha, 'sources': {'naampy': 'doi:10.7910/DVN/WZGJBM (CC0 1.0)', 'ssa': 'ssa.gov/oact/babynames/names.zip (public domain)',
                       'wgnd': 'doi:10.7910/DVN/MSEGSJ (CC0 1.0)'}}
    json.dump(man, open(os.path.join(os.path.dirname(a.out), 'name_gender.manifest.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != 'sources'}, indent=1))


if __name__ == '__main__':
    main()
