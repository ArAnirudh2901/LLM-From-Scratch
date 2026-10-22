"""Gender detection test (written 2026-10-21, never trained on). Every answer flips with someone's gender; the kind
says where that gender comes from: pronoun, constraint (wife/husband, only son, no sisters), name (engine name
priors), unisex (must stay open), mixed.

    python -m eval.gender_set --ckpt runs/v10-r1-1.36M/final.pt
Reports answer accuracy by cue, per-person gender read by the model (facts only, no name priors) against gold, and
name-prior coverage on common names.
"""
from eval.authored_set import q

G = [
    # ---- pronouns ----
    q('g01', 'pronoun', 2, 'Arya and Neel are siblings. Arya told her teacher that Neel is the son of Ved. How is Arya related to Ved? (a) Son (b) Daughter (c) Niece (d) Wife',
      'Arya female ; Arya sibling Neel ; Ved son Neel ; ask Ved Arya', 'b'),
    q('g02', 'pronoun', 2, 'Sam is a sibling of Alex. He is three years older than Alex. How is Sam related to Alex? (a) Brother (b) Sister (c) Cousin (d) Father',
      'Sam male ; Alex sibling Sam ; ask Alex Sam', 'a'),
    q('g03', 'pronoun', 2, "Taylor is Robin's only child. Robin walked her down the aisle at the wedding. How is Taylor related to Robin? (a) Son (b) Daughter (c) Niece (d) Daughter-in-law",
      'Robin only child Taylor ; Taylor female ; ask Robin Taylor', 'b'),
    q('g11', 'pronoun', 2, 'Alex has a sibling, Jamie. When Jamie got married, she wore her grandmother\'s ring. How is Jamie related to Alex? (a) Brother (b) Sister (c) Cousin (d) Wife',
      'Alex sibling Jamie ; Jamie female ; ask Alex Jamie', 'b'),
    q('g12', 'pronoun', 2, "Morgan's father is Dev. Morgan and his wife live next door to Dev. How is Morgan related to Dev? (a) Son (b) Daughter (c) Son-in-law (d) Nephew",
      'Morgan father Dev ; Morgan male ; Morgan wife #1 ; ask Dev Morgan', 'a'),
    q('g13', 'pronoun', 2, "Quinn is Ravi's grandchild. Quinn often says that her favourite person is Ravi. How is Quinn related to Ravi? (a) Grandson (b) Granddaughter (c) Niece (d) Daughter",
      'Ravi grandchild Quinn ; Quinn female ; ask Ravi Quinn', 'b'),
    # ---- constraints ----
    q('g04', 'constraint', 2, "Chris lives with a sibling, Jordan. Jordan's wife cooks for both of them. How is Jordan related to Chris? (a) Brother (b) Sister (c) Cousin (d) Brother-in-law",
      'Chris sibling Jordan ; Jordan wife #1 ; ask Chris Jordan', 'a'),
    q('g05', 'constraint', 2, "Devi is Kiran's mother. Kiran's husband loves Devi's recipes. How is Kiran related to Devi? (a) Son (b) Daughter (c) Daughter-in-law (d) Niece",
      'Kiran mother Devi ; Kiran husband #1 ; ask Devi Kiran', 'b'),
    q('g14', 'constraint', 2, "Lee is Pat's parent. Lee's husband is Sam. How is Lee related to Pat? (a) Father (b) Mother (c) Uncle (d) Aunt",
      'Pat parent Lee ; Lee husband Sam ; ask Pat Lee', 'b'),
    q('g15', 'constraint', 3, 'Avi is the only son of Rama. Charlie is also a child of Rama. How is Charlie related to Avi? (a) Brother (b) Sister (c) Cousin (d) Niece',
      'Rama only son Avi ; Rama child Charlie ; ask Avi Charlie', 'b'),
    q('g16', 'constraint', 2, 'Kim has no sisters. Robin is a sibling of Kim. How is Robin related to Kim? (a) Brother (b) Sister (c) Cousin (d) Father',
      'Kim no-sister ; Kim sibling Robin ; ask Kim Robin', 'a'),
    # ---- names only ----
    q('g06', 'name', 2, 'Pointing to Ashwin, Neha said, "His mother is my mother\'s only daughter." How is Ashwin related to Neha? (a) Son (b) Nephew (c) Brother (d) Cousin',
      'Ashwin male ; Ashwin mother #1 ; Neha mother #2 ; #2 only daughter #1 ; ask Neha Ashwin', 'a'),
    q('g07', 'name', 1, 'Nikita and Rohit are siblings. How is Rohit related to Nikita? (a) Brother (b) Sister (c) Cousin (d) Uncle',
      'Nikita sibling Rohit ; ask Nikita Rohit', 'a'),
    q('g08', 'name', 1, "Ayesha's parent Salim has one other child, Zainab. How is Zainab related to Ayesha? (a) Brother (b) Sister (c) Cousin (d) Niece",
      'Ayesha parent Salim ; Salim child Zainab ; ask Ayesha Zainab', 'b'),
    q('g17', 'name', 1, 'Deepika is a child of Mahesh. How is Deepika related to Mahesh? (a) Son (b) Daughter (c) Niece (d) Wife',
      'Mahesh child Deepika ; ask Mahesh Deepika', 'b'),
    q('g18', 'name', 1, 'Karthikeyan and Lakshmi are the children of Murugan. How is Karthikeyan related to Lakshmi? (a) Brother (b) Sister (c) Cousin (d) Husband',
      'Murugan child Karthikeyan ; Murugan child Lakshmi ; ask Lakshmi Karthikeyan', 'a'),
    q('g19', 'name', 1, "George is Emily's parent. How is George related to Emily? (a) Father (b) Mother (c) Uncle (d) Grandfather",
      'Emily parent George ; ask Emily George', 'a'),
    q('g20', 'name', 2, "Yusuf's child Fatima is married to Imran. How is Imran related to Yusuf? (a) Son (b) Son-in-law (c) Daughter-in-law (d) Nephew",
      'Yusuf child Fatima ; Fatima spouse Imran ; ask Yusuf Imran', 'b'),
    q('g21', 'name', 2, "Vihaan's mother is the sister of Ananya's father. How is Ananya related to Vihaan's mother? (a) Nephew (b) Niece (c) Daughter (d) Sister",
      'Vihaan mother #1 ; Ananya father #2 ; #2 sister #1 ; ask #1 Ananya', 'b'),
    # ---- gender must stay open ----
    q('g22', 'unisex', 2, 'Shashi and Kiran are siblings. How is Kiran related to Shashi? (a) Brother (b) Sister (c) Cousin (d) Cannot be determined',
      'Shashi sibling Kiran ; ask Shashi Kiran', 'd'),
    q('g23', 'unisex', 2, 'Rohit and Priya are siblings. How is Priya related to Rohit? (a) Brother (b) Sister (c) Cousin (d) Cannot be determined',
      'Rohit sibling Priya ; ask Rohit Priya', 'd'),
    q('g24', 'unisex', 2, 'P is the child of Q. Q is the father of R. How is P related to R? (a) Brother (b) Sister (c) Cousin (d) Cannot be determined',
      'Q child P ; R father Q ; ask R P', 'd'),
    q('g25', 'unisex', 2, 'Jyoti is a daughter of Prakash. Harsha is the other child of Prakash. How is Harsha related to Jyoti? (a) Brother (b) Sister (c) Cousin (d) Cannot be determined',
      'Prakash daughter Jyoti ; Prakash child Harsha ; ask Jyoti Harsha', 'd'),
    # ---- pronoun and name together ----
    q('g26', 'mixed', 1, "Ashok's child Chandni is a pilot; she flies to Dubai every week. How is Chandni related to Ashok? (a) Son (b) Daughter (c) Niece (d) Wife",
      'Ashok child Chandni ; Chandni female ; ask Ashok Chandni', 'b'),
]

NAMES = {   # common names with a clear gender, for the name-prior coverage check
    'm': 'rahul rohit amit vikram arjun karthik suresh ramesh mahesh ganesh imran yusuf salim farhan harpreet_no gurdeep_no '
         'manpreet_no rajesh anil sunil vijay ajay sanjay deepak ashok prakash venkatesh murugan karthikeyan srinivas '
         'abdul mohammed george james david michael john vihaan aarav ishaan kabir dhruv',
    'f': 'priya neha nikita ayesha zainab fatima deepika lakshmi meena sunita anita kavita pooja shalini divya swati '
         'ananya diya isha saanvi radha sita gita emily sarah mary jennifer jessica chandni nalini lubna shabnam'}


def gender_of(lf, people):
    """{person: 'male'/'female'} for everyone whose gender every family that fits the facts agrees on (no priors)."""
    from kinlogic.engine import solve
    from kinlogic.facts import parse
    st, bad = parse(lf)
    r = solve(st, people=people, max_steps=20000) if not bad else None
    if not r or not r['ok'] or not r['worlds']:
        return None
    out = {}
    for p in people:
        if p in r['ids']:
            gs = {w.g(r['ids'][p]) for w in r['worlds']}
            if len(gs) == 1 and None not in gs:
                out[p] = gs.pop()
    return out


if __name__ == '__main__':
    import argparse
    import collections
    from eval.check_authored import check
    from eval.exam_set import gold_facts
    from kinlogic.names import name_priors
    from kinlogic.prepare import prepare
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    a = ap.parse_args()
    check(G)
    names = [(n, g) for g, s in (('male', NAMES['m']), ('female', NAMES['f'])) for n in s.split() if not n.endswith('_no')]
    pri = name_priors([{'id': f'@{i}', 'text': n.title(), 'kind': 'name'} for i, (n, _) in enumerate(names)])
    hit = sum(pri.get(f'@{i}') == g[0].upper() for i, (_, g) in enumerate(names))
    miss = [n for i, (n, g) in enumerate(names) if pri.get(f'@{i}') != g[0].upper()]
    print(f'name priors: {hit}/{len(names)} common names gendered correctly; missing or wrong: {", ".join(miss) or "none"}')
    if a.ckpt:
        from eval.evaluate import load
        from kinlogic.exam import answer
        from reader import generate
        from train import get_device
        dev = get_device()
        model, vocab = load(a.ckpt, dev)
        read = lambda b: generate(model, vocab, b, dev, k=8)
        by, wrong = collections.defaultdict(lambda: [0, 0]), []
        pg = collections.Counter()
        for it in G:
            r = answer(it['text'], read)
            ok = r.get('option') == it['expect']
            by[it['kind']][0] += ok
            by[it['kind']][1] += 1
            if not ok:
                wrong.append(f"{it['id']} ({it['kind']}) want {it['expect']} got {r.get('option')} ({r.get('answer')})")
            p = prepare(it['text'])
            people = sorted({t for t in p['tokens'] if t[0] == '@'})
            top = generate(model, vocab, [p['tokens']], dev, k=1)[0]
            top = top if isinstance(top, str) else (top[0] if isinstance(top[0], str) else top[0][0])
            gold, got = gender_of(gold_facts(it, p['ents']), people), gender_of(top, people) or {}
            for x in people:
                if gold and x in gold:
                    pg['stated'] += 1
                    pg['read right'] += got.get(x) == gold[x]
                elif x in got:
                    pg['invented'] += 1
        n = sum(v[0] for v in by.values())
        print(f'answers: {n}/{len(G)} | ' + ', '.join(f'{k} {a}/{b}' for k, (a, b) in sorted(by.items())))
        print(f"genders the text fixes (pronoun, relation word, constraint): model read {pg['read right']}/{pg['stated']} right; "
              f"genders it invented where the text leaves them open: {pg['invented']}")
        print('\n'.join(wrong))
