"""Fresh questions written 2026-10-21 after training, for a clean test: new names, wording and code symbols, never in
any training or dev data. Same format as eval/authored_set.py; every expected option is checked with gold facts.

    python -m eval.fresh_set --ckpt runs/v10-r1-1.36M/final.pt     (gold check first, then the model)
"""
from eval.authored_set import q

F = [
    # ---- statements ----
    q('f01', 'statements', 1, "Kiran's father Mohan has a sister, Leela. Leela's son is Tarun. How is Tarun related to Kiran? (a) Brother (b) Cousin (c) Nephew (d) Uncle",
      'Kiran father Mohan ; Mohan sister Leela ; Leela son Tarun ; ask Kiran Tarun', 'b'),
    q('f02', 'statements', 2, 'Neeraj is the husband of Pooja, and Ishaan is their son. Pooja\'s father lives with them. How is Pooja\'s father related to Ishaan? (a) Paternal grandfather (b) Maternal grandfather (c) Uncle (d) Father-in-law',
      'Neeraj wife Pooja ; Pooja son Ishaan ; Pooja father #1 ; ask Ishaan #1', 'b'),
    q('f03', 'statements', 2, "Ravi's wife Sunita has a brother named Ajay. Ajay's daughter is Meera. How is Ravi related to Meera? (a) Father (b) Uncle (c) Brother-in-law (d) Grandfather",
      'Ravi wife Sunita ; Sunita brother Ajay ; Ajay daughter Meera ; ask Meera Ravi', 'b'),
    q('f04', 'statements', 1, "Asha is the daughter of Vikram. Vikram's mother Kamala also has a daughter, Rekha, and Rekha has a son, Dev. How is Dev related to Asha? (a) Brother (b) Cousin (c) Nephew (d) Uncle",
      'Vikram daughter Asha ; Vikram mother Kamala ; Kamala daughter Rekha ; Rekha son Dev ; ask Asha Dev', 'b'),
    q('f05', 'statements', 2, "Farhan married Zoya. Zoya's sister Sana later married Imran. How is Imran related to Farhan? (a) Co-brother (b) Cousin (c) Father-in-law (d) Nephew",
      'Farhan wife Zoya ; Zoya sister Sana ; Sana husband Imran ; ask Farhan Imran', 'a'),
    q('f06', 'statements', 2, "Gita is the wife of the only son of Lata's mother's mother. How is Gita related to Lata? (a) Mother (b) Maternal aunt (c) Grandmother (d) Sister",
      'Lata mother #1 ; #1 mother #2 ; #2 only son #3 ; #3 wife Gita ; ask Lata Gita', 'b'),
    q('f07', 'statements', 1, "Nisha's son Arav is married to Kiara, and Tia is their daughter. How is Nisha related to Tia? (a) Mother (b) Grandmother (c) Aunt (d) Mother-in-law",
      'Nisha son Arav ; Arav wife Kiara ; Kiara daughter Tia ; ask Tia Nisha', 'b'),
    q('f08', 'statements', 2, "Om and his brother Jai grew up in Pune. Jai's wife is Priya, and Priya's only son is Kunal. How is Om related to Kunal? (a) Father (b) Paternal uncle (c) Maternal uncle (d) Cousin",
      'Om male ; Om brother Jai ; Jai wife Priya ; Priya only son Kunal ; ask Kunal Om', 'b'),
    q('f09', 'statements', 2, "Bhavna's husband's father is Suresh, and Suresh's wife is Usha. How is Usha related to Bhavna? (a) Mother (b) Mother-in-law (c) Aunt (d) Sister-in-law",
      'Bhavna husband #1 ; #1 father Suresh ; Suresh wife Usha ; ask Bhavna Usha', 'b'),
    q('f10', 'statements', 1, 'Raghav told me that Yamini is the daughter of his only sister. How is Raghav related to Yamini? (a) Paternal uncle (b) Maternal uncle (c) Father (d) Grandfather',
      'Raghav male ; Raghav only sister #1 ; #1 daughter Yamini ; ask Yamini Raghav', 'b'),
    # ---- pointing and speech ----
    q('f11', 'speech', 2, 'Holding up an old wedding photograph, Tanvi pointed to a man in it and said, "He is the only son of my mother\'s father." How is the man related to Tanvi? (a) Brother (b) Maternal uncle (c) Father (d) Cousin',
      '~man male ; Tanvi mother #1 ; #1 father #2 ; #2 only son ~man ; ask Tanvi ~man', 'b'),
    q('f12', 'speech', 3, 'Introducing a woman to his guests, Rohan said, "Her husband is the only grandson of my paternal grandfather." How is the woman related to Rohan? (a) Sister (b) Wife (c) Sister-in-law (d) Cousin',
      '~woman female ; Rohan male ; ~woman husband #1 ; Rohan paternal grandfather #2 ; #2 only grandson #1 ; ask Rohan ~woman', 'b'),
    q('f13', 'speech', 3, 'Pointing to a boy, Shalini said, "His father\'s only sister is my mother-in-law." How is the boy related to Shalini\'s husband? (a) Nephew (b) Cousin (c) Brother (d) Son',
      "~boy male ; ~boy father #1 ; #1 only sister #2 ; Shalini mother-in-law #2 ; Shalini husband #3 ; #2 son #3 ; ask #3 ~boy", 'b'),
    q('f14', 'speech', 3, 'Showing a photograph of a girl to his friends, Vinod said, "She is the daughter of the son of my father\'s only son." How is the girl related to Vinod? (a) Daughter (b) Niece (c) Granddaughter (d) Grand-niece',
      '~girl female ; Vinod male ; Vinod father #1 ; #1 only son #2 ; #2 son #3 ; #3 daughter ~girl ; ask Vinod ~girl', 'c'),
    q('f15', 'speech', 3, 'Introducing a girl to his colleagues, Abhay said, "Her father is the only son of my paternal grandfather\'s only son." How is the girl related to Abhay? (a) Niece (b) Daughter (c) Sister (d) Cousin',
      '~girl female ; Abhay male ; Abhay paternal grandfather #1 ; #1 only son #2 ; #2 only son #3 ; #3 daughter ~girl ; ask Abhay ~girl', 'b'),
    # ---- coded (new symbols, reversed and possessive definitions) ----
    q('f16', 'coded', 2, 'If P ⊕ Q means P is the wife of Q, P ⊗ Q means P is the son of Q and P ⊘ Q means P is the brother of Q, how is M related to K in K ⊗ L ⊕ N ⊘ M ⊗ R? (a) Father (b) Paternal uncle (c) Maternal uncle (d) Grandfather',
      'L son K ; N wife L ; M brother N ; R son M ; ask K M', 'b'),
    q('f17', 'coded', 2, '"A ▲ B" means "B is the mother of A"; "A ▼ B" means "B is the brother of A"; "A ■ B" means "B is the daughter of A". In the expression S ▲ T ▼ U ■ V, how is V related to S? (a) Sister (b) Niece (c) Cousin (d) Aunt',
      'S mother T ; T brother U ; U daughter V ; ask S V', 'c'),
    q('f18', 'coded', 2, 'If X & Y means X is married to Y, X $ Y means X is the father of Y and X # Y means X is the sister of Y, then in G $ H # J & K, where J is a man, how is K related to G? (a) Daughter (b) Daughter-in-law (c) Sister-in-law (d) Niece',
      'H father G ; J sister H ; J spouse K ; J male ; ask G K', 'b'),
    q('f19', 'coded', 3, "A + B means A is B's father, A - B means A is B's wife, A × B means A is B's daughter and A ÷ B means A is B's brother. How is P related to T in P × Q - R ÷ S + T? (a) Sister (b) Cousin (c) Niece (d) Aunt",
      'Q daughter P ; R wife Q ; S brother R ; T father S ; ask T P', 'b'),
    q('f20', 'coded', 3, 'If A @ B means A is the son of B, A % B means A is the husband of B and A ~ B means A is the brother of B, how is the mother of D related to A in A ~ B @ C % D? (a) Mother (b) Maternal grandmother (c) Paternal grandmother (d) Aunt',
      'B brother A ; C son B ; D husband C ; D mother #1 ; ask A #1', 'b'),
    q('f21', 'coded', 3, 'P ◇ Q means P is the father of Q, and P ○ Q means P is the mother of Q. How is A related to G in A ◇ B ○ C ◇ D ○ E ◇ F ○ G? (a) Great-great-great-grandfather (b) Great-great-great-great-grandfather (c) Great-great-grandfather (d) Great-great-great-great-great-grandfather',
      'B father A ; C mother B ; D father C ; E mother D ; F father E ; G mother F ; ask G A', 'b'),
    # ---- deep chains in plain prose ----
    q('f22', 'deep', 2, "Arjun's father Dhruv is Rina's son. Rina's mother is Saroj, whose husband Bimal is the son of Hari. How is Hari related to Arjun? (a) Great-grandfather (b) Great-great-grandfather (c) Great-great-great-grandfather (d) Grandfather",
      'Arjun father Dhruv ; Dhruv mother Rina ; Rina mother Saroj ; Saroj husband Bimal ; Bimal father Hari ; ask Arjun Hari', 'b'),
    q('f23', 'story', 2, "On Sunday, Leena baked a cake for her grandson Mihir. Mihir's father, Sanjay, is Leena's only son. Sanjay's wife, Payal, has one brother, Nitin. How is Nitin related to Mihir? (a) Paternal uncle (b) Maternal uncle (c) Father (d) Cousin",
      'Leena grandson Mihir ; Mihir father Sanjay ; Leena only son Sanjay ; Sanjay wife Payal ; Payal brother Nitin ; ask Mihir Nitin', 'b'),
    q('f24', 'deep', 3, "Kavi's father is Lokesh, and Lokesh's father is Mukund. Mukund's own father was Narayan. Mukund's brother Om had a son named Prem, and Prem's son is Qadir. How is Qadir related to Kavi? (a) Cousin (b) Second cousin (c) Third cousin (d) Second cousin once removed",
      'Kavi father Lokesh ; Lokesh father Mukund ; Mukund father Narayan ; Mukund brother Om ; Om son Prem ; Prem son Qadir ; ask Kavi Qadir', 'b'),
    q('f25', 'deep', 3, "From the Rathore family records: Bhim was the father of Chand; Chand's daughter was Devi; Devi's son was Eklavya; Eklavya's son was Fateh; Fateh's daughter was Gauri; Gauri's son was Harsh; Harsh's son was Indra; and Indra's daughter is Jyoti. How is Bhim related to Jyoti? (a) Great-great-great-great-great-grandfather (b) Great-great-great-great-great-great-grandfather (c) Great-great-great-great-great-great-great-grandfather (d) Great-great-great-great-grandfather",
      'Chand father Bhim ; Chand daughter Devi ; Devi son Eklavya ; Eklavya son Fateh ; Fateh daughter Gauri ; Gauri son Harsh ; Harsh son Indra ; Indra daughter Jyoti ; ask Jyoti Bhim', 'b'),
    q('f26', 'deep', 3, "Riya's mother's mother's mother is Sudha. Sudha's son is Tushar, and Tushar's daughter is Uttara. How is Uttara related to Riya? (a) Cousin (b) Mother\'s cousin (c) Second cousin (d) Aunt",
      'Riya mother #1 ; #1 mother #2 ; #2 mother Sudha ; Sudha son Tushar ; Tushar daughter Uttara ; ask Riya Uttara', 'b'),
    q('f27', 'deep', 3, 'Pointing to a girl in a photograph, Kunal told his wife, "She is the daughter of the only son of the only son of the only son of my father\'s father\'s father." How is the girl related to Kunal? (a) Niece (b) Daughter (c) Granddaughter (d) Cousin',
      '~girl female ; Kunal male ; Kunal father #1 ; #1 father #2 ; #2 father #3 ; #3 only son #4 ; #4 only son #5 ; #5 only son #6 ; #6 daughter ~girl ; ask Kunal ~girl', 'b'),
    q('f28', 'deep', 2, "Gargi lives with her son Hitesh. Hitesh's son is Ishan, Ishan's daughter is Juhi, Juhi's son is Kartik, and Kartik's daughter is Lavanya. How is Gargi related to Lavanya? (a) Great-great-grandmother (b) Great-great-great-grandmother (c) Great-great-great-great-grandmother (d) Great-grandmother",
      'Gargi female ; Gargi son Hitesh ; Hitesh son Ishan ; Ishan daughter Juhi ; Juhi son Kartik ; Kartik daughter Lavanya ; ask Lavanya Gargi', 'b'),
    q('f29', 'deep', 3, "Sohan and Mohan are brothers. Sohan's son is Ajit, Ajit's son is Bunty, and Bunty's son is Chintu. Mohan's daughter is Dimple, and Dimple's son is Eshan. How is Chintu related to Eshan? (a) Second cousin (b) Second cousin once removed (c) Third cousin (d) Cousin twice removed",
      'Sohan brother Mohan ; Mohan brother Sohan ; Sohan son Ajit ; Ajit son Bunty ; Bunty son Chintu ; Mohan daughter Dimple ; Dimple son Eshan ; ask Eshan Chintu', 'b'),
    # ---- closed-family puzzles ----
    q('f30', 'puzzle', 2, 'A family has six members P, Q, R, S, T and U in three generations, with two married couples. Q is the husband of P. R is the daughter of P. S is the husband of R. T and U are the children of S; T is a boy and U is a girl. How is Q related to U? (a) Paternal grandfather (b) Maternal grandfather (c) Father (d) Uncle',
      'members 6 ; couples 2 ; generations 3 ; P husband Q ; P daughter R ; R husband S ; S son T ; S daughter U ; ask U Q', 'b'),
    q('f31', 'puzzle', 2, 'A family has six members P, Q, R, S, T and U in three generations, with two married couples. Q is the husband of P. R is the daughter of P. S is the husband of R. T and U are the children of S; T is a boy and U is a girl. How many male members are there in the family? (a) 2 (b) 3 (c) 4 (d) Cannot be determined',
      'members 6 ; couples 2 ; generations 3 ; P husband Q ; P daughter R ; R husband S ; S son T ; S daughter U ; count male', 'b'),
    q('f32', 'puzzle', 1, "Five people sit for a family photo: Aakash, his wife Bindu, their son Chetan, Chetan's wife Dolly and their daughter Esha. Who is the mother-in-law of Dolly? (a) Bindu (b) Esha (c) Aakash (d) Chetan",
      'members 5 ; Aakash male ; Aakash wife Bindu ; Bindu son Chetan ; Chetan wife Dolly ; Chetan daughter Esha ; who Dolly mother-in-law', 'a'),
    q('f33', 'puzzle', 1, "Five people sit for a family photo: Aakash, his wife Bindu, their son Chetan, Chetan's wife Dolly and their daughter Esha. How many female members are in the photo? (a) 2 (b) 3 (c) 4 (d) Cannot be determined",
      'members 5 ; Aakash male ; Aakash wife Bindu ; Bindu son Chetan ; Chetan wife Dolly ; Chetan daughter Esha ; count female', 'b'),
    # ---- gender from names and constraints ----
    q('f34', 'statements', 2, 'Nalini said, "Vikas is the only son of my mother-in-law\'s husband." How is Vikas related to Nalini? (a) Brother-in-law (b) Husband (c) Father-in-law (d) Son',
      'Nalini mother-in-law #1 ; #1 husband #2 ; #2 only son Vikas ; ask Nalini Vikas', 'b'),
    q('f35', 'statements', 2, "Tejas has two sisters, Riddhi and Siddhi. Riddhi's son is Om. Siddhi is married to Kunal. How is Kunal related to Om? (a) Paternal uncle (b) Maternal uncle (c) Father (d) Cousin",
      'Tejas sister Riddhi ; Tejas sister Siddhi ; Riddhi son Om ; Siddhi husband Kunal ; ask Om Kunal', 'b'),
    q('f36', 'statements', 3, "Mira is the only daughter of Kishore. Kishore's wife Latha has three children, the youngest of whom is Naveen. How is Naveen related to Mira? (a) Brother (b) Cousin (c) Sister (d) Uncle",
      'Kishore only daughter Mira ; Kishore wife Latha ; Latha child Naveen ; ask Mira Naveen', 'a'),
    q('f37', 'speech', 1, 'Dhruv said to Esha, "My father\'s sister is your mother." How is Esha related to Dhruv? (a) Sister (b) Cousin (c) Niece (d) Aunt',
      'Dhruv father #1 ; #1 sister #2 ; Esha mother #2 ; ask Dhruv Esha', 'b'),
    q('f38', 'statements', 1, "Ananya's father has only one sister, Bhumi. Bhumi's daughter is Charu. How is Charu related to Ananya? (a) Sister (b) Cousin (c) Niece (d) Aunt",
      'Ananya father #1 ; #1 only sister Bhumi ; Bhumi daughter Charu ; ask Ananya Charu', 'b'),
    # ---- natural stories ----
    q('f39', 'story', 2, "Last summer, Irfan went fishing with his son Javed. Javed's sister Kausar stayed home to help their mother, Lubna, cook dinner for Lubna's father, Majid. How is Majid related to Javed? (a) Paternal grandfather (b) Maternal grandfather (c) Uncle (d) Father-in-law",
      'Irfan son Javed ; Javed sister Kausar ; Kausar mother Lubna ; Lubna father Majid ; ask Javed Majid', 'b'),
    q('f40', 'story', 2, "Nandini was excited to visit her daughter Ojasvi at college. Ojasvi's roommate is her cousin Pari, whose father Qamar is Nandini's brother. How is Qamar related to Ojasvi? (a) Paternal uncle (b) Maternal uncle (c) Father (d) Cousin",
      'Nandini female ; Nandini daughter Ojasvi ; Ojasvi cousin Pari ; Pari father Qamar ; Nandini brother Qamar ; ask Ojasvi Qamar', 'b'),
]


if __name__ == '__main__':
    import argparse
    import collections
    from eval.check_authored import check
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--beam', type=int, default=8)
    a = ap.parse_args()
    check(F)   # gold facts must give every expected option before the model is scored
    if a.ckpt:
        from eval.evaluate import load
        from kinlogic.exam import answer
        from reader import generate
        from train import get_device
        dev = get_device()
        model, vocab = load(a.ckpt, dev)
        read = lambda b: generate(model, vocab, b, dev, k=a.beam)
        by, wrong = collections.defaultdict(lambda: [0, 0]), []
        for it in F:
            r = answer(it['text'], read)
            ok = r.get('option') == it['expect']
            by[it['kind']][0] += ok
            by[it['kind']][1] += 1
            if not ok:
                wrong.append(f"{it['id']} want {it['expect']} got {r.get('option')} ({r.get('answer')})")
        n = sum(v[0] for v in by.values())
        print(f'model: {n}/{len(F)} right | ' + ', '.join(f'{k} {a}/{b}' for k, (a, b) in sorted(by.items())))
        print('\n'.join(wrong))
