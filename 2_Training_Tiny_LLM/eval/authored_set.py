"""Original blood-relation questions written for this project (not copied from any site), each with its exact facts,
the expected option and a difficulty level (1 easy, 2 exam standard, 3 hard). Free to use for training except the
locked 'test' split, which is only ever reported.

Facts use names, "~man" for a description ("a man", "the old man") and #k for unnamed people; see eval/exam_set.py.
Check every item against the engine with:  python -m eval.check_authored
"""


def q(id, kind, level, text, facts, expect, reads=None):
    """reads: for "which expression" questions, each option's own facts (the pipeline reads every option)."""
    return dict(id=id, kind=kind, level=level, text=text, facts=facts, expect=expect, **({'reads': reads} if reads else {}))


A = [
    # ---- pointing, photographs, dialogue ----
    q('sp01', 'speech', 2, 'Pointing to a photograph of a boy, Suresh said, "He is the son of the only son of my mother." How is the boy related to Suresh? (a) Brother (b) Nephew (c) Son (d) Cousin',
      '~boy male ; Suresh mother #1 ; #1 only son #2 ; #2 son ~boy ; ask Suresh ~boy', 'c'),
    q('sp02', 'speech', 2, 'Showing a lady in a photograph, Kavya said, "Her mother is the only daughter of my maternal grandmother." How is the lady related to Kavya? (a) Sister (b) Cousin (c) Niece (d) Aunt',
      '~lady female ; ~lady mother #1 ; Kavya maternal grandmother #2 ; #2 only daughter #1 ; ask Kavya ~lady', 'a'),
    q('sp03', 'speech', 3, 'Looking at a portrait of a man, Arjun said, "His mother is the wife of my father\'s only son. I have no brothers." How is the man related to Arjun? (a) Nephew (b) Son (c) Brother (d) Cousin',
      "~man male ; ~man mother #1 ; Arjun father #2 ; #2 only son #3 ; #3 wife #1 ; Arjun no-brother ; ask Arjun ~man", 'b'),
    q('sp04', 'speech', 2, 'Pointing to a woman, Rahul told his friend, "She is the daughter-in-law of my mother\'s only son." How is the woman related to Rahul? (a) Wife (b) Sister-in-law (c) Daughter (d) Daughter-in-law',
      '~woman female ; Rahul male ; Rahul friend #1 ; Rahul mother #2 ; #2 only son #3 ; #3 daughter-in-law ~woman ; ask Rahul ~woman', 'd'),
    q('sp05', 'speech', 3, 'Pointing to a gentleman, Meena said, "His only brother is the father of my daughter\'s father." How is the gentleman related to Meena? (a) Grandfather (b) Father-in-law (c) Husband\'s uncle (d) Brother-in-law',
      "~gentleman male ; ~gentleman only brother #1 ; Meena daughter #2 ; #2 father #3 ; #3 father #1 ; ask Meena ~gentleman", 'c'),
    q('sp06', 'speech', 3, 'Pointing to a girl in the park, Nikhil said, "She is the granddaughter of the only son of my paternal grandfather, and I am the only child of my parents." How is the girl related to Nikhil? (a) Niece (b) Daughter (c) Sister (d) Cousin',
      '~girl female ; Nikhil paternal grandfather #1 ; #1 only son #2 ; #2 granddaughter ~girl ; Nikhil no-sibling ; ask Nikhil ~girl', 'b'),
    q('sp07', 'speech', 2, 'Pointing to Varun, Anita said, "His father is the only son of my father-in-law." How is Varun related to Anita? (a) Nephew (b) Brother-in-law (c) Son (d) Cousin',
      'Varun male ; Varun father #1 ; Anita father-in-law #2 ; #2 only son #1 ; ask Anita Varun', 'c'),
    q('sp08', 'speech', 2, 'At a wedding, Deepak pointed to an old man and told his cousin, "That man is the father of my mother\'s only brother." How is the old man related to Deepak? (a) Paternal grandfather (b) Maternal uncle (c) Great-uncle (d) Maternal grandfather',
      '~man male ; Deepak male ; Deepak cousin #1 ; Deepak mother #2 ; #2 only brother #3 ; #3 father ~man ; ask Deepak ~man', 'd'),
    q('sp09', 'speech', 2, 'Pointing to a lady on the stage, Ritu said, "She is the mother-in-law of my brother\'s only sister." How is the lady related to Ritu? (a) Mother (b) Mother-in-law (c) Aunt (d) Sister',
      '~lady female ; Ritu brother #1 ; #1 only sister #2 ; #2 mother-in-law ~lady ; ask Ritu ~lady', 'b'),
    q('sp10', 'speech', 1, 'Showing a man on the television, Neha said, "He is the brother of my uncle\'s daughter." How is the man related to Neha? (a) Cousin (b) Brother (c) Nephew (d) Uncle',
      '~man male ; Neha uncle #1 ; #1 daughter #2 ; #2 brother ~man ; ask Neha ~man', 'a'),
    q('sp11', 'speech', 2, 'Pointing to a woman, Harish said, "She is the only daughter of the mother-in-law of my wife." How is the woman related to Harish? (a) Wife (b) Sister-in-law (c) Mother (d) Sister',
      '~woman female ; Harish wife #1 ; #1 mother-in-law #2 ; #2 only daughter ~woman ; ask Harish ~woman', 'd'),
    q('sp12', 'speech', 2, 'A woman pointing to a man said, "His wife is the only daughter of my mother." How is the woman related to the man? (a) Sister (b) Wife (c) Sister-in-law (d) Mother-in-law',
      '~woman female ; ~man male ; ~man wife #1 ; ~woman mother #2 ; #2 only daughter #1 ; ask ~man ~woman', 'b'),
    q('sp13', 'speech', 2, 'Pointing to a boy, Kiran said, "He is the son of my mother\'s brother\'s only sister." How is the boy related to Kiran? (a) Cousin (b) Nephew (c) Brother (d) Son',
      '~boy male ; Kiran mother #1 ; #1 brother #2 ; #2 only sister #3 ; #3 son ~boy ; ask Kiran ~boy', 'c'),
    q('sp14', 'speech', 2, 'Pointing to a girl, Sam said, "She is the daughter of the only child of my mother." How is the girl related to Sam? (a) Niece (b) Daughter (c) Sister (d) Cousin',
      '~girl female ; Sam mother #1 ; #1 only child #2 ; #2 daughter ~girl ; ask Sam ~girl', 'b'),
    q('sp15', 'speech', 3, 'Referring to a man in a group photo, Leela told her husband, "His sister is the daughter of your father\'s only sister." How is the man related to Leela\'s husband? (a) Brother (b) Nephew (c) Uncle (d) Cousin',
      '~man male ; Leela female ; Leela husband #1 ; ~man sister #2 ; #1 father #3 ; #3 only sister #4 ; #4 daughter #2 ; ask #1 ~man', 'd'),

    # ---- introductions ----
    q('in01', 'introduce', 2, "A man introduces a woman as the daughter of the sister of his father's wife. How is the woman related to the man? (a) Sister (b) Niece (c) Cousin (d) Aunt",
      '~man male ; ~woman female ; ~man father #1 ; #1 wife #2 ; #2 sister #3 ; #3 daughter ~woman ; ask ~man ~woman', 'c'),
    q('in02', 'introduce', 3, 'Introducing a girl, a boy said, "She is the only daughter of my paternal grandmother\'s only daughter-in-law." How is the girl related to the boy? (a) Cousin (b) Niece (c) Aunt (d) Sister',
      '~girl female ; ~boy male ; ~boy paternal grandmother #1 ; #1 only daughter-in-law #2 ; #2 only daughter ~girl ; ask ~boy ~girl', 'd'),
    q('in03', 'introduce', 2, 'Introducing Rakesh, Sneha said, "His father is the only son of my father." How is Sneha related to Rakesh? (a) Mother (b) Aunt (c) Sister (d) Niece',
      'Rakesh male ; Rakesh father #1 ; Sneha father #2 ; #2 only son #1 ; ask Rakesh Sneha', 'b'),
    q('in04', 'introduce', 3, "A lady introduces a young man as the only son of the father-in-law of her mother's sister. How is the young man related to the lady? (a) Cousin (b) Brother (c) Uncle (d) Nephew",
      '~lady female ; ~man male ; ~lady mother #1 ; #1 sister #2 ; #2 father-in-law #3 ; #3 only son ~man ; ask ~lady ~man', 'c'),
    q('in05', 'introduce', 2, 'While introducing a woman, a man said, "Her husband is the only son of my mother\'s only son." How is the woman related to the man? (a) Daughter-in-law (b) Daughter (c) Sister-in-law (d) Niece',
      '~woman female ; ~man male ; ~woman husband #1 ; ~man mother #2 ; #2 only son #3 ; #3 only son #1 ; ask ~man ~woman', 'a'),
    q('in06', 'introduce', 3, 'Introducing a boy, Ajay said, "He is the son of the only daughter of the father of my paternal uncle." How is the boy related to Ajay? (a) Brother (b) Nephew (c) Cousin (d) Uncle',
      '~boy male ; Ajay paternal uncle #1 ; #1 father #2 ; #2 only daughter #3 ; #3 son ~boy ; ask Ajay ~boy', 'c'),
    q('in07', 'introduce', 3, 'Introducing Priya, Gautam said, "She is the wife of my father\'s only grandson." Gautam has no brothers or sisters. How is Priya related to Gautam? (a) Wife (b) Sister-in-law (c) Daughter-in-law (d) Niece',
      'Priya female ; Gautam father #1 ; #1 only grandson #2 ; #2 wife Priya ; Gautam no-sibling ; ask Gautam Priya', 'c'),

    # ---- nested descriptions ----
    q('ne01', 'nested', 1, "Ramesh is the son of the only brother of Sita's mother. How is Ramesh related to Sita? (a) Brother (b) Maternal cousin (c) Nephew (d) Uncle",
      'Sita mother #1 ; #1 only brother #2 ; #2 son Ramesh ; ask Sita Ramesh', 'b'),
    q('ne02', 'nested', 3, "Meera is the daughter of the son of the only daughter of Anil's paternal grandmother. How is Meera related to Anil? (a) Niece (b) Cousin (c) Cousin's daughter (d) Granddaughter",
      'Anil paternal grandmother #1 ; #1 only daughter #2 ; #2 son #3 ; #3 daughter Meera ; ask Anil Meera', 'c'),
    q('ne03', 'nested', 2, "Kunal is the father of the husband of Tina's only sister. How is Kunal related to Tina? (a) Sister's father-in-law (b) Father-in-law (c) Uncle (d) Grandfather",
      'Tina only sister #1 ; #1 husband #2 ; #2 father Kunal ; ask Tina Kunal', 'a'),
    q('ne04', 'nested', 2, 'Asha is the mother of the brother of the father of Bhanu. How is Asha related to Bhanu? (a) Maternal grandmother (b) Paternal aunt (c) Paternal grandmother (d) Mother',
      'Bhanu father #1 ; #1 brother #2 ; #2 mother Asha ; ask Bhanu Asha', 'c'),
    q('ne05', 'nested', 3, 'How is the husband of the daughter of the only son of my maternal grandfather related to me? (a) Brother-in-law (b) Uncle (c) Nephew (d) Cousin\'s husband',
      '#1 maternal grandfather #2 ; #2 only son #3 ; #3 daughter #4 ; #4 husband #5 ; ask #1 #5', 'd'),
    q('ne06', 'nested', 2, "Pramod is the brother of the husband of Lata's daughter. How is Pramod related to Lata? (a) Son-in-law (b) Son-in-law's brother (c) Son (d) Brother-in-law",
      'Lata daughter #1 ; #1 husband #2 ; #2 brother Pramod ; ask Lata Pramod', 'b'),
    q('ne07', 'nested', 1, 'Q is the son of the only sister of the father of R, and R is a girl. How is R related to Q? (a) Cousin (b) Sister (c) Niece (d) Aunt',
      'R father #1 ; #1 only sister #2 ; #2 son Q ; R female ; ask Q R', 'a'),
    q('ne08', 'nested', 3, "Nisha is the wife of the son of the brother of the father of Vikram's mother. How is Nisha related to Vikram? (a) Aunt (b) Great-aunt (c) Cousin (d) Mother's cousin's wife",
      'Vikram mother #1 ; #1 father #2 ; #2 brother #3 ; #3 son #4 ; #4 wife Nisha ; ask Vikram Nisha', 'd'),
    q('ne09', 'nested', 2, "Dev is the only grandson of the only son of Rohan's paternal grandfather, and Rohan has no brothers or sisters. How is Dev related to Rohan? (a) Son (b) Nephew (c) Grandson (d) Cousin",
      'Rohan paternal grandfather #1 ; #1 only son #2 ; #2 only grandson Dev ; Rohan no-sibling ; ask Rohan Dev', 'a'),
    q('ne10', 'nested', 1, "The wife of the only son of Sunil's mother's mother is Kalpana. How is Kalpana related to Sunil? (a) Mother (b) Aunt (c) Grandmother (d) Sister-in-law",
      'Sunil mother #1 ; #1 mother #2 ; #2 only son #3 ; #3 wife Kalpana ; ask Sunil Kalpana', 'b'),
    q('ne11', 'nested', 2, "Ira is the only daughter of the husband of Tarun's father's sister. How is Ira related to Tarun? (a) Sister (b) Niece (c) Cousin (d) Aunt",
      'Tarun father #1 ; #1 sister #2 ; #2 husband #3 ; #3 only daughter Ira ; ask Tarun Ira', 'c'),
    q('ne12', 'nested', 3, 'Om is the father of the father of the mother of the father of Lila. How is Om related to Lila? (a) Great-grandfather (b) Great-great-grandfather (c) Grandfather (d) Great-uncle',
      'Lila father #1 ; #1 mother #2 ; #2 father #3 ; #3 father Om ; ask Lila Om', 'b'),

    # ---- statements ----
    q('st01', 'statements', 1, 'A is the brother of B. C is the mother of A. D is the father of C. E is the daughter of D. How is E related to B? (a) Mother (b) Sister (c) Maternal aunt (d) Grandmother',
      'B brother A ; A mother C ; C father D ; D daughter E ; ask B E', 'c'),
    q('st02', 'statements', 2, 'M is the sister of N. N is the father of O. O is the wife of P. Q is the son of P. How is M related to Q? (a) Grandmother (b) Aunt (c) Mother-in-law (d) Great-aunt',
      'N sister M ; O father N ; P wife O ; P son Q ; ask Q M', 'd'),
    q('st03', 'statements', 1, 'R is the wife of S. T is the brother of S. U is the son of T. V is the daughter of R. How is U related to V? (a) Brother (b) Cousin (c) Nephew (d) Uncle',
      'S wife R ; S brother T ; T son U ; R daughter V ; ask V U', 'b'),
    q('st04', 'statements', 1, 'J is the child of K. K is the sister of L. L is the father of M. How is J related to M? (a) Brother (b) Sister (c) Nephew (d) Cousin',
      'K child J ; L sister K ; M father L ; ask M J', 'd'),
    q('st05', 'statements', 2, 'P is a sibling of Q. Q is the son of R. How is P related to R? (a) Son (b) Daughter (c) Son or daughter (d) Nephew',
      'Q sibling P ; R son Q ; ask R P', 'c'),
    q('st06', 'statements', 2, 'F is the mother of G. G is the husband of H. I is the brother of H. J is the son of I. How is F related to J? (a) Grandmother (b) Aunt (c) Mother (d) Aunt\'s mother-in-law',
      'G mother F ; H husband G ; H brother I ; I son J ; ask J F', 'd'),
    q('st07', 'statements', 2, 'X is the only daughter of Y. Z is the mother of Y. W is the son of Z and is unmarried. How is W related to X? (a) Uncle (b) Father (c) Brother (d) Grandfather',
      'Y only daughter X ; Y mother Z ; Z son W ; W unmarried ; ask X W', 'a'),
    q('st08', 'statements', 1, 'A has two sons, B and C. D is the wife of B. E is the daughter of C. How is D related to E? (a) Mother (b) Sister (c) Aunt (d) Cousin',
      'A has 2 son ; A son B ; A son C ; B wife D ; C daughter E ; ask E D', 'c'),
    q('st09', 'statements', 1, "Kamal's father is the brother of Neelam's father. Neelam's only brother is Ravi. How is Kamal related to Ravi? (a) Brother (b) Uncle (c) Cousin (d) Nephew",
      'Kamal father #1 ; Neelam father #2 ; #2 brother #1 ; Neelam only brother Ravi ; ask Ravi Kamal', 'c'),
    q('st10', 'statements', 2, 'D is the brother of E. F is the father of D. G is the brother of F. H is the father of G. How is E related to H? (a) Grandson (b) Granddaughter (c) Son (d) Grandson or granddaughter',
      'E brother D ; D father F ; F brother G ; G father H ; ask H E', 'd'),
    q('st11', 'statements', 2, "Lalit is the father of Mona. Mona is the sister of Nitin. Nitin is married to Ojas's daughter Pari. How is Lalit related to Pari? (a) Father (b) Uncle (c) Father-in-law (d) Grandfather",
      'Mona father Lalit ; Nitin sister Mona ; Nitin spouse Pari ; Ojas daughter Pari ; ask Pari Lalit', 'c'),

    # ---- gender from names, no pronouns ----
    q('nm01', 'names', 2, 'Priyanka is the child of Suresh. Suresh is the brother of Geeta. How is Priyanka related to Geeta? (a) Nephew (b) Niece (c) Daughter (d) Cousin',
      'Suresh child Priyanka ; Geeta brother Suresh ; ask Geeta Priyanka', 'b'),
    q('nm02', 'names', 1, 'Arvind and Sunita are the parents of Rahul and Pooja. How is Pooja related to Rahul? (a) Brother (b) Cousin (c) Mother (d) Sister',
      'Rahul parent Arvind ; Rahul parent Sunita ; Pooja parent Arvind ; Pooja parent Sunita ; ask Rahul Pooja', 'd'),
    q('nm03', 'names', 1, 'Deepika is the spouse of Manoj. Manoj is the child of Harish. How is Deepika related to Harish? (a) Daughter (b) Daughter-in-law (c) Wife (d) Sister',
      'Manoj spouse Deepika ; Harish child Manoj ; ask Harish Deepika', 'b'),
    q('nm04', 'names', 2, "Leela is a parent of Karan. Mohan is Leela's sibling, and Nandini is Mohan's child. How is Mohan related to Karan? (a) Paternal uncle (b) Father (c) Maternal uncle (d) Grandfather",
      'Karan parent Leela ; Leela sibling Mohan ; Mohan child Nandini ; ask Karan Mohan', 'c'),
    q('nm05', 'names', 2, 'Sneha is a sibling of Vivek. Vivek is a parent of Isha. How is Sneha related to Isha? (a) Maternal aunt (b) Paternal aunt (c) Mother (d) Sister',
      'Vivek sibling Sneha ; Isha parent Vivek ; ask Isha Sneha', 'b'),
    q('nm06', 'names', 3, "Anand is married to Rekha. Pradeep is Rekha's sibling, and Shobha is Pradeep's spouse. How is Shobha related to Anand? (a) Wife's sister (b) Cousin (c) Wife's brother's wife (d) Aunt",
      'Anand spouse Rekha ; Rekha sibling Pradeep ; Pradeep spouse Shobha ; ask Anand Shobha', 'c'),
    q('nm07', 'names', 2, "Gauri and Tushar are siblings. Ved is Tushar's child and Rhea is Ved's child. How is Gauri related to Rhea? (a) Grandmother (b) Aunt (c) Great-grandmother (d) Great-aunt",
      'Gauri sibling Tushar ; Tushar child Ved ; Ved child Rhea ; ask Rhea Gauri', 'd'),
    q('nm08', 'names', 3, "Harsha is married to Kiran, and Mahesh is a parent of Kiran. Who is the father-in-law of Harsha? (a) Kiran (b) Mahesh (c) Harsha (d) Cannot be determined",
      'Harsha spouse Kiran ; Kiran parent Mahesh ; who Harsha father-in-law', 'b'),

    # ---- cousins, in-laws, grand and great relations, Hindi terms ----
    q('co01', 'cousin', 2, "Kabir is the son of the daughter of Neha's father's brother. How is Kabir related to Neha? (a) Nephew (b) Cousin (c) Cousin's son (d) Grandson",
      'Neha father #1 ; #1 brother #2 ; #2 daughter #3 ; #3 son Kabir ; ask Neha Kabir', 'c'),
    q('co02', 'cousin', 3, "Riya is the daughter of the son of the brother of Asha's mother's mother. How is Riya related to Asha? (a) Cousin (b) Niece (c) Aunt (d) Second cousin",
      'Asha mother #1 ; #1 mother #2 ; #2 brother #3 ; #3 son #4 ; #4 daughter Riya ; ask Asha Riya', 'd'),
    q('co03', 'inlaw', 2, 'Vinod is the husband of Rita. Rita\'s sister Mala is married to Kishan. How is Kishan related to Vinod? (a) Co-brother (b) Brother (c) Cousin (d) Uncle',
      'Rita husband Vinod ; Rita sister Mala ; Mala spouse Kishan ; ask Vinod Kishan', 'a'),
    q('co04', 'hindi', 2, "Mahesh is Rohan's mama, and Tanvi is Mahesh's daughter. How is Tanvi related to Rohan? (a) Sister (b) Niece (c) Maternal cousin (d) Aunt",
      'Rohan maternal uncle Mahesh ; Mahesh daughter Tanvi ; ask Rohan Tanvi', 'c'),
    q('co05', 'hindi', 2, "Pihu's bua is married to Rakesh, and Yash is Rakesh's son. How is Yash related to Pihu? (a) Brother (b) Nephew (c) Cousin (d) Uncle",
      'Pihu paternal aunt #1 ; #1 spouse Rakesh ; Rakesh son Yash ; ask Pihu Yash', 'c'),
    q('co06', 'inlaw', 1, 'Simran is the daughter-in-law of Prakash. Prakash has only one child, Aman. How is Simran related to Aman? (a) Sister (b) Wife (c) Sister-in-law (d) Daughter',
      'Prakash daughter-in-law Simran ; Prakash has 1 child ; Prakash child Aman ; ask Aman Simran', 'b'),
    q('co07', 'great', 1, "Kusum is the mother of Ashok's grandmother. How is Ashok related to Kusum? (a) Grandson (b) Great-grandson (c) Nephew (d) Great-nephew",
      'Ashok male ; Ashok grandmother #1 ; #1 mother Kusum ; ask Kusum Ashok', 'b'),
    q('co08', 'great', 2, "Hari is the brother of Pawan's paternal grandfather. How is Pawan related to Hari? (a) Nephew (b) Grandson (c) Grand-nephew (d) Cousin",
      'Pawan male ; Pawan paternal grandfather #1 ; #1 brother Hari ; ask Hari Pawan', 'c'),

    # ---- deep chains ----
    q('dp01', 'deep', 3, 'A is the father of B, B is the father of C, C is the mother of D, D is the father of E, E is the mother of F and F is the father of G. How is A related to G? (a) Great-great-great-grandfather (b) Great-great-grandfather (c) Great-great-great-great-grandfather (d) Great-grandfather',
      'B father A ; C father B ; D mother C ; E father D ; F mother E ; G father F ; ask G A', 'c'),
    # ---- closed-family puzzles ----
    q('pz01', 'puzzle', 2, 'There are six members A, B, C, D, E and F in a family of three generations with two married couples. B is the son of A. C is the wife of B. D is the daughter of C. E is the husband of A. F is the brother of D. How many female members are there in the family? (a) 2 (b) 3 (c) 4 (d) Cannot be determined',
      'members 6 ; couples 2 ; generations 3 ; A son B ; B wife C ; C daughter D ; A husband E ; D brother F ; count female', 'b'),
    q('pz02', 'puzzle', 2, 'There are six members A, B, C, D, E and F in a family of three generations with two married couples. B is the son of A. C is the wife of B. D is the daughter of C. E is the husband of A. F is the brother of D. Who is the grandfather of F? (a) A (b) B (c) E (d) C',
      'members 6 ; couples 2 ; generations 3 ; A son B ; B wife C ; C daughter D ; A husband E ; D brother F ; who F grandfather', 'c'),
    q('pz03', 'puzzle', 1, 'There are six members A, B, C, D, E and F in a family of three generations with two married couples. B is the son of A. C is the wife of B. D is the daughter of C. E is the husband of A. F is the brother of D. How is C related to E? (a) Daughter (b) Daughter-in-law (c) Wife (d) Sister',
      'members 6 ; couples 2 ; generations 3 ; A son B ; B wife C ; C daughter D ; A husband E ; D brother F ; ask E C', 'b'),
    q('pz04', 'puzzle', 2, 'In a family of five persons J, K, L, M and N there is only one married couple. J is the mother of K. L is the brother of K. M is the father of L. N is the daughter of J. K is the son of M. Which of the following is a pair of brothers? (a) K, L (b) L, N (c) J, M (d) K, N',
      'members 5 ; couples 1 ; K mother J ; K brother L ; L father M ; J daughter N ; M son K ; pair brother', 'a'),
    q('pz05', 'puzzle', 2, 'P, Q, R, S and T are members of a family. P is the mother of Q. R is the husband of P. S is the sister of Q. T is the son of R. What is the gender of Q? (a) Male (b) Female (c) Cannot be determined (d) None of these',
      'members 5 ; Q mother P ; P husband R ; Q sister S ; R son T ; gender Q', 'c'),
    q('pz06', 'puzzle', 3, 'There are seven members A, B, C, D, E, F and G in a family of three generations with two married couples. A is the grandmother of F. B is the son of A and the father of F. C is the wife of B. D is the daughter of A. E is the husband of D. G is the son of E. F is a girl. How many male members are there in the family? (a) 2 (b) 3 (c) 4 (d) 5',
      'members 7 ; generations 3 ; couples 2 ; F grandmother A ; A son B ; F father B ; B wife C ; A daughter D ; D husband E ; E son G ; F female ; count male', 'b'),
    q('pz07', 'puzzle', 2, 'There are seven members A, B, C, D, E, F and G in a family of three generations with two married couples. A is the grandmother of F. B is the son of A and the father of F. C is the wife of B. D is the daughter of A. E is the husband of D. G is the son of E. F is a girl. How is G related to F? (a) Brother (b) Cousin (c) Nephew (d) Uncle',
      'members 7 ; generations 3 ; couples 2 ; F grandmother A ; A son B ; F father B ; B wife C ; A daughter D ; D husband E ; E son G ; F female ; ask F G', 'b'),

    # ---- data sufficiency ----
    q('ds01', 'sufficiency', 2, "How is M related to K? I. M is the daughter of the husband of K's only sister. II. K's sister has no children other than M. (a) Only I is sufficient (b) Only II is sufficient (c) Either I or II is sufficient (d) Both I and II together are sufficient (e) Neither I nor II is sufficient",
      'ask K M ; part 1 ; K only sister #1 ; #1 husband #2 ; #2 daughter M ; part 2 ; K sister #3 ; #3 only child M', 'a'),
    q('ds02', 'sufficiency', 2, "How is X related to Y? I. Y is the father of Z, and Z is the brother of X. II. X is the daughter of Y's wife. (a) Only I is sufficient (b) Only II is sufficient (c) Either I or II is sufficient (d) Both I and II together are sufficient (e) Neither I nor II is sufficient",
      'ask Y X ; part 1 ; Z father Y ; X brother Z ; part 2 ; Y wife #1 ; #1 daughter X', 'b'),
    q('ds03', 'sufficiency', 3, 'How is P related to Q? I. P is the mother of R, and R is the father of Q. II. P is the wife of S, and S is the grandfather of Q. (a) Only I is sufficient (b) Only II is sufficient (c) Either I or II is sufficient (d) Both I and II together are sufficient (e) Neither I nor II is sufficient',
      'ask Q P ; part 1 ; R mother P ; Q father R ; part 2 ; S wife P ; Q grandfather S', 'c'),
    q('ds04', 'sufficiency', 2, 'How is A related to D? I. A is the brother of B, and B is the son of C. II. C is the daughter of D. (a) Only I is sufficient (b) Only II is sufficient (c) Either I or II is sufficient (d) Both I and II together are sufficient (e) Neither I nor II is sufficient',
      'ask D A ; part 1 ; B brother A ; C son B ; part 2 ; D daughter C', 'd'),
    q('ds05', 'sufficiency', 1, 'How is K related to L? I. K is the brother of M. II. L is the friend of M. (a) Only I is sufficient (b) Only II is sufficient (c) Either I or II is sufficient (d) Both I and II together are sufficient (e) Neither I nor II is sufficient',
      'ask L K ; part 1 ; M brother K ; part 2 ; M friend L', 'e'),

    # ---- definitely true ----
    q('cl01', 'claims', 2, 'A is the mother of B. B is the sister of C. D is the son of C. E is the husband of A. Which of the following is/are definitely true? I. E is the grandfather of D. II. B is the paternal aunt of D. III. C is the son of A. (a) Only I (b) Only I and II (c) Only II and III (d) All of these',
      'B mother A ; C sister B ; C son D ; A husband E ; claim 1 ; D grandfather E ; claim 2 ; D paternal aunt B ; claim 3 ; A son C', 'a'),
    q('cl02', 'claims', 2, 'P is the son of Q. R is the wife of Q. S is the daughter of R. T is the brother of R. How many of the following statements are definitely true? I. T is the maternal uncle of P. II. S is the sister of P. III. Q is the father of S. IV. P is the nephew of T. (a) One (b) Two (c) Three (d) Four',
      'Q son P ; Q wife R ; R daughter S ; R brother T ; claim 1 ; P maternal uncle T ; claim 2 ; P sister S ; claim 3 ; S father Q ; claim 4 ; T nephew P', 'd'),

    # ---- coded relations ----
    q('cd01', 'coded', 2, 'If P × Q means P is the brother of Q, P + Q means P is the mother of Q, P − Q means P is the father of Q and P ÷ Q means P is the sister of Q, how is D related to A in A + B ÷ C × D? (a) Son (b) Daughter (c) Son or daughter (d) Nephew',
      'B mother A ; C sister B ; D brother C ; ask A D', 'c'),
    q('cd02', 'coded', 3, "'X @ Y' means 'Y is the father of X', 'X # Y' means 'Y is the sister of X' and 'X $ Y' means 'Y is the son of X'. In the expression K @ L # M $ N, how is N related to K? (a) Brother (b) Cousin (c) Nephew (d) Uncle",
      'K father L ; L sister M ; M son N ; ask K N', 'b'),
    q('cd03', 'coded', 2, 'If A $ B means A is the wife of B, A % B means A is the son of B and A & B means A is the brother of B, then in the expression P & Q % R $ S, which of the following is definitely true? I. S is the father of P. II. P is the nephew of S. (a) Only I (b) Only II (c) Both I and II (d) Neither I nor II',
      'Q brother P ; R son Q ; S wife R ; claim 1 ; P father S ; claim 2 ; S nephew P', 'a'),
    q('cd04', 'coded', 3, 'A + B means A is the father of B; A − B means A is the husband of B; A × B means A is the brother of B; A ÷ B means A is the daughter of B. If P + Q × R − S ÷ T, how is S related to P? (a) Daughter (b) Daughter-in-law (c) Sister-in-law (d) Niece',
      'Q father P ; R brother Q ; S husband R ; T daughter S ; ask P S', 'b'),
    q('cd05', 'coded', 2, 'If M # N means M is the mother of N, M @ N means M is the brother of N and M $ N means M is the son of N, how is X related to Z in X @ Y # Z? (a) Paternal uncle (b) Maternal uncle (c) Father (d) Brother',
      'Y brother X ; Z mother Y ; ask Z X', 'b'),

    # ---- more dialogue ----
    q('sp16', 'speech', 3, 'Pointing to a man, a woman said, "His mother\'s only daughter is my mother." How is the woman related to the man? (a) Niece (b) Sister (c) Daughter (d) Cousin',
      '~man male ; ~woman female ; ~man mother #1 ; #1 only daughter #2 ; ~woman mother #2 ; ask ~man ~woman', 'a'),
    q('sp17', 'speech', 3, 'A man said to a woman, "Your mother\'s husband\'s sister is my aunt." How is the woman related to the man? (a) Granddaughter (b) Daughter (c) Sister (d) Aunt',
      '~man male ; ~woman female ; ~woman mother #1 ; #1 husband #2 ; #2 sister #3 ; ~man aunt #3 ; ask ~man ~woman', 'c'),
    q('sp18', 'speech', 1, 'Introducing a man, Shalu said, "He is the only son of my mother\'s father." How is the man related to Shalu? (a) Uncle (b) Brother (c) Father (d) Cousin',
      '~man male ; Shalu mother #1 ; #1 father #2 ; #2 only son ~man ; ask Shalu ~man', 'a'),
    q('sp19', 'speech', 3, 'Seeing Raghav at a party, Ishita told her son, "Raghav\'s father is the brother of your paternal grandfather\'s only daughter-in-law." How is Raghav related to Ishita\'s son? (a) Brother (b) Cousin (c) Uncle (d) Nephew',
      'Ishita female ; Ishita son #1 ; Raghav father #2 ; #1 paternal grandfather #3 ; #3 only daughter-in-law #4 ; #4 brother #2 ; ask #1 Raghav', 'b'),
    q('sp20', 'speech', 2, 'Rohit told Kavi, "The girl I met yesterday is the youngest daughter of the brother-in-law of my friend\'s mother." How is the girl related to Rohit\'s friend? (a) Sister (b) Cousin (c) Niece (d) Aunt',
      '~girl female ; Rohit friend #1 ; #1 mother #2 ; #2 brother-in-law #3 ; #3 daughter ~girl ; ask #1 ~girl', 'b'),

    # ---- more statements: counts, only, negation ----
    q('st12', 'statements', 2, 'Gopal has three children, Anu, Binu and Chinu. Anu and Binu are daughters. Chinu is married to Deepa. How is Deepa related to Anu? (a) Sister (b) Sister-in-law (c) Daughter-in-law (d) Cousin',
      'Gopal has 3 child ; Gopal child Anu ; Gopal child Binu ; Gopal child Chinu ; Anu female ; Binu female ; Chinu spouse Deepa ; ask Anu Deepa', 'b'),
    q('st13', 'statements', 1, "Kishore has only one sister, Lata, and no brothers. Manu is Lata's son and Neel is Kishore's son. How is Neel related to Lata? (a) Son (b) Nephew (c) Cousin (d) Brother",
      'Kishore only sister Lata ; Kishore no-brother ; Lata son Manu ; Kishore son Neel ; ask Lata Neel', 'b'),
    q('st14', 'statements', 1, 'X is the father of Y, but Y is not the son of X. Z is the brother of X. How is Y related to Z? (a) Nephew (b) Niece (c) Daughter (d) Sister',
      'Y father X ; X not son Y ; X brother Z ; ask Z Y', 'b'),
    q('st15', 'statements', 3, 'P is a parent of Q, but P is not the mother of Q. R is the wife of P. S is the only daughter of R. T is the brother of S. How is Q related to T? (a) Sister (b) Brother (c) Cousin (d) Cannot be determined',
      'Q parent P ; Q not mother P ; P wife R ; R only daughter S ; S brother T ; ask T Q', 'b'),

    # ---- Hindi terms ----
    q('hi01', 'hindi', 1, "Aarav's chacha is the father of Isha. How is Isha related to Aarav? (a) Sister (b) Cousin (c) Niece (d) Aunt",
      'Aarav paternal uncle #1 ; Isha father #1 ; ask Aarav Isha', 'b'),
    q('hi02', 'hindi', 1, "Nidhi's nani has only one son, Vinod. How is Vinod related to Nidhi? (a) Paternal uncle (b) Maternal uncle (c) Grandfather (d) Father",
      'Nidhi maternal grandmother #1 ; #1 only son Vinod ; ask Nidhi Vinod', 'b'),

    # ---- deep chains ----
    q('dp02', 'deep', 3, "Bhim is the son of the brother of the father of the father of Raman's mother. How is Bhim related to Raman? (a) Great-uncle (b) Maternal grandfather's cousin (c) Cousin (d) Great-grandfather",
      'Raman mother #1 ; #1 father #2 ; #2 father #3 ; #3 brother #4 ; #4 son Bhim ; ask Raman Bhim', 'b'),
    q('dp03', 'deep', 3, 'A is the mother of B, B is the father of C, C is the father of D, D is the mother of E, E is the father of F, F is the mother of G, G is the father of H, H is the father of I, I is the mother of J and J is the father of K, who is a boy. How is K related to A? (a) great-great-great-great-great-great-great-grandson (b) great-great-great-great-great-great-great-great-grandson (c) great-great-great-great-great-great-great-great-great-grandson (d) great-great-great-great-great-great-grandson',
      'B mother A ; C father B ; D father C ; E mother D ; F father E ; G mother F ; H father G ; I father H ; J mother I ; K father J ; K male ; ask A K', 'b'),
    # ---- batch 3: dialogue ----
    q('sp21', 'speech', 1, 'Looking at a photograph, Tanvi said, "This man is the husband of my mother\'s only sister." How is the man related to Tanvi? (a) Uncle (b) Father (c) Brother-in-law (d) Grandfather',
      '~man male ; Tanvi mother #1 ; #1 only sister #2 ; #2 husband ~man ; ask Tanvi ~man', 'a'),
    q('sp22', 'speech', 3, 'Pointing to a lady, Kunal said, "The son of her only brother is the brother of my wife." How is the lady related to Kunal? (a) Mother\'s sister (b) Sister of father-in-law (c) Grandmother (d) Mother-in-law',
      '~lady female ; ~lady only brother #1 ; #1 son #2 ; Kunal wife #3 ; #3 brother #2 ; ask Kunal ~lady', 'b'),
    q('sp23', 'speech', 2, 'Pointing to a boy, Rina said, "He is the son of the only daughter of the father of my brother." How is the boy related to Rina? (a) Brother (b) Nephew (c) Son (d) Cousin',
      '~boy male ; Rina brother #1 ; #1 father #2 ; #2 only daughter #3 ; #3 son ~boy ; ask Rina ~boy', 'c'),
    q('sp24', 'speech', 2, 'On seeing a woman, Varsha said to her daughter, "She is the only daughter-in-law of your maternal grandmother." How is the woman related to Varsha? (a) Sister (b) Sister-in-law (c) Mother (d) Aunt',
      '~woman female ; Varsha female ; Varsha daughter #1 ; #1 maternal grandmother #2 ; #2 only daughter-in-law ~woman ; ask Varsha ~woman', 'b'),
    q('sp25', 'speech', 2, 'Pointing to Sonia, Aman said, "I am the only son of her mother\'s only brother." How is Sonia related to Aman? (a) Sister (b) Cousin (c) Niece (d) Aunt',
      'Sonia female ; Sonia mother #1 ; #1 only brother #2 ; #2 only son Aman ; ask Aman Sonia', 'b'),
    q('sp26', 'speech', 3, 'Showing a picture, Neeraj said, "The woman in the picture is the mother of the only grandson of my mother\'s only son." Neeraj has no daughter and no brother. How is the woman related to Neeraj? (a) Daughter (b) Daughter-in-law (c) Wife (d) Sister',
      '~woman female ; Neeraj mother #1 ; #1 only son #2 ; #2 only grandson #3 ; #3 mother ~woman ; Neeraj has 0 daughter ; Neeraj no-brother ; ask Neeraj ~woman', 'b'),
    q('sp27', 'speech', 3, 'Ravi said to Sumit, "My father is the only son of your grandfather." Sumit has no brothers or sisters. How is Ravi related to Sumit? (a) Brother (b) Cousin (c) Nephew (d) Uncle',
      'Ravi father #1 ; Sumit grandfather #2 ; #2 only son #1 ; Sumit no-sibling ; ask Sumit Ravi', 'b'),
    q('sp28', 'speech', 2, 'Kriti said to Nitin, "Your mother is the sister of my mother\'s husband." How is Nitin related to Kriti? (a) Brother (b) Cousin (c) Nephew (d) Uncle',
      'Nitin mother #1 ; Kriti mother #2 ; #2 husband #3 ; #3 sister #1 ; ask Kriti Nitin', 'b'),

    # ---- batch 3: counts, who, gender ----
    q('ct01', 'count', 1, 'P is the father of Q, R and S. Q and R are boys and S is a girl. T is the wife of P. How many sons does T have? (a) 1 (b) 2 (c) 3 (d) Cannot be determined',
      'Q father P ; R father P ; S father P ; Q male ; R male ; S female ; P wife T ; count T son', 'b'),
    q('ct02', 'count', 2, 'A family has five members V, W, X, Y and Z. X is the grandfather of Y and Z. Y is the son of W and Z is the daughter of W. V is the wife of W. How many children does V have? (a) 1 (b) 2 (c) 3 (d) 4',
      'members 5 ; Y grandfather X ; Z grandfather X ; W son Y ; W daughter Z ; W wife V ; count V child', 'b'),
    q('wh01', 'who', 2, 'R is the father of T. S is the mother of T. U is the son of S. V is the daughter of R. Who is definitely the sister of U? (a) S (b) T (c) V (d) Cannot be determined',
      'T father R ; T mother S ; S son U ; R daughter V ; who U sister', 'c'),
    q('gd01', 'gender', 1, 'A is the brother of B. B is the sister of C. C is the father of D. What is the gender of B? (a) Male (b) Female (c) Cannot be determined (d) None of these',
      'B brother A ; C sister B ; D father C ; gender B', 'b'),
    q('gd02', 'gender', 2, 'M is a child of N. N is the wife of O. P is the brother of M. Q is the uncle of P. What is the gender of M? (a) Male (b) Female (c) Cannot be determined (d) None of these',
      'N child M ; O wife N ; M brother P ; P uncle Q ; gender M', 'c'),

    # ---- batch 3: three-couple puzzle ----
    q('pz08', 'puzzle', 3, 'There are eight members P, Q, R, S, T, U, V and W in a family of three generations with three married couples. P is the father of R. Q is the wife of P. R is married to S. T is the daughter of R. U is the son of S. V is the husband of T. W is the mother of S. How is W related to T? (a) Mother (b) Grandmother (c) Aunt (d) Sister',
      'members 8 ; generations 3 ; couples 3 ; R father P ; P wife Q ; R spouse S ; R daughter T ; S son U ; T husband V ; S mother W ; ask T W', 'b'),
    q('pz09', 'puzzle', 3, 'There are eight members P, Q, R, S, T, U, V and W in a family of three generations with three married couples. P is the father of R. Q is the wife of P. R is married to S. T is the daughter of R. U is the son of S. V is the husband of T. W is the mother of S. How many male members are there in the family? (a) 3 (b) 4 (c) 5 (d) Cannot be determined',
      'members 8 ; generations 3 ; couples 3 ; R father P ; P wife Q ; R spouse S ; R daughter T ; S son U ; T husband V ; S mother W ; count male', 'b'),
    q('pz10', 'puzzle', 2, 'There are eight members P, Q, R, S, T, U, V and W in a family of three generations with three married couples. P is the father of R. Q is the wife of P. R is married to S. T is the daughter of R. U is the son of S. V is the husband of T. W is the mother of S. Which of the following is a married couple? (a) P, W (b) S, V (c) R, S (d) Q, R',
      'members 8 ; generations 3 ; couples 3 ; R father P ; P wife Q ; R spouse S ; R daughter T ; S son U ; T husband V ; S mother W ; pair couple', 'c'),

    # ---- batch 3: claims ----
    q('cl03', 'claims', 3, 'A is the father of B and C. B is the husband of D. E is the son of D. C is unmarried. Which of the following cannot be definitely true? (a) E is the grandson of A (b) C is the uncle of E (c) D is the daughter-in-law of A (d) B is the father of E',
      'B father A ; C father A ; D husband B ; D son E ; C unmarried ; claim 1 ; A grandson E ; claim 2 ; E uncle C ; claim 3 ; A daughter-in-law D ; claim 4 ; E father B', 'b'),
    q('cl04', 'claims', 2, 'P is the sister of Q. Q is the son of R. R is the wife of S. Which of the following is definitely false? I. S is the father of P. II. P is the son of R. III. Q is the brother of P. (a) Only I (b) Only II (c) Only III (d) Only II and III',
      'Q sister P ; R son Q ; S wife R ; claim 1 ; P father S ; claim 2 ; R son P ; claim 3 ; P brother Q', 'b'),

    # ---- batch 3: coded ----
    q('cd06', 'coded', 2, 'A @ B means A is the mother of B; A # B means A is the father of B; A % B means A is the daughter of B. If P @ Q # R % S, how is P related to S? (a) Mother (b) Mother-in-law (c) Grandmother (d) Sister',
      'Q mother P ; R father Q ; S daughter R ; ask S P', 'b'),
    q('cd07', 'coded', 3, "'A © B' means 'B is the son of A', 'A ∆ B' means 'B is the father of A' and 'A * B' means 'B is the wife of A'. If J ∆ K © L * M, how is M related to J? (a) Sister (b) Sister-in-law (c) Mother (d) Aunt",
      'J father K ; K son L ; L wife M ; ask J M', 'b'),
    q('cd08', 'coded', 2, 'If X ★ Y means X is the brother of Y, X ◆ Y means X is the father of Y and X ● Y means X is the wife of Y, what does A ◆ B ★ C ● D mean? (a) A is the father-in-law of D (b) A is the father of D (c) D is the son of A (d) A is the uncle of D',
      'B father A ; C brother B ; D wife C ; claim 1 ; D father-in-law A ; claim 2 ; D father A ; claim 3 ; A son D ; claim 4 ; D uncle A', 'a'),
    q('cd09', 'coded', 2, 'M % N means M is the sister of N; M & N means M is the son of N; M # N means M is the mother of N. How is Z related to X in X % Y # Z & W? (a) Son (b) Nephew (c) Brother (d) Cousin',
      'Y sister X ; Z mother Y ; W son Z ; ask X Z', 'b'),

    # ---- batch 3: in-laws ----
    q('il01', 'inlaw', 2, 'Seema and Reema are married to two brothers, Amit and Sumit respectively. How is Seema related to Reema? (a) Sister (b) Co-sister (c) Cousin (d) Aunt',
      'Seema spouse Amit ; Reema spouse Sumit ; Amit brother Sumit ; ask Reema Seema', 'b'),
    q('il02', 'inlaw', 2, "Rajesh is the husband of the sister of Kavita's husband. How is Rajesh related to Kavita? (a) Husband's brother (b) Husband's sister's husband (c) Cousin (d) Uncle",
      'Kavita husband #1 ; #1 sister #2 ; #2 husband Rajesh ; ask Kavita Rajesh', 'b'),
    q('il03', 'inlaw', 2, "Bobby is the brother of Anup's wife, and Chiku is Bobby's daughter. How is Chiku related to Anup? (a) Daughter (b) Niece (c) Cousin (d) Sister-in-law",
      'Anup wife #1 ; #1 brother Bobby ; Bobby daughter Chiku ; ask Anup Chiku', 'b'),
    q('il04', 'inlaw', 2, "Ramesh is the father-in-law of Mohit's sister, and Kartik is Ramesh's only son. How is Kartik related to Mohit? (a) Brother (b) Brother-in-law (c) Cousin (d) Uncle",
      'Mohit sister #1 ; #1 father-in-law Ramesh ; Ramesh only son Kartik ; ask Mohit Kartik', 'b'),
    q('il05', 'inlaw', 1, 'Pallavi is the daughter-in-law of Sudha, whose only daughter is Neha. How is Neha related to Pallavi? (a) Sister (b) Sister-in-law (c) Mother-in-law (d) Cousin',
      'Sudha daughter-in-law Pallavi ; Sudha only daughter Neha ; ask Pallavi Neha', 'b'),

    # ---- batch 3: great relations ----
    q('gr01', 'great', 1, 'Champa is the mother of Manohar. Manohar is the father of Lokesh. Lokesh is the father of Ishan. How is Champa related to Ishan? (a) Grandmother (b) Great-grandmother (c) Great-aunt (d) Great-great-grandmother',
      'Manohar mother Champa ; Lokesh father Manohar ; Ishan father Lokesh ; ask Ishan Champa', 'b'),
    q('gr02', 'great', 3, "Usha is the sister of Ishan's great-grandmother. How is Usha related to Ishan? (a) Great-aunt (b) Great-great-aunt (c) Great-grandmother (d) Aunt",
      'Ishan great-grandmother #1 ; #1 sister Usha ; ask Ishan Usha', 'b'),
    # ---- which expression ----
    q('we01', 'expression', 2, 'If A + B means A is the brother of B, A − B means A is the sister of B, A × B means A is the father of B and A ÷ B means A is the mother of B, which of the following means that P is the maternal uncle of Q? (a) P + M × Q (b) P + M ÷ Q (c) P − M ÷ Q (d) P × M + Q',
      'claim 1 ; Q maternal uncle P', 'b',
      reads={'a': 'M brother P ; Q father M', 'b': 'M brother P ; Q mother M', 'c': 'M sister P ; Q mother M', 'd': 'M father P ; Q brother M'}),
    q('we02', 'expression', 2, 'If X $ Y means X is the son of Y, X # Y means X is the wife of Y and X @ Y means X is the brother of Y, which of the following shows that K is the daughter-in-law of N? (a) K # L $ N (b) K @ L $ N (c) N # L $ K (d) K $ L # N',
      'claim 1 ; N daughter-in-law K', 'a',
      reads={'a': 'L wife K ; N son L', 'b': 'L brother K ; N son L', 'c': 'L wife N ; K son L', 'd': 'L son K ; N wife L'}),
    q('we03', 'expression', 3, "'P ★ Q' means 'Q is the mother of P', 'P ◆ Q' means 'Q is the father of P' and 'P ● Q' means 'Q is the sister of P'. Which of the following shows that T is the paternal grandmother of R? (a) R ◆ S ★ T (b) R ★ S ◆ T (c) R ◆ S ◆ T (d) R ● S ★ T",
      'claim 1 ; R paternal grandmother T', 'a',
      reads={'a': 'R father S ; S mother T', 'b': 'R mother S ; S father T', 'c': 'R father S ; S father T', 'd': 'R sister S ; S mother T'}),
    q('we04', 'expression', 3, 'If A + B means A is the father of B, A − B means A is the wife of B, A × B means A is the brother of B and A ÷ B means A is the daughter of B, which of the following means that J is the aunt of M? (a) J × K + M (b) J − K × L + M (c) J ÷ K × L + M (d) J + K ÷ M',
      'claim 1 ; M aunt J', 'b',
      reads={'a': 'K brother J ; M father K', 'b': 'K wife J ; L brother K ; M father L', 'c': 'K daughter J ; L brother K ; M father L', 'd': 'K father J ; M daughter K'}),
    # ---- batch 4: dialogue ----
    q('sp29', 'speech', 2, 'Pointing to a man, Gita said, "He is the son of my mother-in-law\'s only son." How is the man related to Gita? (a) Son (b) Nephew (c) Brother-in-law (d) Husband',
      '~man male ; Gita mother-in-law #1 ; #1 only son #2 ; #2 son ~man ; ask Gita ~man', 'a'),
    q('sp30', 'speech', 3, 'Pointing to an elderly woman, Mihir said, "She is the mother-in-law of my mother\'s only daughter-in-law, and I have no brothers." How is the elderly woman related to Mihir? (a) Grandmother (b) Mother (c) Mother-in-law (d) Aunt',
      '~woman female ; Mihir mother #1 ; #1 only daughter-in-law #2 ; #2 mother-in-law ~woman ; Mihir no-brother ; ask Mihir ~woman', 'b'),
    q('sp31', 'speech', 3, 'Introducing a woman, Bharat said, "Her mother\'s husband\'s only son\'s wife is my wife." How is the woman related to Bharat? (a) Sister (b) Wife (c) Sister-in-law (d) Daughter',
      '~woman female ; ~woman mother #1 ; #1 husband #2 ; #2 only son #3 ; #3 wife #4 ; Bharat wife #4 ; ask Bharat ~woman', 'a'),
    q('sp32', 'speech', 2, 'Ankita told Raj, "The only son of my maternal grandmother\'s only daughter is your father." How is Raj related to Ankita? (a) Nephew (b) Cousin (c) Son (d) Brother',
      'Ankita maternal grandmother #1 ; #1 only daughter #2 ; #2 only son #3 ; Raj father #3 ; ask Ankita Raj', 'a'),

    # ---- batch 4: named families ----
    q('pz11', 'puzzle', 2, 'In a family of six members, Ramesh, Suresh, Kamla, Vimla, Ajay and Neha, there are two married couples and three generations. Ramesh is the husband of Kamla. Suresh is the only son of Ramesh. Vimla is the daughter-in-law of Kamla. Ajay and Neha are the children of Suresh, and Neha is a girl. How is Ajay related to Ramesh? (a) Son (b) Grandson (c) Nephew (d) Brother',
      'members 6 ; couples 2 ; generations 3 ; Kamla husband Ramesh ; Ramesh only son Suresh ; Kamla daughter-in-law Vimla ; Suresh child Ajay ; Suresh child Neha ; Neha female ; ask Ramesh Ajay', 'b'),
    q('pz12', 'puzzle', 2, 'In a family of six members, Ramesh, Suresh, Kamla, Vimla, Ajay and Neha, there are two married couples and three generations. Ramesh is the husband of Kamla. Suresh is the only son of Ramesh. Vimla is the daughter-in-law of Kamla. Ajay and Neha are the children of Suresh, and Neha is a girl. How is Vimla related to Neha? (a) Aunt (b) Mother (c) Grandmother (d) Sister',
      'members 6 ; couples 2 ; generations 3 ; Kamla husband Ramesh ; Ramesh only son Suresh ; Kamla daughter-in-law Vimla ; Suresh child Ajay ; Suresh child Neha ; Neha female ; ask Neha Vimla', 'b'),
    q('pz13', 'puzzle', 1, 'A family has five members: Om, Prem, Rani, Sita and Tara. Om and Rani are a married couple and Prem is their son. Sita is married to Prem. Tara is the daughter of Sita. Who is the mother-in-law of Sita? (a) Om (b) Rani (c) Tara (d) Prem',
      'members 5 ; Om spouse Rani ; Om son Prem ; Rani son Prem ; Sita spouse Prem ; Sita daughter Tara ; who Sita mother-in-law', 'b'),

    # ---- batch 4: statements ----
    q('st16', 'statements', 2, 'A has three children: two sons and one daughter. B and C are the sons of A. D is the wife of C. E is the only daughter of A. How is E related to D? (a) Sister (b) Sister-in-law (c) Daughter (d) Aunt',
      'A has 3 child ; A has 2 son ; A has 1 daughter ; A son B ; A son C ; C wife D ; A only daughter E ; ask D E', 'b'),
    q('st17', 'statements', 2, 'B is not the brother of A, though both are children of C. A is a boy. How is B related to A? (a) Brother (b) Sister (c) Cousin (d) Cannot be determined',
      'A not brother B ; C child B ; C child A ; A male ; ask A B', 'b'),
    q('st18', 'statements', 3, 'X is the son of Y. Z is the daughter of W. Y is the brother of W. V is the wife of X. How is V related to Z? (a) Sister-in-law (b) Cousin\'s wife (c) Aunt (d) Niece',
      'Y son X ; W daughter Z ; W brother Y ; X wife V ; ask Z V', 'b'),
    q('st19', 'statements', 1, "Lalita is the daughter of Mohan. Nakul is the son of Lalita. Mohan's wife is Rukmini. How is Mohan related to Nakul? (a) Paternal grandfather (b) Maternal grandfather (c) Uncle (d) Father",
      'Mohan daughter Lalita ; Lalita son Nakul ; Mohan wife Rukmini ; ask Nakul Mohan', 'b'),
    q('st20', 'statements', 1, 'Sudha is the wife of Gopal. Hari is the son of Gopal. Kiran is the daughter of Hari. How is Sudha related to Kiran? (a) Maternal grandmother (b) Paternal grandmother (c) Mother (d) Aunt',
      'Gopal wife Sudha ; Gopal son Hari ; Hari daughter Kiran ; ask Kiran Sudha', 'b'),
    q('st21', 'statements', 2, 'Kamal is the brother of Nirmala. Nirmala is the mother of Ojas. Ojas is the father of Pari. How is Kamal related to Pari? (a) Uncle (b) Grandfather (c) Great-uncle (d) Cousin',
      'Nirmala brother Kamal ; Ojas mother Nirmala ; Pari father Ojas ; ask Pari Kamal', 'c'),
    q('st22', 'statements', 2, "Raghu and Sanjay are brothers. Tina is Raghu's daughter and Uday is Sanjay's son. Varun is Uday's son. How is Varun related to Tina? (a) Nephew (b) Cousin's son (c) Son (d) Cousin",
      'Raghu brother Sanjay ; Sanjay brother Raghu ; Raghu daughter Tina ; Sanjay son Uday ; Uday son Varun ; ask Tina Varun', 'b'),
    q('st23', 'statements', 3, "Arun and Bina are siblings. Chirag is Arun's son and Dolly is Bina's daughter. Esha is Chirag's daughter and Farhan is Dolly's son. How is Farhan related to Esha? (a) Cousin (b) Second cousin (c) Nephew (d) Brother",
      'Arun sibling Bina ; Arun son Chirag ; Bina daughter Dolly ; Chirag daughter Esha ; Dolly son Farhan ; ask Esha Farhan', 'b'),

    # ---- batch 4: coded ----
    q('cd10', 'coded', 3, 'A ! B means A is the son of B; A @ B means A is the sister of B; A # B means A is the mother of B; A $ B means A is the husband of B. If P ! Q @ R # S $ T, how is T related to Q? (a) Niece (b) Nephew\'s wife (c) Daughter-in-law (d) Sister-in-law',
      'Q son P ; R sister Q ; S mother R ; T husband S ; ask Q T', 'b'),
    q('cd11', 'coded', 3, "'M $ N' means 'N is the brother of M', 'M £ N' means 'N is the mother of M' and 'M € N' means 'N is the wife of M'. If K £ L $ J € H, how is H related to K? (a) Aunt (b) Mother (c) Grandmother (d) Sister-in-law",
      'K mother L ; L brother J ; J wife H ; ask K H', 'a'),
    q('cd12', 'coded', 2, 'If P # Q means P is the child of Q, P @ Q means P is the sibling of Q and P $ Q means P is the spouse of Q, how is A related to C in A # B $ C? (a) Son (b) Daughter (c) Child (d) Nephew',
      'B child A ; C spouse B ; ask C A', 'c'),

    # ---- batch 4: sufficiency ----
    q('ds06', 'sufficiency', 2, "How is Rohit related to Mohit? I. Rohit's father is the only brother of Mohit's father. II. Mohit's father has no sisters. (a) Only I is sufficient (b) Only II is sufficient (c) Either I or II is sufficient (d) Both I and II together are sufficient (e) Neither I nor II is sufficient",
      'ask Mohit Rohit ; part 1 ; Rohit father #1 ; Mohit father #2 ; #2 only brother #1 ; part 2 ; Mohit father #3 ; #3 no-sister', 'a'),
    q('ds07', 'sufficiency', 3, 'How is P related to T? I. P is the brother of Q, and Q is the daughter of R. II. R is the wife of S, and S is the son of T. (a) Only I is sufficient (b) Only II is sufficient (c) Either I or II is sufficient (d) Both I and II together are sufficient (e) Neither I nor II is sufficient',
      'ask T P ; part 1 ; Q brother P ; R daughter Q ; part 2 ; S wife R ; T son S', 'd'),

    # ---- batch 4: claims ----
    q('cl05', 'claims', 3, 'D is the brother of E. F is the father of D. G is the mother of F. H is the sister of G. Which of the following is definitely true? (a) H is the great-aunt of E (b) E is the brother of D (c) G is the maternal grandmother of E (d) F is the son of H',
      'E brother D ; D father F ; F mother G ; G sister H ; claim 1 ; E great-aunt H ; claim 2 ; D brother E ; claim 3 ; E maternal grandmother G ; claim 4 ; H son F', 'a'),

    # ---- batch 4: Hindi terms ----
    q('hi03', 'hindi', 1, "Kabir calls Asha his mausi. Asha's husband is Deepak. How is Deepak related to Kabir? (a) Uncle (b) Father (c) Grandfather (d) Brother",
      'Kabir maternal aunt Asha ; Asha husband Deepak ; ask Kabir Deepak', 'a'),
    q('hi04', 'hindi', 1, "Riya's dadi has two sons, Mohan and Sohan, and Mohan is Riya's father. How is Sohan related to Riya? (a) Maternal uncle (b) Paternal uncle (c) Grandfather (d) Cousin",
      'Riya paternal grandmother #1 ; #1 has 2 son ; #1 son Mohan ; #1 son Sohan ; Riya father Mohan ; ask Riya Sohan', 'b'),
    q('hi05', 'hindi', 2, "Vikram is Tanu's jija. How is Tanu related to Vikram's wife? (a) Sister (b) Cousin (c) Daughter (d) Niece",
      'Tanu sister #1 ; #1 husband Vikram ; Vikram wife #2 ; ask #2 Tanu', 'a'),

    # ---- batch 4: deep ----
    q('dp04', 'deep', 3, "Kunal is the son of the daughter of the son of the brother of Leena's mother. How is Kunal related to Leena? (a) Cousin's grandson (b) Nephew (c) Grandson (d) Cousin",
      'Leena mother #1 ; #1 brother #2 ; #2 son #3 ; #3 daughter #4 ; #4 son Kunal ; ask Leena Kunal', 'a'),
]


def split(item) -> str:
    """'test' (locked: only reported), 'dev' (may guide choices) or 'train'. Fixed by id."""
    import zlib
    h = zlib.crc32(('authored:' + item['id']).encode()) % 10
    return 'test' if h < 3 else 'dev' if h < 5 else 'train'


def examples(which=('train',)):
    """Reader examples {tokens, lf, kind} for the given splits (gold facts placed the way prepare() numbers names)."""
    from eval.exam_set import gold_facts
    from kinlogic.prepare import prepare
    out = []
    for it in A:
        if split(it) in which:
            p = prepare(it['text'])
            out.append({'tokens': p['tokens'], 'lf': gold_facts(it, p['ents']), 'kind': 'authored-' + it['kind']})
    return out
