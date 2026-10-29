"""Hand-written twisted questions for the gauntlet (evaluation only, never training): self-references through "only",
grandparents whose side is not given, long lines of descent, LogiQAte-style detours, gender-dependent twins, coded
symbols, Hindi kin words, sufficiency, claims and "cannot be determined". Written for this project, not copied from any
site; each has its exact facts and expected option. Check with:  python -m eval.twisted_set
"""
from eval.authored_set import q

T = [
    q('tw01', 'speech', 3, 'Pointing to a boy, Neeta said, "He is the son of the only sister of my only brother." How is the boy related to Neeta? (a) Nephew (b) Son (c) Cousin (d) Brother',
      '~boy male ; Neeta female ; Neeta only brother #1 ; #1 only sister #2 ; #2 son ~boy ; ask Neeta ~boy', 'b'),
    q('tw02', 'speech', 3, 'Pointing to a girl, Kunal said, "Her mother is the wife of the only son of my paternal grandfather." How is the girl related to Kunal? (a) Cousin (b) Sister (c) Niece (d) Daughter',
      '~girl female ; ~girl mother #1 ; Kunal paternal grandfather #2 ; #2 only son #3 ; #3 wife #1 ; ask Kunal ~girl', 'b'),
    q('tw03', 'generations', 3, 'A is the father of B. B is the mother of C. C is the father of D. D is the father of E. E is the mother of F. F is the father of G. How is A related to G? (a) Great-great-great-great-grandfather (b) Great-great-great-grandfather (c) Great-great-grandfather (d) Grandfather',
      'B father A ; C mother B ; D father C ; E father D ; F mother E ; G father F ; ask G A', 'a'),
    q('tw04', 'generations', 3, 'P is the grandfather of Q. Q is the grandmother of R. R is the grandfather of S. S is the father of T. T is the mother of U. How is U related to P? (a) Great-great-great-great-great-great-grandchild (b) Great-great-great-great-great-grandchild (c) Great-great-great-great-great-great-great-grandchild (d) Great-great-great-great-great-great-grandfather',
      'Q grandfather P ; R grandmother Q ; S grandfather R ; T father S ; U mother T ; ask P U', 'a'),
    q('tw05', 'statements', 3, 'P is the brother of Q. R is the father of P. S is the only daughter of R. How is Q related to S? (a) Sister (b) Brother (c) Cousin (d) Cannot be determined',
      'Q brother P ; P father R ; R only daughter S ; ask S Q', 'b'),
    q('tw06', 'statements', 2, 'M is the sibling of N. N is the son of O. How is M related to O? (a) Son (b) Daughter (c) Cannot be determined (d) Nephew',
      'N sibling M ; O son N ; ask O M', 'c'),
    q('tw07', 'statements', 2, 'Ravi is the son of Mohan. Sita is the daughter of Gopal. Mohan and Gopal are close friends. How is Ravi related to Sita? (a) Brother (b) Cousin (c) No relation (d) Brother-in-law',
      'Mohan son Ravi ; Gopal daughter Sita ; ask Sita Ravi', 'c'),
    q('tw08', 'coded', 2, 'A + B means A is the son of B; A − B means A is the wife of B; A × B means A is the brother of B; A ÷ B means A is the mother of B. If P × Q + R − S, how is P related to S? (a) Son (b) Brother (c) Nephew (d) Father',
      'Q brother P ; R son Q ; S wife R ; ask S P', 'a'),
    q('tw09', 'hindi', 3, 'Ramesh is the chacha of Suresh, and Suresh is the mama of Kiran. How is Ramesh related to Kiran? (a) Maternal grandfather (b) Great-uncle (c) Uncle (d) Cousin',
      'Suresh paternal uncle Ramesh ; Kiran maternal uncle Suresh ; ask Kiran Ramesh', 'b'),
    q('tw10', 'story', 3, "Last Diwali, Farah visited her maternal grandmother Zubeida in Lucknow with her school friend Neha. Zubeida's only son Imran runs a bakery there. Imran's wife Sana made sweets for everyone, while Neha's father Arif drove them around the city. Sana and Imran have one son, Kabir, who is younger than Farah. How is Kabir related to Farah? (a) Brother (b) Cousin (c) Nephew (d) Cannot be determined",
      'Farah female ; Farah maternal grandmother Zubeida ; Farah friend Neha ; Zubeida only son Imran ; Imran wife Sana ; Neha father Arif ; Sana son Kabir ; Imran son Kabir ; ask Farah Kabir', 'b'),
    q('tw11', 'sufficiency', 3, "How is J related to K? I. J is the only daughter of L, who is the mother-in-law of K's wife. II. K has one sister. (a) Only I is sufficient (b) Only II is sufficient (c) Either I or II is sufficient (d) Both I and II together are sufficient (e) Neither I nor II is sufficient",
      'ask K J ; part 1 ; L only daughter J ; K wife #1 ; #1 mother-in-law L ; part 2 ; K has 1 sister', 'a'),
    q('tw12', 'puzzle', 2, 'A family has six members L, M, N, O, P and Q. M is the son of L. N is the wife of M. O is the daughter of N. P is the husband of L. Q is the sister of M. How many female members are there in the family? (a) 2 (b) 3 (c) 4 (d) 5',
      'members 6 ; L son M ; M wife N ; N daughter O ; L husband P ; M sister Q ; count female', 'c'),
    q('tw13', 'claims', 3, 'X is the mother of Y. Y is the brother of Z. W is the daughter of Z. V is the husband of X. Which of the following is/are definitely true? I. V is the grandfather of W. II. Y is the paternal uncle of W. III. Z is the son of X. (a) Only I (b) Only II (c) Only I and III (d) All of these',
      'Y mother X ; Z brother Y ; Z daughter W ; X husband V ; claim 1 ; W grandfather V ; claim 2 ; W paternal uncle Y ; claim 3 ; X son Z', 'a'),
    q('tw14', 'detour', 3, 'Pointing to a woman, Vikram said, "She is the wife of the only son of the husband of my mother\'s mother-in-law." How is the woman related to Vikram? (a) Mother (b) Aunt (c) Grandmother (d) Sister-in-law',
      '~woman female ; Vikram mother #1 ; #1 mother-in-law #2 ; #2 husband #3 ; #3 only son #4 ; #4 wife ~woman ; ask Vikram ~woman', 'a'),
    q('tw15', 'detour', 3, 'Introducing Rohan, Priya said, "His father\'s only sister\'s only brother is my husband." How is Rohan related to Priya? (a) Son (b) Nephew (c) Brother-in-law (d) Cousin',
      "Rohan male ; Rohan father #1 ; #1 only sister #2 ; #2 only brother #3 ; Priya husband #3 ; ask Priya Rohan", 'a'),
    q('tw16', 'statements', 2, "Asha's brother is Deepak's son. How is Deepak related to Asha? (a) Father (b) Uncle (c) Brother (d) Grandfather",
      'Asha brother #1 ; Deepak son #1 ; ask Asha Deepak', 'a'),
    q('tw17', 'nested', 3, "Mohan is the son of the only sister of the father of Ravi's only brother. How is Mohan related to Ravi? (a) Brother (b) Cousin (c) Nephew (d) Uncle",
      'Ravi only brother #1 ; #1 father #2 ; #2 only sister #3 ; #3 son Mohan ; ask Ravi Mohan', 'b'),
    q('tw18', 'statements', 2, 'Ajay is the husband of Rani and Vijay is the husband of Mani. Rani and Mani are sisters. How is Vijay related to Ajay? (a) Brother-in-law (b) Co-brother (c) Cousin (d) Friend',
      'Ajay wife Rani ; Vijay wife Mani ; Rani sister Mani ; ask Ajay Vijay', 'b'),
    q('tw19', 'speech', 3, 'Showing a photograph of a man, Meera said, "This man\'s son\'s father is the son-in-law of my mother." Meera has no sister. How is the man related to Meera? (a) Brother (b) Husband (c) Father (d) Brother-in-law',
      '~man male ; ~man son #1 ; #1 father #2 ; Meera mother #3 ; #3 son-in-law #2 ; Meera no-sister ; ask Meera ~man', 'b'),
    q('tw20', 'speech', 3, 'Pointing to Sneha, Arjun said, "She is the daughter of the only child of my grandfather\'s only son." How is Sneha related to Arjun? (a) Daughter (b) Niece (c) Cannot be determined (d) Sister',
      'Sneha female ; Arjun grandfather #1 ; #1 only son #2 ; #2 only child #3 ; #3 daughter Sneha ; ask Arjun Sneha', 'a'),   # "grandfather's only son": father
    q('tw21', 'speech', 2, 'Rekha told Sunil that his mother is the only daughter of Rekha\'s paternal grandfather. How is Rekha related to Sunil? (a) Sister (b) Cousin (c) Niece (d) Aunt',
      'Sunil mother #1 ; Rekha paternal grandfather #2 ; #2 only daughter #1 ; ask Sunil Rekha', 'b'),
    q('tw22', 'nested', 3, "P is the son of the only daughter of Q's father. Q is a man. How is P related to Q? (a) Son (b) Nephew (c) Brother (d) Cousin",
      'Q male ; Q father #1 ; #1 only daughter #2 ; #2 son P ; ask Q P', 'b'),
    q('tw23', 'nested', 3, "P is the son of the only daughter of Q's father. Q is a woman. How is P related to Q? (a) Son (b) Nephew (c) Brother (d) Cousin",
      'Q female ; Q father #1 ; #1 only daughter #2 ; #2 son P ; ask Q P', 'a'),
    q('tw24', 'speech', 2, 'Pointing to a man, Lata said, "He is the father of my son\'s only sister." How is the man related to Lata? (a) Husband (b) Brother (c) Father (d) Father-in-law',
      '~man male ; Lata female ; Lata son #1 ; #1 only sister #2 ; #2 father ~man ; ask Lata ~man', 'a'),
    q('tw25', 'sufficiency', 2, 'How is T related to U? I. T is the brother of V. II. V is the daughter of U. (a) Only I is sufficient (b) Only II is sufficient (c) Either I or II is sufficient (d) Both I and II together are sufficient (e) Neither I nor II is sufficient',
      'ask U T ; part 1 ; V brother T ; part 2 ; U daughter V', 'd'),
    q('tw26', 'generations', 3, "Hari is Om's great-great-grandfather. Hari's only son is Dev. Dev's only son is Jay, and Jay is Om's grandfather. How is Dev related to Om? (a) Great-grandfather (b) Great-great-grandfather (c) Grandfather (d) Great-uncle",
      'Om great-great-grandfather Hari ; Hari only son Dev ; Dev only son Jay ; Om grandfather Jay ; ask Om Dev', 'a'),
    q('tw27', 'statements', 3, "Kiran's father's mother's only son's daughter is Pooja. Kiran is a boy. How is Pooja related to Kiran? (a) Cousin (b) Sister (c) Niece (d) Aunt",
      'Kiran male ; Kiran father #1 ; #1 mother #2 ; #2 only son #3 ; #3 daughter Pooja ; ask Kiran Pooja', 'b'),
    q('tw28', 'detour', 3, 'Pointing to an old man, Tanvi said, "His only son is the father-in-law of my husband\'s only sister." Tanvi has no brother-in-law. How is the old man related to Tanvi\'s husband\'s sister\'s husband? (a) Grandfather (b) Father (c) Uncle (d) Father-in-law',
      '~man male ; ~man only son #1 ; Tanvi husband #2 ; #2 only sister #3 ; #3 father-in-law #1 ; #3 husband #4 ; ask #4 ~man', 'a'),
]


if __name__ == '__main__':
    from eval.check_authored import check
    check(T)
