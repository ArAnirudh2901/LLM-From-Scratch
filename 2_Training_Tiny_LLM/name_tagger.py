from __future__ import annotations
import argparse, math, os, random, re, time
from collections import Counter

import torch
import torch.nn.functional as F

from model_gpt import GPT
from qa_data import hidden_states, load_splits, pad_batch
from train import get_device

# Finds person names in plain text (no [brackets]) with the same GPT class.
# Causal GPT sees only the left context, so the text is fed twice: "<bos> text <sep> text".
# Each word of the second copy can attend to the whole first copy and is tagged <N> (name) or <O>.
# Training names are replaced by random real or invented names, so the tagger learns from context
# ("X's daughter", "her brother X") and capitalization, not from memorized names.

WORD_RE = re.compile(r"'s|\w+(?:-\w+)*|[^\w\s]")
SPECIALS = ['<pad>', '<unk>', '<Unk>', '<bos>', '<sep>', '<N>', '<O>']
PLACES = ['{X} visited {P} last summer.', '{X} flew to {P} for work.', 'They all moved to {P}.',
          '{X} loves the food in {P}.', 'The family went to {P} on vacation.', '{X} grew up near {P}.',
          '{X} grew up in {P}.', '{X} is moving to {P} next month.', '{X} got a new job at {P}.',
          '{X} has worked for {P} for years.', '{X} ordered a sandwich from {P}.', '{X} graduated from {P} last spring.',
          '{X} read an article about {P}.', 'The trip to {P} was the highlight of the year for {X}.',
          '{X} shops at {P} every weekend.', '{X} spent a week in {P} with friends.', '{X} speaks fluent {P}.',
          '{X} loves celebrating {P} with the family.', 'On {P}, {X} went hiking.', '{X} drove from {P} to the coast.',
          '{X} bought a new phone from {P}.']
PLACE_PHRASES = ['in {P}', 'at {P}', 'near {P}', 'from {P}', 'to {P}', 'around {P}']   # attached to a story sentence
PERSON_CONTEXTS = ['{X} visited {Y} last summer.', '{X} had lunch with {Y}.', '{X} called {Y} on Sunday.',
                   '{X} sent a letter to {Y}.', '{X} works with {Y}.', 'Last week {X} met {Y} downtown.',
                   '{X} got a gift from {Y}.', '{X} read a story about {Y} to the kids.']
# Capitalized words that are not people (places, companies, schools, languages, holidays, months). In the tagger
# vocab, so a known one is never a name; REAL_PLACES below are left out to test unseen places.
GAZETTEER = '''France Germany Spain Italy Portugal Greece Turkey Egypt Nigeria Ghana Ethiopia Morocco Algeria China Korea
Vietnam Thailand Indonesia Malaysia Singapore Philippines Pakistan Bangladesh Nepal Iran Iraq Israel Russia Ukraine
Poland Sweden Norway Finland Denmark Iceland Ireland Scotland England Wales Belgium Netherlands Switzerland Austria
Hungary Romania Argentina Chile Colombia Venezuela Cuba Jamaica Australia Peru Africa Europe Antarctica
California Florida Ohio Michigan Oregon Nevada Arizona Utah Colorado Kansas Iowa Alabama Alaska Hawaii Maine Vermont
Kentucky Tennessee Louisiana Oklahoma Arkansas Missouri Minnesota Wisconsin Illinois Indiana Pennsylvania
Massachusetts Connecticut Delaware Maryland Idaho Montana Wyoming Nebraska Rome Milan Venice Naples Barcelona Seville
Lisbon Porto Athens Istanbul Moscow Warsaw Prague Budapest Amsterdam Brussels Geneva Zurich Munich Hamburg Frankfurt
Copenhagen Stockholm Helsinki Edinburgh Glasgow Manchester Liverpool Birmingham Leeds Bristol Beijing Shanghai Seoul
Bangkok Delhi Karachi Dhaka Jakarta Manila Dubai Riyadh Tehran Baghdad Nairobi Lagos Accra Johannesburg Montreal
Vancouver Ottawa Calgary Houston Dallas Phoenix Miami Atlanta Detroit Portland Philadelphia Pittsburgh
Baltimore Cleveland Nashville Memphis Orlando Tampa Honolulu Anchorage Melbourne Brisbane Perth Auckland Lima Bogota
Santiago Havana Quito Apple Facebook Instagram Twitter Tesla Intel Samsung Sony Honda Disney Pepsi Coca-Cola Costco
Kroger Safeway Spotify Uber Airbnb Adobe Oracle IBM Nvidia Adidas Gucci Prada Lego Nintendo Volkswagen BMW Audi
Ferrari Pixar Verizon FedEx Expedia Reddit YouTube Wikipedia PayPal Visa Mastercard Subway Chipotle Dunkin Panera
Ikea Zara Lufthansa Emirates Amazon Princeton Cornell Columbia Berkeley Cambridge Purdue Rutgers Georgetown
Vanderbilt Northwestern MIT Everest Disneyland Hollywood Broadway Manhattan Brooklyn Alcatraz Sahara Himalayas
Pacific Atlantic Mediterranean Caribbean French Spanish German Italian Chinese Japanese Korean Russian Arabic Hindi
English Mexican Canadian American British Irish Greek Christmas Easter Thanksgiving Halloween Hanukkah Diwali
Ramadan Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March July September October
November December'''.split()
SYL = ['ka', 'lo', 'mi', 'ra', 'zu', 'ne', 'ti', 'sha', 'vo', 'den', 'mar', 'lin', 'xi', 'qu', 'bel',
       'tor', 'yan', 'ese', 'ola', 'fen', 'gri', 'dha', 'wu', 'ji', 'nok', 'pre', 'sol', 'ul', 'ae', 'ry']


def invented_name(rng: random.Random) -> str:
    return ''.join(rng.choice(SYL) for _ in range(rng.randint(1, 4))).capitalize()


def plain(story: str, rename: dict) -> tuple[list[str], list[int]]:
    """Remove brackets, apply renaming; return words and 1/0 name labels."""
    text = re.sub(r'\[([^\]]+)\]', lambda g: f' \x00{rename.get(g.group(1), g.group(1))}\x00 ', story)
    words, labels = [], []
    for i, part in enumerate(text.split('\x00')):   # odd parts are names
        for w in WORD_RE.findall(part):
            words.append(w); labels.append(i % 2)
    return words, labels


class TagVocab:
    def __init__(self, itos):
        self.itos, self.stoi = itos, {t: i for i, t in enumerate(itos)}

    @classmethod
    def build(cls, stories: list[str], min_freq: int = 3, extra: list[str] = ()):
        c = Counter(w for s in stories for w in WORD_RE.findall(re.sub(r'\[[^\]]+\]', ' ', s)))
        words = {w for w, n in c.items() if n >= min_freq} | set(extra) | {w.lower() for w in extra}
        return cls(SPECIALS + sorted(words))

    def encode(self, words):
        return [self.stoi.get(w, self.stoi['<Unk>' if w[:1].isupper() else '<unk>']) for w in words]


def _fill(tpl: str, slots: dict) -> tuple[list[str], list[int]]:
    """Template words and labels; slots map 'X'/'Y'/'P' to (word, label)."""
    words, labels = [], []
    for part in re.split(r'\{([XYP])\}', tpl):   # odd parts are slot names
        if part in slots:
            words.append(slots[part][0]); labels.append(slots[part][1])
        else:
            ws = WORD_RE.findall(part)
            words += ws; labels += [0] * len(ws)
    return words, labels


def _insert(words, labels, new, rng):
    """Insert a sentence at a sentence boundary."""
    at = rng.choice([0] + [i + 1 for i, w in enumerate(words) if w in '.!?'])
    return words[:at] + new[0] + words[at:], labels[:at] + new[1] + labels[at:]


def _attach(words, labels, phrase: str, place: str, rng):
    """Put "in <place>" before a sentence's final punctuation, or "In <place> ," at its start."""
    ph = [place if w == '\x00' else w for w in phrase.split()]
    lab = [int(False)] * len(ph)
    ends = [i for i, w in enumerate(words) if w in '.!?']
    if not ends:
        return words, labels
    if rng.random() < 0.7:
        at = rng.choice(ends)
    else:
        at = rng.choice([0] + [i + 1 for i in ends[:-1]])
        ph, lab = [ph[0].capitalize()] + ph[1:] + [','], lab + [0]
    return words[:at] + ph + words[at:], labels[:at] + lab + labels[at:]


def make_example(vocab, story, rng, pool):
    names = list(dict.fromkeys(re.findall(r'\[([^\]]+)\]', story)))
    new = set()
    rename = {}
    for n in names:   # half real names from the pool, half invented; never reuse within a story
        while True:
            m = rng.choice(pool) if rng.random() < 0.5 else invented_name(rng)
            if m not in new:
                break
        new.add(m); rename[n] = m
    words, labels = plain(story, rename)
    if names:
        people_now = list(rename.values())
        for _ in range(rng.choice([0, 1, 1, 2, 2])):   # capitalized places/companies are not people
            # mostly invented (unknown words): unseen places must be recognized from context ("lives in", "works at")
            p = rng.choice(GAZETTEER) if rng.random() < 0.4 else \
                invented_name(rng) + rng.choice(['', '', 'ville', 'land', 'burg', 'ton', 'shire', 'port'])
            words, labels = _insert(words, labels, _fill(rng.choice(PLACES), {'X': (rng.choice(people_now), 1), 'P': (p, 0)}), rng)
        if rng.random() < 0.4:    # a place inside a family sentence: "... for his birthday in Zanzibar."
            p = rng.choice(GAZETTEER) if rng.random() < 0.4 else invented_name(rng)
            words, labels = _attach(words, labels, rng.choice(PLACE_PHRASES).replace('{P}', '\x00'), p, rng)
        if rng.random() < 0.25:   # people in place-like spots: the word, not the spot, decides
            y = rng.choice(pool) if rng.random() < 0.5 else invented_name(rng)
            words, labels = _insert(words, labels, _fill(rng.choice(PERSON_CONTEXTS), {'X': (rng.choice(people_now), 1), 'Y': (y, 1)}), rng)
    if rng.random() < 0.3:   # 1-3 sentences, like a typed question: copy 2 then starts at an early position
        ends = [i + 1 for i, w in enumerate(words) if w in '.!?']
        if len(ends) > 1:
            starts = [0] + ends[:-1]
            a = rng.randrange(len(ends))
            b = min(len(ends), a + rng.randint(1, 3))
            words, labels = words[starts[a]:ends[b - 1]], labels[starts[a]:ends[b - 1]]
    if rng.random() < 0.25:            # lowercase text: names must come from context alone
        words = [w.lower() for w in words]
    ids = vocab.encode(words)
    return [vocab.stoi['<bos>']] + ids + [vocab.stoi['<sep>']] + ids, labels


def batch(vocab, items, device):
    seqs = [s for s, _ in items]
    idx = pad_batch(seqs, vocab.stoi['<pad>'], device)
    tgt = torch.full_like(idx, -100)
    for i, (s, lab) in enumerate(items):
        n = len(lab)
        tgt[i, n + 2:2 * n + 2] = torch.tensor([vocab.stoi['<N>'] if l else vocab.stoi['<O>'] for l in lab])
    return idx, tgt


@torch.no_grad()
def find_names(model, vocab, text: str, device) -> list[str]:
    """Person names in plain text, in order of first appearance. Text longer than the model's window
    (it reads the text twice) is tagged in chunks of whole sentences."""
    words = WORD_RE.findall(text)
    limit = (model.block_size - 2) // 2
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if w in '.!?' and len(cur) > limit // 2:
            chunks.append(cur); cur = []
    chunks.append(cur)
    names = []
    for ws in (c[i:i + limit] for c in chunks for i in range(0, max(1, len(c)), limit)):
        ids = vocab.encode(ws)
        idx = torch.tensor([[vocab.stoi['<bos>']] + ids + [vocab.stoi['<sep>']] + ids], device=device)
        h = hidden_states(model, idx)[0, len(ids) + 2:]
        logits = model.head(h)[:, [vocab.stoi['<N>'], vocab.stoi['<O>']]]
        names += [w for w, t in zip(ws, logits.argmax(-1).tolist()) if t == 0]
    return list(dict.fromkeys(names))


def bracket_names(text: str, names: list[str]) -> str:
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(rf'(?<!\[)\b{re.escape(n)}\b(?!\])', f'[{n}]', text)
    return text


@torch.no_grad()
def evaluate(model, vocab, stories, device, seed=123, lower=False):
    """Token-level precision/recall/F1 on held-out stories whose names are all replaced by unseen invented names."""
    model.eval()
    rng = random.Random(seed)
    tp = fp = fn = 0
    for s in stories:
        names = list(dict.fromkeys(re.findall(r'\[([^\]]+)\]', s)))
        rename = {n: invented_name(rng) + 'x' for n in names}   # suffix: never a training name
        words, labels = plain(s, rename)
        if lower:
            words = [w.lower() for w in words]
        pred = set(find_names(model, vocab, ' '.join(words), device))
        for w, l in zip(words, labels):
            p = w in pred
            tp += p and l; fp += p and not l; fn += (not p) and l
    model.train()
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return prec, rec, 2 * prec * rec / max(1e-9, prec + rec)


# Real places and organizations (never in training) for the non-person check; no words that are also common first names.
REAL_PLACES = ['Paris', 'London', 'Tokyo', 'Berlin', 'Madrid', 'Chicago', 'Toronto', 'Sydney', 'Mumbai', 'Cairo',
               'Boston', 'Texas', 'Canada', 'India', 'Japan', 'Brazil', 'Mexico', 'Kenya', 'Seattle', 'Dublin', 'Vienna',
               'Oslo', 'Google', 'Microsoft', 'Walmart', 'Starbucks', 'Harvard', 'Stanford', 'Toyota', 'Nike',
               'Netflix', 'Boeing', 'Oxford', 'Yale']
NON_PERSON_TEMPLATES = ['{X} works at {P}.', '{X} bought a coffee at {P} this morning.', '{X} studied history at {P}.',
                        '{X} lives in {P} now.', '{X} took a photo in {P}.', '{X} applied for a job at {P}.',
                        '{X} watched a documentary about {P}.', 'Last year {X} sent a letter to {P}.']


@torch.no_grad()
def non_person_check(model, vocab, stories, device, seed=7, lower=False, short=False):
    """Held-out stories plus a real place/company (never in training), in an added sentence or inside a family
    sentence: share of those places tagged as a person (should be 0) and share of stories where every real
    person is still found (should be 1)."""
    model.eval()
    rng = random.Random(seed)
    place_hit = all_found = 0
    for s in stories:
        names = list(dict.fromkeys(re.findall(r'\[([^\]]+)\]', s)))
        place = rng.choice(REAL_PLACES)
        text = re.sub(r'\[([^\]]+)\]', r'\1', s)
        if rng.random() < 0.5:
            text += ' ' + rng.choice(NON_PERSON_TEMPLATES).format(X=rng.choice(names), P=place)
        else:   # inside a family sentence: "... asked her mom for help in Tokyo."
            sents = re.split(r'(?<=[.!?])\s+', text)
            j = rng.randrange(len(sents))
            if sents[j][-1:] in '.!?':
                sents[j] = sents[j][:-1] + ' ' + rng.choice(['in', 'at', 'from', 'near']) + ' ' + place + sents[j][-1]
            else:
                sents[j] += f' in {place}.'
            text = ' '.join(sents)
        if short:   # the sentence with the place, alone or followed by one more: like a typed question
            sents = re.split(r'(?<=[.!?])\s+', text)
            j = next(i for i, x in enumerate(sents) if place in x)
            text = ' '.join(sents[j:j + rng.randint(1, 2)])
            names = [n for n in names if n in text]
        found = {w.lower() for w in find_names(model, vocab, text.lower() if lower else text, device)}
        place_hit += place.lower() in found
        all_found += all(n.lower() in found for n in names)
    model.train()
    return place_hit / len(stories), all_found / len(stories)


def report(model, vocab, stories, device) -> tuple[float, str]:
    """Selection score (mean of F1 and 1 - place-as-person rate, normal and lowercase) and a summary line."""
    score, parts = 0.0, []
    for lower in (False, True):
        prec, rec, f1 = evaluate(model, vocab, stories, device, lower=lower)
        place, found = non_person_check(model, vocab, stories, device, lower=lower)
        score += (f1 + 1 - place) / 5
        parts.append(f"{'lowercase' if lower else 'normal'}: unseen names P {prec:.3f} R {rec:.3f} F1 {f1:.3f} | "
                     f"real place/company tagged as person {place:.3f} | all people still found {found:.3f}")
    place, found = non_person_check(model, vocab, stories, device, short=True)
    score += (1 - place) / 5
    parts.append(f"short texts: real place/company tagged as person {place:.3f} | all people still found {found:.3f}")
    return score, ' || '.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/name-tagger-v4/tagger.pt')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eval_only', action='store_true')   # evaluate the saved tagger at --out
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)
    device = get_device(args.cpu)
    train, dev, val, test = load_splits(other_tests=True)
    final = [e['story'] for e in val[:300] if not e['seen']] + [e['story'] for e in test[:300] if not e['seen']]
    if args.eval_only:
        ck = torch.load(args.out, map_location=device)
        model, vocab = GPT(**ck['config']).to(device), TagVocab(ck['vocab'])
        model.load_state_dict(ck['model'])
        print(report(model, vocab, final, device)[1], flush=True)
        return
    stories = [e['story'] for e in train]
    pool = sorted({n for s in stories for n in re.findall(r'\[([^\]]+)\]', s)})
    vocab = TagVocab.build(stories, extra=GAZETTEER)
    config = dict(vocab_size=len(vocab.itos), block_size=512, n_layer=4, n_head=4, n_embd=256, dropout=0.1)
    model = GPT(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"device {device} | stories {len(stories)} | name pool {len(pool)} | vocab {len(vocab.itos)} | "
          f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    held = [e['story'] for e in dev[:300] if not e['seen']]   # model selection; val/test for the final report
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best, t0 = -1.0, time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g['lr'] = lr
        items = [make_example(vocab, rng.choice(stories), rng, pool) for _ in range(args.batch_size)]
        idx, tgt = batch(vocab, items, device)
        h = hidden_states(model, idx)
        m = tgt != -100
        loss = F.cross_entropy(model.head(h[m]), tgt[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == 'mps' and step % 50 == 0:
            torch.mps.empty_cache()
        if step % 500 == 0 or step == args.steps:
            score, line = report(model, vocab, held, device)
            print(f"step {step} | loss {loss.item():.4f} | dev {line} | {time.time() - t0:.0f}s", flush=True)
            if score > best:
                best = score
                torch.save({'model': model.state_dict(), 'config': config, 'vocab': vocab.itos, 'step': step}, args.out)
    model.load_state_dict(torch.load(args.out, map_location=device)['model'])
    print('[final] val+test ' + report(model, vocab, final, device)[1], flush=True)

if __name__ == '__main__':
    main()
