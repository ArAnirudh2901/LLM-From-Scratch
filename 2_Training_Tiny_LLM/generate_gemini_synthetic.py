"""Generate a resumable, balanced pool of Gemini-reworded kinship questions.

This is a candidate-pool builder, not a trainer. It draws only from verified
synthetic train rows in dataset_v6 (never test/real questions), targets measured
mix gaps first, and accepts rows only through kinlogic.reword's independent
fact extraction, semantic-equivalence, and solver checks.

Example (source the one authorized key without echoing it):
  set -a; source ~/.config/kinthread/gemini.env; set +a
  REWORD_GEMINI_MODEL=gemini-3.5-flash-lite \
  REWORD_EXTRACT_MODEL=gemini-3.5-flash-lite \
  python generate_gemini_synthetic.py --max-tried 5000 --request-interval 8 --pause 0

The script stops at a daily quota limit, checkpoints every successful batch,
and can be rerun later with the same output/state paths. It never rotates keys.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error

from kinlogic import reword as reword_module
from kinlogic.reword import SKIP, STYLES, reword_batch
from kinlogic.facts import parse
from kinlogic.kin import KIN
from phrase_audit import PAT
from kinlogic.names import INDIAN_F, INDIAN_M, WORLD_F, WORLD_M, LETTERS


# These are bins used by build_mix.py: zero-based bin 4 means six people,
# and bin 5 means seven people (the manifest stores bin indices, not counts).
PEOPLE_EDGES = [2, 3, 4, 5, 6, 7, 8, 10, 13, 99]
LENGTH_EDGES = [30, 45, 60, 80, 100, 130, 170, 999]
FOCUS_TAGS = [
    "focus:if", "focus:married", "focus:people-bin-4", "focus:people-bin-5",
    "focus:length-bin-4", "focus:length-bin-5",
]
NAME_STYLE_TAGS = ["name:indian", "name:world", "name:mixed"]
# Previous audited batches show materially better semantic yield from these kinds.
# Do not spend a limited free-tier quota on kinds whose rewrites almost never pass.
KIND_PRIORITY = ["deep-story", "statements", "story", "deep-nested", "deep-statements"]
REWORDABLE_KINDS = set(KIND_PRIORITY)
MAX_RESERVOIR = {
    "focus:": 500,
    "kind:": 300,
    "name:": 400,
    "answer:": 80,
    "all": 5000,
}
SAFE_STYLES = [
    STYLES[0], STYLES[1], STYLES[2], STYLES[4], STYLES[5], STYLES[8],
]

FOCUS_STYLES = {
    "focus:if": [
        "use a clearly conditional exam form that still contains an explicit If-clause; keep all facts and the final question unchanged in meaning",
        "make it a concise conditional aptitude-test question, retaining the explicit If-clause and every fact",
        "write it as a natural short story framed by an explicit If-condition, with the same facts and question",
    ],
    "focus:married": [
        "keep the literal phrase 'married to' for the spouse relationship and vary the surrounding sentence structure",
        "write naturally in exam prose while retaining the literal wording 'married to' for that relationship",
    ],
    "focus:people-bin-4": [
        "write a clear, moderately detailed exam question, retaining all named people and every fact",
        "use a natural multi-sentence family puzzle with all named people and facts preserved",
    ],
    "focus:people-bin-5": [
        "write a clear, detailed exam question, retaining all named people and every fact",
        "use a natural multi-sentence family puzzle with all named people and facts preserved",
    ],
    "focus:length-bin-4": [
        "use a fuller but concise exam-story style while preserving every name, fact, and the question",
        "vary the sentence structure into a natural 80-to-100-token-style exam prompt without adding facts",
    ],
    "focus:length-bin-5": [
        "use a fuller exam-story style while preserving every name, fact, and the question",
        "vary the sentence structure into a natural 101-to-130-token-style exam prompt without adding facts",
    ],
}

INDIAN = {x.lower() for x in INDIAN_M + INDIAN_F}
WORLD = {x.lower() for x in WORLD_M + WORLD_F}
LETTER_SET = {x.lower() for x in LETTERS}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
REMOVALS = {"once": 1, "twice": 2, "thrice": 3}


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def bin_of(value: int, edges: list[int]) -> int:
    return next(i for i, edge in enumerate(edges) if value <= edge)


def name_profile(row: dict) -> str:
    names = [str(n).strip().lower() for n in (row.get("names") or {}).values()]
    names = [n for n in names if n]
    if not names:
        return "unknown"
    if all(n in LETTER_SET for n in names):
        return "letters"
    has_indian = any(n in INDIAN for n in names)
    has_world = any(n in WORLD for n in names)
    if has_indian and has_world:
        return "mixed"
    if has_indian:
        return "indian"
    if has_world:
        return "world"
    return "unknown"


def tags_for(row: dict) -> set[str]:
    tags = {"all", f"kind:{row.get('kind', 'unknown')}", f"answer:{row.get('truth', 'unknown')}",
            f"name:{name_profile(row)}"}
    text = " ".join(row.get("tokens") or [])
    if PAT["if-conditional (if X is ... , how)"].search(text):
        tags.add("focus:if")
    if PAT["married to"].search(text):
        tags.add("focus:married")
    n_people = len(set(re.findall(r"[@#]\d+", row.get("lf", ""))))
    pbin = bin_of(n_people, PEOPLE_EDGES)
    if pbin in (4, 5):
        tags.add(f"focus:people-bin-{pbin}")
    lbin = bin_of(len(row.get("tokens") or []), LENGTH_EDGES)
    if lbin in (4, 5):
        tags.add(f"focus:length-bin-{lbin}")
    return tags


def supported_source(row: dict) -> bool:
    """Only reword fact graphs the current independent checker can compare completely."""
    if row.get("kind") not in REWORDABLE_KINDS:
        return False
    truth = row.get("truth", "").lower()
    # Extremely distant relationships are synthetic edge cases and have shown
    # much lower rewrite validity; use quota on relations likely in exam questions.
    if len(re.findall(r"\bgreat-", truth)) > 1:
        return False
    cousin_degree = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)-cousin\b", truth)
    if cousin_degree and ORDINALS[cousin_degree.group(1)] > 3:
        return False
    removed = re.search(r"\b(\d+|once|twice|thrice) times? removed\b|\b(once|twice|thrice) removed\b", truth)
    if removed:
        amount = next((g for g in removed.groups() if g), "0")
        distance = int(amount) if amount.isdigit() else REMOVALS.get(amount, 99)
        if distance > 2:
            return False
    person_count = len(set(re.findall(r"[@#]\d+", row.get("lf", ""))))
    if person_count > 8:
        return False
    # The current extraction/entailment checker models positive kin facts and gender.
    # Leave constraint-bearing questions to the exact programmatic generator; a
    # same-answer paraphrase alone cannot prove that 'only', counts, or negation survived.
    if re.search(r"\b(?:only|not|no|cannot|can't|neither|either|exactly|both|at\s+least|at\s+most)\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:sons?|daughters?|children|siblings?|brothers?|sisters?)\b",
                 row.get("text", ""), re.I):
        return False
    # These prompts should produce a clean standalone question. Skip MCQ
    # stems rather than asking the rewriter to silently discard answer choices.
    if len(re.findall(r"(?i)(?:\([1-5]\)|\b[A-E][.)])\s+", row.get("text", ""))) >= 2:
        return False
    facts, bad = parse(row["lf"])
    if bad or not facts:
        return False
    rels = 0
    for fact in facts:
        if fact.get("t") == "ask":
            continue
        if fact.get("t") == "rel":
            rels += 1
            if fact.get("neg") or fact.get("only") or fact.get("part") or fact.get("claim"):
                return False
            if fact.get("r") not in KIN:
                return False
        elif fact.get("t") != "gender":
            return False
    return rels > 0 and len(facts) <= 12


def verified_reference(row: dict) -> str | None:
    """Render the source LF as an unambiguous, name-bound fact list for the rewriter."""
    facts, bad = parse(row["lf"])
    names = row.get("names") or {}
    if bad or not facts:
        return None
    def label(entity: str | None) -> str | None:
        if entity in names:
            return names[entity]
        if isinstance(entity, str) and re.fullmatch(r"#\d+", entity):
            return f"unnamed person {entity}"
        return None
    lines = []
    for fact in facts:
        typ = fact.get("t")
        if typ == "rel":
            x, y = label(fact.get("x")), label(fact.get("y"))
            rel = fact.get("r")
            side = fact.get("side")
            if not x or not y or not rel:
                return None
            qualifier = f"{side} " if side else ""
            # In LF, `x relation y` means that y is x's relation. Spell that
            # direction out so the model cannot silently reverse parent/spouse facts.
            lines.append(f"{x}'s {qualifier}{rel} is {y}.")
        elif typ == "gender":
            x = label(fact.get("x"))
            gender = {"M": "male", "F": "female"}.get(fact.get("g"))
            if not x or not gender:
                return None
            lines.append(f"{x} is {gender}.")
        elif typ == "ask":
            x, y = label(fact.get("x")), label(fact.get("y"))
            if not x or not y:
                return None
            lines.append(f"Question pair: how is {y} related to {x}?")
        else:
            return None
    return "\n".join(lines)


def source_ready(row: dict) -> bool:
    """Avoid spending quota when a seed is inconsistent or cannot be name-bound."""
    return reword_module.accept(row["lf"], row["truth"], row["tokens"]) and verified_reference(row) is not None


def source_hashes(paths: list[str]) -> set[str]:
    done = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if row.get("source"):
                            done.add(row["source"])
                    except (ValueError, TypeError):
                        continue
        except (EOFError, OSError):
            continue
    return done


def reservoir_add(bucket: list, seen: dict, key: str, row: dict, cap: int, rng: random.Random) -> None:
    seen[key] += 1
    if len(bucket) < cap:
        bucket.append(row)
    else:
        j = rng.randrange(seen[key])
        if j < cap:
            bucket[j] = row


def collect_candidates(root: str, blocked: set[str], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    seen = collections.Counter()
    files = sorted(glob.glob(f"{root}/kinds/*.jsonl.gz")) + sorted(glob.glob(f"{root}/deep/**/*.jsonl.gz", recursive=True))
    files = [p for p in files if "/test/" not in p]
    if not files:
        raise FileNotFoundError(f"no synthetic train files found under {root}")
    n = 0
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (not isinstance(row.get("truth"), str) or not row.get("names") or
                        row.get("kind") in SKIP or not row.get("tokens") or not row.get("lf")):
                    continue
                if not supported_source(row):
                    continue
                if name_profile(row) == "letters":
                    continue
                src = sha1(row.get("text", ""))
                if not row.get("text") or src in blocked or "'s " in row["truth"]:
                    continue
                row = {k: row.get(k) for k in ("kind", "text", "tokens", "lf", "truth", "depth", "names")}
                row["_source"] = src
                tags = tags_for(row)
                for tag in tags:
                    cap = next((v for prefix, v in MAX_RESERVOIR.items() if tag.startswith(prefix)), MAX_RESERVOIR["all"])
                    reservoir_add(buckets[tag], seen, tag, row, cap, rng)
                n += 1
    # One row can be present in several buckets; this is intentional for set-cover sampling.
    print(f"scanned {len(files)} training files; {n:,} eligible source rows; {len(buckets):,} balanced strata", flush=True)
    return buckets


def make_queue(rows: list[dict], rng: random.Random) -> collections.deque:
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        key = (row.get("kind"), row.get("truth"), name_profile(row))
        groups[key].append(row)
    keys = list(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    out = collections.deque()
    while keys:
        next_keys = []
        for key in keys:
            if groups[key]:
                out.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return out


def select_batch(order: list[str], queues: dict[str, collections.deque], attempted: set[str], batch_size: int):
    for tag in order:
        q = queues.get(tag)
        if not q:
            continue
        batch = []
        while q and len(batch) < batch_size:
            row = q.popleft()
            if row["_source"] not in attempted:
                row["_selected_focus"] = tag
                batch.append(row)
        if batch:
            if len(batch) < batch_size and tag != "all":
                selected = {row["_source"] for row in batch}
                all_queue = queues.get("all")
                extra = []
                while all_queue and len(batch) + len(extra) < batch_size:
                    row = all_queue.popleft()
                    if row["_source"] not in attempted and row["_source"] not in selected:
                        row["_selected_focus"] = "all"
                        selected.add(row["_source"])
                        extra.append(row)
                if extra:
                    batch.extend(extra)
                    return "all", batch
            return tag, batch
    return None, []


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"attempted": [], "batches": 0, "last_status": "new"}
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("attempted", [])
    state.setdefault("batches", 0)
    return state


def save_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def append_gzip(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        for row in rows:
            row.pop("_source", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_audit(path: str, source_rows: list[dict], results: list[dict | None], focus: str,
                 reasons: list[str] | None = None, parse_error: str | None = None,
                 api_calls: int = 0, batch_id: int | None = None,
                 rewrite_temperature: float = 0.2,
                 rewrite_texts: list[str] | None = None,
                 extracted_lfs: list[str | None] | None = None,
                 extracted_readings: list[dict | None] | None = None) -> None:
    if not source_rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for i, (source, row) in enumerate(zip(source_rows, results)):
            kept = bool(row)
            row_focus = source.get("_selected_focus", focus)
            focus_kept = target_retained(row, row_focus) if row else False
            f.write(json.dumps({"source": source.get("_source"), "focus": row_focus, "batch_focus": focus,
                                "input_kind": source.get("kind"), "input_truth": source.get("truth"),
                                "input_name_profile": name_profile(source),
                                "batch_id": batch_id,
                                "batch_api_calls": api_calls,
                                "rewrite_temperature": rewrite_temperature,
                                "thinking_level": "low",
                                "rejection_reason": (reasons[i] if reasons and i < len(reasons) else None),
                                "extract_parse_error": parse_error,
                                "rewrite_text": (rewrite_texts[i] if rewrite_texts and i < len(rewrite_texts) else None),
                                "extracted_lf": (extracted_lfs[i] if extracted_lfs and i < len(extracted_lfs) else None),
                                "extracted_reading": (extracted_readings[i] if extracted_readings and i < len(extracted_readings) else None),
                                "accepted": kept, "focus_retained": focus_kept,
                                "output_kind": row.get("kind") if row else None,
                                "output_truth": row.get("truth") if row else None,
                                "output_text": row.get("text") if row else None}, ensure_ascii=False) + "\n")


def target_retained(row: dict, focus: str) -> bool:
    tags = tags_for(row)
    if focus == "all":
        return any(tag in tags for tag in FOCUS_TAGS)
    return focus in FOCUS_TAGS and focus in tags


def error_info(exc: urllib.error.HTTPError) -> tuple[str, bool, int | None]:
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        body = ""
    # Keep output safe and compact: no key material, raw prompt, or full server body.
    low = body.lower()
    daily = any(x in low for x in ("perday", "per_day", "requests_per_day", "daily quota",
                                   "generaterequestsperday", "limit: 0")) or bool(
        re.search(r"quota.?metric.{0,40}per.?day", low))
    retry = exc.headers.get("Retry-After") if exc.headers else None
    body_retry = None
    try:
        payload = json.loads(body)
        details = payload.get("error", {}).get("details", [])
        for detail in details if isinstance(details, list) else []:
            delay = detail.get("retryDelay") or detail.get("retry_delay")
            if delay:
                match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s\s*", str(delay), re.I)
                if match:
                    body_retry = float(match.group(1))
                    break
    except (ValueError, TypeError, AttributeError):
        pass
    try:
        wait = max(1, min(3600, int(float(retry)))) if retry else None
    except (ValueError, TypeError):
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone
            retry_at = parsedate_to_datetime(str(retry))
            wait = max(1, min(3600, int((retry_at - datetime.now(timezone.utc)).total_seconds())))
        except (TypeError, ValueError, OverflowError):
            wait = None
    if wait is None and body_retry is not None:
        wait = max(1, min(3600, int(body_retry + 0.999)))
    reason = "daily quota reached" if daily else "rate limit or temporary service limit"
    return reason, daily, wait


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/dataset_v6")
    ap.add_argument("--out", default="data/gemini_synthetic_v1.jsonl.gz")
    ap.add_argument("--focused-out", default="data/gemini_synthetic_v1.focus.jsonl.gz",
                    help="only validated rows that still satisfy the chosen underrepresented feature")
    ap.add_argument("--audit", default="data/gemini_synthetic_v1.audit.jsonl")
    ap.add_argument("--state", default="data/gemini_synthetic_v1.state.json")
    ap.add_argument("--max-tried", type=int, default=5000, help="maximum source questions to attempt; stops sooner on API quota")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--request-interval", type=float, default=8.0,
                    help="minimum seconds between Gemini API requests; starts conservatively for free-tier stability")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="optional extra pause after a completed batch (per-request pacing is separate)")
    ap.add_argument("--seed", type=int, default=8417)
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        ap.error("GEMINI_API_KEY is not set; load only the authorized gemini.env first")
    if args.batch < 1 or args.batch > 20:
        ap.error("batch must be between 1 and 20; larger batches are not validated for this workflow")
    if args.request_interval < 4 or args.request_interval > 300:
        ap.error("request-interval must be between 4 and 300 seconds")
    os.environ["REWORD_GEMINI_MIN_INTERVAL"] = str(args.request_interval)

    state = load_state(args.state)
    state["request_interval_seconds"] = args.request_interval
    done = source_hashes([args.out])
    protected = source_hashes(["data/reworded.jsonl.gz", "data/reworded_dev.jsonl.gz",
                               "data/dataset_v6/reworded.jsonl.gz"] +
                              glob.glob("data/reworded_parts/**/*.jsonl.gz", recursive=True))
    attempted = set(state["attempted"]) | done | protected
    buckets = collect_candidates(args.src, attempted, args.seed)
    rng = random.Random(args.seed)
    queues = {tag: make_queue(rows, rng) for tag, rows in buckets.items()}

    kinds = []
    for k in KIND_PRIORITY:
        tag = f"kind:{k}"
        if queues.get(tag):
            kinds.append(tag)
    kinds += sorted(tag for tag in queues if tag.startswith("kind:") and tag not in kinds)
    profiles = [tag for tag in NAME_STYLE_TAGS if queues.get(tag)]
    answers = sorted(tag for tag in queues if tag.startswith("answer:") and queues[tag])
    focus_order = [tag for tag in FOCUS_TAGS if queues.get(tag)]
    broad_order = kinds + profiles + answers + (["all"] if queues.get("all") else [])
    # Use a fixed, repeating coverage schedule. Rare measured gaps get regular
    # attempts, interleaved with kind/name/relation strata for breadth.
    schedule = []
    for idx in range(max(len(focus_order), len(kinds), len(profiles), len(answers), 1)):
        if focus_order:
            schedule.append(focus_order[idx % len(focus_order)])
        if kinds:
            schedule.append(kinds[idx % len(kinds)])
        if profiles:
            schedule.append(profiles[idx % len(profiles)])
        if answers:
            schedule.append(answers[idx % len(answers)])
    schedule = [x for x in schedule if x]
    if not schedule:
        print("no eligible unseen source questions remain", flush=True)
        return 0

    tried_this_run = 0
    accepted_this_run = 0
    attempts = set(attempted)
    target_retained_count = collections.Counter()
    print(f"using one configured Gemini key; {len(attempted):,} source rows already completed/protected; "
          f"batch={args.batch}, minimum request interval={args.request_interval:g}s, "
          f"max new attempts={args.max_tried:,}", flush=True)
    exhausted = False
    deferred_tail = False
    while tried_this_run < args.max_tried:
        remaining = args.max_tried - tried_this_run
        if tried_this_run and 0 < remaining < min(args.batch, 5):
            deferred_tail = True
            break
        # Rotate schedule start across resumed runs and checkpoint the cursor.
        start = int(state.get("batches", 0)) % len(schedule)
        order = schedule[start:] + schedule[:start]
        focus, batch = select_batch(order, queues, attempts, min(args.batch, remaining))
        if not batch:
            print("all sampled strata exhausted", flush=True)
            exhausted = True
            break
        desired_batch_size = len(batch)
        locally_invalid = [row for row in batch if not source_ready(row)]
        if locally_invalid:
            invalid_hashes = {row["_source"] for row in locally_invalid}
            attempts.update(invalid_hashes)
            batch = [row for row in batch if row["_source"] not in invalid_hashes]
            selected = {row["_source"] for row in batch}
            valid_before_fill = len(batch)
            all_queue = queues.get("all")
            while all_queue and len(batch) < desired_batch_size:
                row = all_queue.popleft()
                if row["_source"] in attempts or row["_source"] in selected:
                    continue
                if not source_ready(row):
                    attempts.add(row["_source"])
                    invalid_hashes.add(row["_source"])
                    continue
                row["_selected_focus"] = "all"
                selected.add(row["_source"])
                batch.append(row)
            attempts.update(invalid_hashes)
            state["attempted"] = sorted(attempts - done - protected)
            state["source_precheck_skipped_total"] = int(state.get("source_precheck_skipped_total", 0)) + len(invalid_hashes)
            save_state(args.state, state)
            print(f"local solver precheck skipped {len(invalid_hashes)} source row(s) before Gemini; "
                  "they were inconsistent or could not be fully name-bound", flush=True)
            if len(batch) > valid_before_fill or len(batch) != desired_batch_size:
                focus = "all"
        if not batch:
            continue
        if focus in FOCUS_STYLES:
            style = rng.choice(FOCUS_STYLES[focus])
        elif focus == "kind:speech":
            style = "rewrite this as a pointing or spoken exam question, preserving the exact speaker, addressee, pronoun referents, and family facts"
        elif focus == "kind:introduce":
            style = "keep the introduction frame and preserve who introduced whom, every descriptor, and every family fact"
        else:
            style = rng.choice(SAFE_STYLES)
        style += "; preserve the direction of every relation and every unnamed person's descriptor head noun (such as man, woman, lady, or gentleman); do not add, omit, or swap people"
        references = [verified_reference(row) for row in batch]
        diagnostics = {}
        request_start = reword_module.GEMINI_API_REQUESTS
        api_calls_before_retry = 0
        try:
            results = reword_batch(batch, "gemini", style=style, temperature=0.2,
                                   stats=diagnostics, references=references)
        except urllib.error.HTTPError as exc:
            reason, daily, retry_after = error_info(exc)
            calls_made = reword_module.GEMINI_API_REQUESTS - request_start
            api_calls_before_retry = calls_made
            state["api_requests_tracked"] = int(state.get("api_requests_tracked", 0)) + calls_made
            if exc.code in (429, 503) and not daily:
                wait = retry_after or (60 if exc.code == 429 else 30)
                print(f"Gemini returned HTTP {exc.code} ({reason}); honoring backoff for {wait}s, then retrying once", flush=True)
                time.sleep(wait)
                request_start = reword_module.GEMINI_API_REQUESTS
                try:
                    results = reword_batch(batch, "gemini", style=style, temperature=0.2,
                                           stats=diagnostics, references=references)
                except urllib.error.HTTPError as retry_exc:
                    reason2, daily2, _ = error_info(retry_exc)
                    state["api_requests_tracked"] = int(state.get("api_requests_tracked", 0)) + (
                        reword_module.GEMINI_API_REQUESTS - request_start)
                    state["last_status"] = "daily quota reached" if daily2 else "rate limit remained after one retry"
                    state["last_error_code"] = retry_exc.code
                    save_state(args.state, state)
                    print(f"stopped safely after retry: HTTP {retry_exc.code} ({reason2}); output and prior attempts are checkpointed", flush=True)
                    break
                except (TimeoutError, OSError, RuntimeError) as retry_exc:
                    state["api_requests_tracked"] = int(state.get("api_requests_tracked", 0)) + (
                        reword_module.GEMINI_API_REQUESTS - request_start)
                    state["last_status"] = f"temporary retry {type(retry_exc).__name__}"
                    save_state(args.state, state)
                    print(f"stopped safely after retry on temporary {type(retry_exc).__name__}; output is checkpointed", flush=True)
                    break
            else:
                state["last_status"] = reason if exc.code == 429 else f"API HTTP {exc.code}"
                state["last_error_code"] = exc.code
                save_state(args.state, state)
                print(f"stopped safely: HTTP {exc.code} ({reason}); output and prior attempts are checkpointed", flush=True)
                break
        except (TimeoutError, OSError, RuntimeError) as exc:
            state["api_requests_tracked"] = int(state.get("api_requests_tracked", 0)) + (
                reword_module.GEMINI_API_REQUESTS - request_start)
            state["last_status"] = f"temporary {type(exc).__name__}"
            save_state(args.state, state)
            print(f"stopped safely on temporary {type(exc).__name__}; rerun to resume without losing output", flush=True)
            break
        except KeyboardInterrupt:
            state["api_requests_tracked"] = int(state.get("api_requests_tracked", 0)) + (
                reword_module.GEMINI_API_REQUESTS - request_start)
            state["last_status"] = "interrupted; current batch was not committed"
            save_state(args.state, state)
            print("interrupted safely; completed batches are saved and this in-flight batch can be retried", flush=True)
            return 130

        saved = []
        focused = []
        focus_kept = 0
        for source, row in zip(batch, results):
            if row:
                row["_source"] = source["_source"]
                if target_retained(row, source.get("_selected_focus", focus)):
                    focus_kept += 1
                    focused.append(dict(row))
                saved.append(row)
        reasons = diagnostics.get("reasons", [])
        api_calls = api_calls_before_retry + reword_module.GEMINI_API_REQUESTS - request_start
        if api_calls_before_retry:
            state["api_requests_tracked"] = int(state.get("api_requests_tracked", 0)) + (
                reword_module.GEMINI_API_REQUESTS - request_start)
        else:
            state["api_requests_tracked"] = int(state.get("api_requests_tracked", 0)) + api_calls
        batch_id = int(state.get("batches", 0)) + 1
        append_audit(args.audit, batch, results, focus, reasons, diagnostics.get("extract_parse_error"),
                     api_calls=api_calls, batch_id=batch_id,
                     rewrite_texts=diagnostics.get("rewrite_texts"),
                     extracted_lfs=diagnostics.get("extracted_lfs"),
                     extracted_readings=diagnostics.get("extracted_readings"))
        append_gzip(args.out, saved)
        append_gzip(args.focused_out, focused)
        attempts.update(row["_source"] for row in batch)
        tried_this_run += len(batch)
        accepted_this_run += len(saved)
        state["attempted"] = sorted(attempts - done - protected)
        state["batches"] = int(state.get("batches", 0)) + 1
        state["last_status"] = "running"
        state["last_focus"] = focus
        state["tried_total"] = int(state.get("tried_total", 0)) + len(batch)
        state["accepted_total"] = int(state.get("accepted_total", 0)) + len(saved)
        save_state(args.state, state)
        target_retained_count[focus] += focus_kept
        print(f"{tried_this_run:,} tried this run; {accepted_this_run:,} validated rewrites saved "
              f"({len(saved)}/{len(batch)} this batch, focus={focus}, retained={focus_kept}, "
              f"API calls={api_calls}, tracked API calls={state['api_requests_tracked']}, "
              f"rejections={dict(collections.Counter(reasons))}, "
              f"extract_parse={diagnostics.get('extract_parse_error')})", flush=True)
        if args.pause > 0:
            time.sleep(args.pause)

    if tried_this_run >= args.max_tried:
        state["last_status"] = "per-run attempt limit reached"
        save_state(args.state, state)
    elif deferred_tail:
        state["last_status"] = "short remaining batch deferred to avoid low-utilization API requests"
        save_state(args.state, state)
    elif exhausted:
        state["last_status"] = "sampled strata exhausted"
        save_state(args.state, state)
    print(f"run complete: {tried_this_run:,} attempted, {accepted_this_run:,} passed semantic validation; "
          f"total saved={state.get('accepted_total', 0):,}; status={state.get('last_status')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
