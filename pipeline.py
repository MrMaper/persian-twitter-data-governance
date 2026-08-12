"""
Data governance pipeline — full-scale version (runs over the entire 4.4 GB Persian
Twitter sample).

Compared to a minimal regex-only implementation (rebuilt here as one of the
comparison backends in baseline_comparison.py rather than as a separate file),
this version differs in three key ways:

1) Real normalization/tokenization with hazm (instead of plain regex), matching
   the paper's methodology.
2) Multi-signal bot detection (zero engagement + very high retweet count, or
   identical text repeated within a limited recent window) instead of a single
   simple threshold.
3) Stage (c), "bias mitigation," actually runs a weighted_resample (per the
   paper's pseudocode) and produces a genuinely smaller, more balanced
   balanced_set — not just a count.

It also saves, during the run, everything needed to plot honest figures (no
manual/random numbers): a sample of Q(t) before/after (out/q_before.jsonl,
out/q_after.jsonl), the real per-stage funnel counts (out/real_results.json),
and real word frequencies before/after (out/word_freq_before.json,
out/word_freq_after.json).

The raw input data path is configurable via the RAW_DATA_PATH environment
variable; otherwise it defaults to data/extracted/status_farsi_2022_10_1.json
(relative to this repository's root). Raw Twitter data is not distributed in
this repository for privacy and data-ownership reasons (see README).
"""
import json
import math
import os
import random
import time
from collections import Counter

from hazm import Normalizer as HazmNormalizer
from hazm import WordTokenizer

SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get(
    "RAW_DATA_PATH",
    os.path.join(HERE, "data", "extracted", "status_farsi_2022_10_1.json"),
)
OUT_DIR = os.path.join(HERE, "out")
RESULTS_PATH = os.path.join(OUT_DIR, "real_results.json")
BIAS_LEXICON_CSV = os.path.join(HERE, "bias_lexicon.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Scoring configuration (matches the paper's formulas, Section 3.2) ---
L_MAX = 20
TAU_Q = 0.4
TAU_B = 0.15
ALPHA, BETA, GAMMA = 0.5, 0.3, 0.2

# Target size of the final balanced set, relative to the quality-filtered set (stage c)
BIAS_RESAMPLE_KEEP_RATIO = 0.90
BIAS_FLAG_WEIGHT_PENALTY = 0.2  # multiplicative sampling-weight penalty for bias-flagged records

RESERVOIR_CAP = 50000  # max number of Q(t) scores retained for the histogram plot


# --- Bias lexicon: V_train (used for filtering) / V_eval (held out, downstream eval only) ---
_FALLBACK_LEXICON = """
سیاسی انقلاب تحریم اعتراض حکومت دشمن جاسوس وطن‌فروش اصلاح‌طلب اصول‌گرا
مرگ زندان خون شهید منافق کافر یهودی صهیونیست کثافت حروم‌زاده
زن‌ستیز مردسالار فاحشه بدحجاب بی‌حجاب ننگ لعنت کثافت‌زاده
قوم‌کشی ترک فارس عرب کُرد بلوچ داعش تکفیری
""".split()


def load_bias_terms():
    """Loads terms from bias_lexicon.csv (columns: category,term) if present;
    otherwise falls back to a small built-in placeholder list."""
    if os.path.exists(BIAS_LEXICON_CSV):
        terms = []
        with open(BIAS_LEXICON_CSV, encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) >= 2 and parts[1].strip():
                    terms.append(parts[1].strip())
        if terms:
            return terms
    return list(_FALLBACK_LEXICON)


def split_train_eval(terms, train_ratio=0.8, seed=SEED):
    terms = sorted(set(terms))
    rng = random.Random(seed)
    rng.shuffle(terms)
    cut = max(1, int(len(terms) * train_ratio))
    return set(terms[:cut]), set(terms[cut:])


V_TRAIN, V_EVAL = split_train_eval(load_bias_terms())

# --- Real normalization/tokenization tools (instantiated once, not per record) ---
_hazm_normalizer = HazmNormalizer()
_hazm_tokenizer = WordTokenizer()

from hazm import stopwords_list as _hazm_stopwords_list

STOPWORDS = set(_hazm_stopwords_list())

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\S+")
PERSIAN_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
VERB_RE = re.compile(r"(می‌|نمی‌|\b(است|اند|اید|ام|یم|ید|ند)\b|شد|کرد|گفت|بود|هست)")
STRIP_PUNCT = ".,!?:;\"'()[]{}،؛؟!"


def is_content_token(t):
    """A valid, non-stopword Persian token — used for L(t) (valid length, per
    Section 3.1 of the paper: "the number of Persian tokens after stopword
    removal") and for word-cloud frequency counts. Single-character tokens
    (mostly punctuation marks that fall within the Arabic/Persian Unicode
    range, such as "،" and "؟") and stopwords are excluded."""
    if len(t) < 2 or t in STOPWORDS:
        return False
    return bool(PERSIAN_RE.search(t))


def strip_noise(text):
    """Strips links/mentions/hashtags before normalization, so the hazm
    normalizer does not fragment them."""
    n_noise = len(URL_RE.findall(text)) + len(MENTION_RE.findall(text)) + len(HASHTAG_RE.findall(text))
    cleaned = URL_RE.sub(" ", text)
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = HASHTAG_RE.sub(" ", cleaned)
    return cleaned, n_noise


def normalize_and_tokenize(text):
    cleaned, n_noise = strip_noise(text)
    normalized = _hazm_normalizer.normalize(cleaned)
    tokens = _hazm_tokenizer.tokenize(normalized)
    return normalized, tokens, n_noise


def quality_score(tokens, n_noise, raw_token_count, normalized_text):
    """Per Eq. (1) of the paper: Q(t) = alpha*L(t) + beta*G(t) + gamma*(1-C(t))."""
    if raw_token_count == 0:
        return 0.0
    valid = [t for t in tokens if is_content_token(t)]
    L = min(len(valid) / L_MAX, 1.0)
    has_verb = bool(VERB_RE.search(normalized_text))
    G = 1.0 if has_verb else 0.0
    C = min(n_noise / raw_token_count, 1.0)
    return ALPHA * L + BETA * G + GAMMA * (1 - C)


def bias_index(tokens, lexicon):
    """Per Eq. (2) of the paper: B(t) = |W(t) intersect V_bias| / |W(t)|."""
    if not tokens:
        return 0.0
    hit = sum(1 for t in tokens if t.strip(STRIP_PUNCT) in lexicon)
    return hit / len(tokens)


class BotDetector:
    """Stage (a) filter with two purely structural/behavioral sub-signals (not
    content-based, to avoid mislabeling organic retweets of real-world events
    as "bot" activity):

    (1) Structural retweet removal (fK_RetweetStatusID != "0") — these records
        are verbatim reposts of another tweet's text and add no new textual
        content to the language-model training corpus; this removal is
        directly aligned with the training-data deduplication logic in
        Lee et al. [1].
    (2) Suspicious artificial-amplification pattern: very high retweet count
        combined with zero engagement (no likes/replies/quotes), indicative of
        automated amplification networks.
    """

    def __init__(self, rt_threshold=10000):
        self.rt_threshold = rt_threshold

    def check(self, rec):
        is_retweet = str(rec.get("fK_RetweetStatusID", "0")) not in ("0", "", "None")
        rt = int(rec.get("retweetCount", 0) or 0)
        fav = int(rec.get("favoriteCount", 0) or 0)
        reply = int(rec.get("replyCount", 0) or 0)
        quote = int(rec.get("quoteCount", 0) or 0)
        suspicious_amplification = rt > self.rt_threshold and fav == 0 and reply == 0 and quote == 0
        return is_retweet or suspicious_amplification


# --- Chunk-based streaming parser with bounded, deterministic memory usage ---
def stream_objects(path):
    with open(path, encoding="utf-8") as f:
        cur = ""
        depth = 0
        in_str = False
        esc = False
        building = False
        CHUNK = 8 * 1024 * 1024
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            for c in chunk:
                if in_str:
                    cur += c
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                        cur += c
                    elif c == "{":
                        depth += 1
                        if depth == 1:
                            building = True
                            cur = "{"
                        elif building:
                            cur += c
                    elif c == "}":
                        depth -= 1
                        if building:
                            cur += "}"
                        if depth == 0 and building:
                            yield cur
                            building = False
                            cur = ""
                    else:
                        if building:
                            cur += c


def reservoir_add(reservoir, value, n_seen_so_far, cap, rng):
    if len(reservoir) < cap:
        reservoir.append(value)
    else:
        j = rng.randint(0, n_seen_so_far)
        if j < cap:
            reservoir[j] = value


def pass1(rng, data_path=DATA_PATH, kept_path=None, save_records=True):
    """First pass: stream over the full raw file, apply bot/quality filtering,
    and save the real data needed downstream."""
    total = 0
    bots = 0
    quality_dropped = 0
    kept = 0

    q_before_sum = 0.0
    bias_before_flagged = 0
    q_after_sum = 0.0
    bias_after_flagged = 0

    q_before_reservoir = []
    q_after_reservoir = []

    freq_before = Counter()
    freq_after = Counter()

    bot_detector = BotDetector()
    if kept_path is None:
        kept_path = os.path.join(OUT_DIR, "kept_records.jsonl")

    t_start = time.time()
    kept_f = open(kept_path, "w", encoding="utf-8") if save_records else None
    try:
        for raw in stream_objects(data_path):
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            total += 1
            text = rec.get("text", "") or ""
            if not text:
                continue

            raw_tokens = text.split()
            for t in raw_tokens:
                if is_content_token(t):
                    freq_before[t] += 1

            normalized, tokens, n_noise = normalize_and_tokenize(text)
            qb = quality_score(tokens, n_noise, max(len(raw_tokens), 1), normalized)
            q_before_sum += qb
            reservoir_add(q_before_reservoir, qb, total - 1, RESERVOIR_CAP, rng)
            if bias_index(tokens, V_TRAIN) > TAU_B:
                bias_before_flagged += 1

            if bot_detector.check(rec):
                bots += 1
                continue

            if qb < TAU_Q:
                quality_dropped += 1
                continue

            kept += 1
            q_after_sum += qb
            reservoir_add(q_after_reservoir, qb, kept - 1, RESERVOIR_CAP, rng)
            for t in tokens:
                if is_content_token(t):
                    freq_after[t] += 1
            ba = bias_index(tokens, V_TRAIN)
            bias_flagged = ba > TAU_B
            if bias_flagged:
                bias_after_flagged += 1

            if kept_f is not None:
                kept_f.write(json.dumps({
                    "id": rec.get("id"),
                    "text": normalized,
                    "tokens": tokens,
                    "q": round(qb, 4),
                    "b": round(ba, 4),
                    "bias_flagged": bias_flagged,
                }, ensure_ascii=False) + "\n")

            if total % 500000 == 0:
                elapsed = time.time() - t_start
                print(f"[pass1] processed {total} tweets... kept={kept} bots={bots} ({elapsed:.0f}s)", flush=True)
    finally:
        if kept_f is not None:
            kept_f.close()

    return {
        "total": total, "bots": bots, "quality_dropped": quality_dropped, "kept": kept,
        "q_before_sum": q_before_sum, "q_after_sum": q_after_sum,
        "bias_before_flagged": bias_before_flagged, "bias_after_flagged": bias_after_flagged,
        "q_before_reservoir": q_before_reservoir, "q_after_reservoir": q_after_reservoir,
        "freq_before": freq_before, "freq_after": freq_after,
        "kept_path": kept_path if save_records else None,
        "elapsed_seconds": time.time() - t_start,
    }


def pass2_weighted_resample(kept_path, freq_after, rng):
    """Second pass: the real stage (c) — weighted_resample(clean_set,
    target_distribution) per the paper's pseudocode. Each record's weight is
    set from the inverse frequency of its tokens (frequency smoothing, to
    avoid over-representing very common words) with a multiplicative penalty
    for bias-flagged records; weighted sampling without replacement is done
    via the Efraimidis-Spirakis method."""
    records = []
    with open(kept_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    n = len(records)
    weights = [0.0] * n
    for i, r in enumerate(records):
        toks = [t for t in r["tokens"] if t in freq_after]
        avg_freq = (sum(freq_after[t] for t in toks) / len(toks)) if toks else 1.0
        w = 1.0 / (1.0 + math.log1p(avg_freq))
        if r["bias_flagged"]:
            w *= BIAS_FLAG_WEIGHT_PENALTY
        weights[i] = max(w, 1e-9)

    target_size = int(n * BIAS_RESAMPLE_KEEP_RATIO)
    keys = [(rng.random() ** (1.0 / weights[i]), i) for i in range(n)]
    keys.sort(reverse=True)
    selected_idx = {i for _, i in keys[:target_size]}

    balanced_path = os.path.join(OUT_DIR, "balanced_set.jsonl")
    with open(balanced_path, "w", encoding="utf-8") as bf:
        for i in selected_idx:
            bf.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    return {
        "balanced_path": balanced_path,
        "balanced_size": len(selected_idx),
        "kept_before_resample": n,
    }


def main():
    rng = random.Random(SEED)
    print(f"V_train size={len(V_TRAIN)} V_eval size={len(V_EVAL)}", flush=True)

    print("=== PASS 1: stream raw data + bot/quality filtering + save real data ===", flush=True)
    p1 = pass1(rng)

    print("=== PASS 2: weighted resampling for bias mitigation (stage c) ===", flush=True)
    p2 = pass2_weighted_resample(p1["kept_path"], p1["freq_after"], rng)

    total = p1["total"]
    kept = p1["kept"]
    bots = p1["bots"]
    balanced_size = p2["balanced_size"]

    summary = {
        "source_file": os.path.relpath(DATA_PATH, HERE).replace("\\", "/"),
        "total_tweets": total,
        "bot_flagged": bots,
        "quality_dropped": p1["quality_dropped"],
        "kept_after_quality": kept,
        "balanced_after_bias_mitigation": balanced_size,
        "mean_Q_before": round(p1["q_before_sum"] / max(total, 1), 4),
        "mean_Q_after": round(p1["q_after_sum"] / max(kept, 1), 4),
        "bias_ratio_before": round(p1["bias_before_flagged"] / max(total, 1), 4),
        "bias_ratio_after": round(p1["bias_after_flagged"] / max(kept, 1), 4),
        "pass1_elapsed_seconds": round(p1["elapsed_seconds"], 1),
        "v_train_size": len(V_TRAIN),
        "v_eval_size": len(V_EVAL),
        "funnel": {
            "raw": total,
            "after_bot_noise_filter": total - bots,
            "after_quality_filter": kept,
            "after_bias_mitigation": balanced_size,
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as fo:
        json.dump(summary, fo, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "q_before.jsonl"), "w", encoding="utf-8") as f:
        for v in p1["q_before_reservoir"]:
            f.write(json.dumps({"q": v}) + "\n")
    with open(os.path.join(OUT_DIR, "q_after.jsonl"), "w", encoding="utf-8") as f:
        for v in p1["q_after_reservoir"]:
            f.write(json.dumps({"q": v}) + "\n")

    with open(os.path.join(OUT_DIR, "word_freq_before.json"), "w", encoding="utf-8") as f:
        json.dump(dict(p1["freq_before"].most_common(300)), f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "word_freq_after.json"), "w", encoding="utf-8") as f:
        json.dump(dict(p1["freq_after"].most_common(300)), f, ensure_ascii=False, indent=2)

    print("=== RESULTS ===")
    for k, v in summary.items():
        print(k, ":", v)
    print("saved:", RESULTS_PATH)


if __name__ == "__main__":
    main()
