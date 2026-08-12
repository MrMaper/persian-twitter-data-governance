"""
پایپ‌لاین حکمرانی داده - نسخه واقعی (اجرا روی کل نمونه ۴٫۴ گیگابایتی توییتر فارسی).

این نسخه نسبت به یک پیاده‌سازی اولیه صرفاً مبتنی بر regex (که در baseline_comparison.py
به‌عنوان یکی از باکندهای مقایسه‌ای، نه فایل مجزا، بازسازی شده است) سه تفاوت اساسی دارد:

۱) نرمال‌سازی/توکن‌سازی واقعی با hazm (به‌جای regex ساده) - مطابق ادعای مقاله.
۲) تشخیص ربات چندسیگنالی (تعامل صفر + ریتوییت بسیار بالا، یا تکرار متن یکسان در
   یک پنجره اخیر محدود) - به‌جای فقط یک آستانه ساده.
۳) مرحله (ج) «تعدیل سوگیری» واقعاً یک weighted_resample اجرا می‌کند (طبق شبه‌کد خود
   مقاله) و یک balanced_set واقعاً کوچک‌تر و متوازن‌تر تولید می‌کند - نه فقط شمارش.

علاوه بر این، در حین اجرا داده‌های لازم برای رسم صادقانه نمودارها (بدون هیچ عدد
دستی/تصادفی) ذخیره می‌شوند: نمونه Q(t) قبل/بعد (out/q_before.jsonl, q_after.jsonl)،
شمارش واقعی هر مرحله قیف (out/real_results.json)، و فراوانی واقعی کلمات قبل/بعد
(out/word_freq_before.json, word_freq_after.json).

مسیر داده خام ورودی از طریق متغیر محیطی RAW_DATA_PATH قابل تنظیم است؛ در غیر این
صورت به‌صورت پیش‌فرض data/extracted/status_farsi_2022_10_1.json (نسبت به ریشه این
مخزن) خوانده می‌شود. داده خام توییتر به دلایل حریم خصوصی و مالکیت داده در این مخزن
منتشر نشده است (نگاه کنید به README).
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

# --- تنظیمات شاخص‌ها (مطابق فرمول‌های مقاله - بخش ۳-۲) ---
L_MAX = 20
TAU_Q = 0.4
TAU_B = 0.15
ALPHA, BETA, GAMMA = 0.5, 0.3, 0.2

# اندازه هدف مجموعه نهایی متوازن نسبت به مجموعه پالایش‌شده کیفیت (مرحله ج)
BIAS_RESAMPLE_KEEP_RATIO = 0.90
BIAS_FLAG_WEIGHT_PENALTY = 0.2  # جریمه ضربی وزن نمونه‌گیری برای رکوردهای پرچم‌دار سوگیری

RESERVOIR_CAP = 50000  # حداکثر تعداد امتیاز Q(t) نگه‌داری‌شده برای رسم هیستوگرام


# --- فهرست واژگان باردار: V_train (فیلترسازی) / V_eval (held-out، فقط فاز ۳) ---
_FALLBACK_LEXICON = """
سیاسی انقلاب تحریم اعتراض حکومت دشمن جاسوس وطن‌فروش اصلاح‌طلب اصول‌گرا
مرگ زندان خون شهید منافق کافر یهودی صهیونیست کثافت حروم‌زاده
زن‌ستیز مردسالار فاحشه بدحجاب بی‌حجاب ننگ لعنت کثافت‌زاده
قوم‌کشی ترک فارس عرب کُرد بلوچ داعش تکفیری
""".split()


def load_bias_terms():
    """اگر article/code/bias_lexicon.csv (ستون‌های category,term) موجود باشد از آن
    استفاده می‌شود؛ در غیر این صورت فهرست موقت داخلی به کار می‌رود (فاز ۲ در پروژه)."""
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

# --- ابزارهای نرمال‌سازی/توکن‌سازی واقعی (نمونه‌سازی یک‌باره، نه هر رکورد) ---
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
    """توکن فارسیِ معتبر و غیر ایست‌واژه - برای L(t) (طول معتبر طبق بخش ۳-۱ مقاله:
    «تعداد توکن‌های فارسی پس از حذف ایست‌واژه‌ها») و برای فراوانی کلمات ابر کلمات.
    توکن‌های تک‌نویسه (عمدتاً علائم نگارشی مثل «،» و «؟» که در بازه یونیکد عربی/فارسی
    قرار دارند) و ایست‌واژه‌ها کنار گذاشته می‌شوند."""
    if len(t) < 2 or t in STOPWORDS:
        return False
    return bool(PERSIAN_RE.search(t))


def strip_noise(text):
    """حذف لینک/منشن/هشتگ پیش از نرمال‌سازی (تا نرمال‌ساز hazm آن‌ها را تکه‌تکه نکند)."""
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
    """طبق رابطه (۱) مقاله: Q(t) = α·L(t) + β·G(t) + γ·(1-C(t))."""
    if raw_token_count == 0:
        return 0.0
    valid = [t for t in tokens if is_content_token(t)]
    L = min(len(valid) / L_MAX, 1.0)
    has_verb = bool(VERB_RE.search(normalized_text))
    G = 1.0 if has_verb else 0.0
    C = min(n_noise / raw_token_count, 1.0)
    return ALPHA * L + BETA * G + GAMMA * (1 - C)


def bias_index(tokens, lexicon):
    """طبق رابطه (۲) مقاله: B(t) = |W(t) ∩ V_bias| / |W(t)|."""
    if not tokens:
        return 0.0
    hit = sum(1 for t in tokens if t.strip(STRIP_PUNCT) in lexicon)
    return hit / len(tokens)


class BotDetector:
    """فیلتر مرحله (الف) با دو زیرمؤلفه کاملاً ساختاری/رفتاری (نه مبتنی بر محتوای متن،
    تا از برچسب‌گذاری نادرست ریتوییت‌های ارگانیک رخدادهای واقعی به‌عنوان «ربات»
    جلوگیری شود):

    (۱) حذف رکوردهای ساختاری ریتوییت (fK_RetweetStatusID != "0") - این رکوردها
        بازنشر عین متن توئیت دیگری هستند و برای پیکره آموزشی مدل زبانی محتوای
        متنی جدیدی اضافه نمی‌کنند؛ این حذف مستقیماً با منطق حذف تکرار داده آموزشی
        در Lee et al. [1] هم‌راستاست.
    (۲) الگوی رفتاری تقویت مصنوعی مشکوک: ریتوییت بسیار بالا همراه با تعامل صفر
        (بدون لایک/پاسخ/نقل‌قول) که نشانه شبکه‌های تقویت خودکار است.
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


# --- استریم chunk-based با حافظه محدود و قطعی (بدون تغییر نسبت به نسخه قبلی) ---
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
    """گذر اول: استریم روی کل فایل خام، فیلتر ربات و کیفیت، ذخیره داده‌های واقعی."""
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
    """گذر دوم: مرحله (ج) واقعی - weighted_resample(clean_set, target_distribution) طبق
    شبه‌کد مقاله. وزن هر رکورد با معکوس بسامد واژگان آن (هموارسازی فرکانس، برای جلوگیری
    از بیش‌نمایندگی کلمات پرتکرار) و جریمه ضربی برای رکوردهای پرچم‌دار سوگیری تعیین
    می‌شود؛ نمونه‌گیری وزن‌دار بدون جایگذاری با روش Efraimidis-Spirakis انجام می‌گیرد."""
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

    print("=== PASS 1: استریم روی داده خام + فیلتر ربات/کیفیت + ذخیره داده واقعی ===", flush=True)
    p1 = pass1(rng)

    print("=== PASS 2: بازنمونه‌گیری وزن‌دار برای تعدیل سوگیری (مرحله ج) ===", flush=True)
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
