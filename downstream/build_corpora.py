"""
ساخت دو پیکره متنی برای آزمایش downstream (فاز ۳، پاسخ به داور ۲):

- RAW: متن خام توئیت‌ها پیش از عبور از چارچوب پیشنهادی.
- GOVERNED: متن نهایی پس از هر سه مرحله (پاک‌سازی، فیلتر کیفیت، تعدیل سوگیری) -
  دقیقاً همان article/out/balanced_set.jsonl تولیدشده توسط pipeline.py.

هر دو پیکره به یک اندازه (بر اساس تعداد رکورد) نمونه‌گیری می‌شوند تا مقایسه Perplexity
منصفانه باشد (کنترل حجم داده آموزشی) و مقدار متفاوت صرفاً از تفاوت کیفیت/پاکیزگی متن
ناشی شود، نه از تفاوت حجم.

هم‌چنین یک validation set کاملاً جدا (بدون همپوشانی با هیچ‌کدام از دو پیکره آموزشی)
از بخشی مجزا از balanced_set.jsonl کنار گذاشته می‌شود.
"""
import json
import os
import random

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, "..")
DATA_PATH = os.environ.get(
    "RAW_DATA_PATH",
    os.path.join(CODE_DIR, "data", "extracted", "status_farsi_2022_10_1.json"),
)
BALANCED_PATH = os.path.join(CODE_DIR, "out", "balanced_set.jsonl")
OUT_DIR = os.path.join(HERE, "corpora")
os.makedirs(OUT_DIR, exist_ok=True)

VAL_SIZE = 20000  # اندازه validation set مشترک (کاملاً جدا از هر دو پیکره آموزشی)


def stream_raw_texts(path, limit=None):
    """استریم متن خام از فایل اصلی JSON (بدون هیچ پردازشی) - برای پیکره RAW."""
    import sys
    sys.path.insert(0, CODE_DIR)
    from pipeline import stream_objects  # noqa: E402

    n = 0
    for raw in stream_objects(path):
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        text = rec.get("text", "") or ""
        if text.strip():
            yield text.strip()
            n += 1
            if limit and n >= limit:
                return


def load_governed_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    if not os.path.exists(BALANCED_PATH):
        print("balanced_set.jsonl یافت نشد. ابتدا pipeline.py را کامل اجرا کنید.")
        return

    governed = load_governed_records(BALANCED_PATH)
    rng = random.Random(SEED)
    rng.shuffle(governed)

    val_records = governed[:VAL_SIZE]
    train_governed = governed[VAL_SIZE:]
    target_train_size = len(train_governed)

    print(f"governed total={len(governed)} val={len(val_records)} train_governed={target_train_size}")

    # پیکره RAW: به همان تعداد train_governed، از متن خام (بدون هیچ فیلتری) نمونه می‌گیریم.
    # چون RAW شامل رکوردهایی است که در GOVERNED هم حذف نشده‌اند، برای استقلال دو پیکره،
    # به سادگی یک نمونه تصادفی مستقل و هم‌اندازه از کل فایل خام برمی‌داریم (نه فقط زیرمجموعه‌ی
    # حذف‌شده) - چون هدف مقایسه «متن خام معمولی» با «متن پالایش‌شده» است، نه دو مجموعه‌ی
    # منحصر به فرد.
    raw_texts = []
    for text in stream_raw_texts(DATA_PATH, limit=target_train_size * 3):
        raw_texts.append(text)
    rng.shuffle(raw_texts)
    raw_texts = raw_texts[:target_train_size]

    with open(os.path.join(OUT_DIR, "train_raw.txt"), "w", encoding="utf-8") as f:
        for t in raw_texts:
            f.write(t.replace("\n", " ").strip() + "\n")

    with open(os.path.join(OUT_DIR, "train_governed.txt"), "w", encoding="utf-8") as f:
        for r in train_governed:
            f.write(r["text"].replace("\n", " ").strip() + "\n")

    with open(os.path.join(OUT_DIR, "validation.txt"), "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(r["text"].replace("\n", " ").strip() + "\n")

    print(f"train_raw: {len(raw_texts)} lines")
    print(f"train_governed: {len(train_governed)} lines")
    print(f"validation: {len(val_records)} lines")
    print("saved to", OUT_DIR)


if __name__ == "__main__":
    main()
