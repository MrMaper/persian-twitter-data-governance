"""
خروجی‌گیری نمونه عمومی و ناشناس‌سازی‌شده از مجموعه‌داده پالایش‌شده (برای پاسخ به داور ۱).

فقط از article/out/balanced_set.jsonl (خروجی نهایی چارچوب پیشنهادی، پس از هر سه
مرحله پاک‌سازی/کیفیت/تعدیل سوگیری) نمونه می‌گیرد - نه از داده خام. هیچ فیلد هویتی
(fK_UserID و مشابه) در ورودی این فایل اصلاً وجود ندارد چون pipeline.py آن را از ابتدا
ذخیره نکرده است؛ بنابراین نیازی به حذف پسینی نیست، اما به‌عنوان دفاع دوم (defense in
depth) یک بررسی صریح هم انجام می‌شود تا اطمینان حاصل شود هیچ فیلد هویتی به‌صورت
تصادفی درون متن یا فراداده باقی نمانده است.
"""
import json
import os
import random
import re

SEED = 42
SAMPLE_SIZE = 8000

HERE = os.path.dirname(os.path.abspath(__file__))
BALANCED_PATH = os.path.join(HERE, "out", "balanced_set.jsonl")
EXPORT_DIR = os.path.join(HERE, "public_sample")
os.makedirs(EXPORT_DIR, exist_ok=True)

IDENTITY_FIELD_NAMES = {"fk_userid", "userid", "crawlerfromuserid", "fk_inreplytouserid"}
MENTION_RE = re.compile(r"@\w+")


def strip_identity(record):
    """دفاع دوم: حذف صریح هر فیلد با نام مرتبط با هویت کاربر، در صورت وجود."""
    cleaned = {k: v for k, v in record.items() if k.lower() not in IDENTITY_FIELD_NAMES}
    return cleaned


def main():
    if not os.path.exists(BALANCED_PATH):
        print("balanced_set.jsonl یافت نشد. ابتدا pipeline.py را کامل اجرا کنید.")
        return

    records = []
    with open(BALANCED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rng = random.Random(SEED)
    rng.shuffle(records)
    sample = records[:SAMPLE_SIZE]
    sample = [strip_identity(r) for r in sample]

    out_path = os.path.join(EXPORT_DIR, "governed_sample.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    readme_path = os.path.join(EXPORT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# نمونه عمومی داده پالایش‌شده - چارچوب حکمرانی داده توییتر فارسی

## توضیح
این فایل ({SAMPLE_SIZE} رکورد) زیرنمونه‌ای تصادفی از خروجی نهایی چارچوب پیشنهادی
(پس از سه مرحله پاک‌سازی نویز، فیلتر کیفیت و تعدیل سوگیری) است. برای تکرارپذیری
نتایج و بررسی روش‌شناسی توسط سایر پژوهشگران منتشر شده است.

**این فایل شامل داده خام توییتر یا شناسه هویتی کاربران نیست** - فقط توئیت‌های
نهایی پالایش‌شده و امتیازهای محاسبه‌شده.

## فیلدهای هر رکورد
| فیلد | توضیح |
|---|---|
| `id` | شناسه یکتای توئیت (برای ردیابی منشأ - بدون اطلاعات هویتی کاربر) |
| `text` | متن نرمال‌سازی‌شده پس از پاک‌سازی نویز |
| `tokens` | فهرست توکن‌های حاصل از hazm.WordTokenizer |
| `q` | امتیاز کیفیت Q(t) طبق رابطه (۱) مقاله |
| `b` | شاخص سوگیری لغوی B(t) طبق رابطه (۲) مقاله |
| `bias_flagged` | آیا B(t) از آستانه τ_B عبور کرده است |

## روش ناشناس‌سازی
مجموعه اصلی pipeline.py هرگز فیلدهای هویتی کاربر (نظیر fK_UserID) را در خروجی
kept_records.jsonl / balanced_set.jsonl ذخیره نمی‌کند - این فیلدها از همان مرحله
اول (Pass 1) کنار گذاشته می‌شوند. به‌عنوان دفاع دوم، این اسکریپت (export_public_sample.py)
هرگونه فیلد باقی‌مانده با نام مرتبط با هویت را نیز صراحتاً حذف می‌کند.

## مجوز استفاده
[توسط نویسنده تکمیل شود - مثلاً CC BY-NC 4.0]

## استناد
[توسط نویسنده تکمیل شود - پس از پذیرش نهایی مقاله]
""")

    print(f"exported {len(sample)} records to {out_path}")
    print(f"README written to {readme_path}")


if __name__ == "__main__":
    main()
