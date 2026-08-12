# Persian Twitter Data Governance Framework

کد پیاده‌سازی چارچوب حکمرانی داده برای پیش‌پردازش، ارزیابی کیفیت ($Q(t)$) و کاهش سوگیری لغوی ($B(t)$) داده‌های توییتر فارسی، به‌عنوان دادگان آموزشی مدل‌های زبانی بزرگ. این مخزن کد همراه مقاله «ارزیابی کیفیت و سوگیری داده‌های شبکه‌های اجتماعی در آموزش مدل‌های زبانی بزرگ فارسی: ارائه یک چارچوب مبتنی بر حکمرانی داده» (نشریه علوم رایانش و فناوری اطلاعات، انجمن کامپیوتر ایران) منتشر شده است.

یک نمونه عمومی و ناشناس‌سازی‌شده از خروجی نهایی این چارچوب (۸۰۰۰ رکورد) به‌صورت جداگانه روی Zenodo منتشر می‌شود؛ لینک آن پس از انتشار در این README افزوده خواهد شد. **داده خام توییتر در این مخزن منتشر نشده است** (ملاحظات حریم خصوصی و مالکیت داده).

## ساختار مخزن

```
pipeline.py                  خط لوله اصلی: پاک‌سازی نویز، امتیازدهی Q(t)/B(t)، تعدیل سوگیری
bias_lexicon.csv             فهرست واژگان باردار V_bias (۶۹ واژه، ۵ دسته)
baseline_comparison.py       مقایسه Regex/Hazm/Parsivar روی همان فرمول Q(t)/B(t)
param_sensitivity.py         تحلیل حساسیت L_max و (α, β, γ)
make_figures.py              رسم نمودارها از خروجی واقعی pipeline.py
export_public_sample.py      خروجی‌گیری نمونه عمومی ناشناس‌سازی‌شده
pass2_hard_filter.py         آزمایش جایگزین: حذف قطعی رکوردهای پرچم‌دار به‌جای وزن‌دهی
downstream/                  ارزیابی downstream (Perplexity، Accuracy) روی دو مدل ادامه‌آموزش‌دیده
  build_corpora.py             ساخت پیکره‌های RAW/GOVERNED/validation
  train_lm.py                  ادامه‌آموزش (Continued Pretraining)
  eval_perplexity.py           سنجش Perplexity روی validation set مشترک
  prepare_sentiment_data.py    دانلود دیتاست ParsiNLU sentiment
  train_classifier.py          فاین‌تیون و سنجش دقت طبقه‌بندی احساس
  eval_fairness.py             سنجش اکتشافی نسبت واژگان باردار در تولیدات (V_eval)
```

## نصب

```bash
pip install -r requirements.txt
```

آموزش/ارزیابی downstream به GPU با CUDA و PyTorch نیاز دارد؛ خط لوله اصلی (`pipeline.py`) صرفاً CPU-bound است.

## اجرا

### ۱. خط لوله اصلی

داده خام (فایل JSON آرایه‌ای از توئیت‌ها با فیلد `text`) را در `data/extracted/status_farsi_2022_10_1.json` قرار دهید یا مسیر آن را با متغیر محیطی `RAW_DATA_PATH` مشخص کنید:

```bash
RAW_DATA_PATH=/path/to/your/data.json python pipeline.py
```

خروجی‌ها (`out/real_results.json`, `out/kept_records.jsonl`, `out/balanced_set.jsonl` و ...) در پوشه `out/` ذخیره می‌شوند.

### ۲. نمودارها، تحلیل حساسیت، مقایسه baseline

```bash
python make_figures.py            # پس از اجرای pipeline.py
python param_sensitivity.py
python baseline_comparison.py /path/to/data.json
```

### ۳. نمونه عمومی

```bash
python export_public_sample.py    # پس از اجرای pipeline.py
```

### ۴. ارزیابی downstream

```bash
cd downstream
python build_corpora.py
python train_lm.py --corpus corpora/train_raw.txt      --out models/raw
python train_lm.py --corpus corpora/train_governed.txt --out models/governed
python eval_perplexity.py --model models/raw
python eval_perplexity.py --model models/governed
python prepare_sentiment_data.py
python train_classifier.py --model models/raw       --out clf/raw
python train_classifier.py --model models/governed  --out clf/governed
python eval_fairness.py --model models/raw
python eval_fairness.py --model models/governed
```

> سنجش اکتشافی `eval_fairness.py` (نسبت واژگان باردار در تولیدات آزاد) در آزمایش‌های ما در مقیاس‌های مختلف نمونه‌برداری نتیجه‌ای پایدار نداد و در مقاله به‌عنوان یافته قطعی گزارش نشده است؛ اسکریپت برای شفافیت و امکان تکرار/ادامه این آزمایش توسط سایر پژوهشگران نگه داشته شده است.

## فهرست واژگان باردار (`bias_lexicon.csv`)

فهرست به‌صورت دستی و در سه گام تدوین شد (استخراج نامزد از توزیع فراوانی واقعی پیکره ← دسته‌بندی مفهومی ← پالایش بر پایه معیار «بار ارزشی»؛ نه صرف ارتباط موضوعی). ستون‌ها: `category, term, notes`. دسته «سیاسی/امنیتی» عمداً متقارن سیاسی طراحی شده و واژگان خنثای گفتمان سیاسی/اجتماعی را شامل نمی‌شود. جزئیات روش‌شناسی کامل در بخش ۳-۱ مقاله آمده است.

## استناد

پس از پذیرش نهایی مقاله، اطلاعات استناد (DOI و ارجاع کامل) در این بخش افزوده می‌شود.

## مجوز

کد این مخزن تحت مجوز MIT منتشر شده است ([LICENSE](LICENSE)).
