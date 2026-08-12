import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_FONT = "DejaVu Sans"
for c in ["IRANSans", "Vazir", "Tahoma", "B Nazanin", "DejaVu Sans"]:
    try:
        fm.findfont(c, fallback_to_default=False)
        _FONT = c
        break
    except Exception:
        pass

plt.rcParams["font.family"] = _FONT
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figs")
OUT_DIR = os.path.join(HERE, "out")
RESULTS_PATH = os.path.join(HERE, "out", "real_results.json")
os.makedirs(FIG_DIR, exist_ok=True)


def _load_jsonl_field(path, field):
    values = []
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(json.loads(line)[field])
            except (json.JSONDecodeError, KeyError):
                continue
    return values


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name):
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


def make_flowchart():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#2c3e50"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.15,rounding_size=0.25", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=11, color="#1a1a1a", fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=18, lw=2, color="#34495e"))

    ax.text(5, 9.4, "نمای کلی خط لوله حکمرانی داده پیشنهادی", ha="center", fontsize=13, fontweight="bold")
    box(3.5, 7.6, 3, 1.1, "داده‌های خام توییتر فارسی", "#fdebd0", "#ca6f1e")
    arrow(5, 7.6, 5, 6.9)
    box(3.5, 5.8, 3, 1.1, "الف) پاک‌سازی نویز (ربات، لینک، نرمال‌سازی)", "#d6eaf8", "#2471a3")
    arrow(5, 5.8, 5, 5.1)
    box(3.5, 4.0, 3, 1.1, "ب) ارزیابی کیفیت متنی (طول، گرامر، محاوره)", "#d5f5e3", "#1e8449")
    arrow(5, 4.0, 5, 3.3)
    box(3.5, 2.2, 3, 1.1, "ج) تشخیص و تعدیل سوگیری (وازگان باردار، متوازن‌سازی)", "#fadbd8", "#b03a2e")
    arrow(5, 2.2, 5, 1.5)
    box(3.5, 0.4, 3, 1.0, "مجموعه‌داده تمیز، باکیفیت و متوازن برای آموزش LLM", "#e8daef", "#6c3483")
    _save(fig, "flowchart.png")


def make_quality_hist():
    """رسم هیستوگرام از امتیازهای Q(t) واقعی که pipeline.py حین اجرا روی داده واقعی
    در article/out/q_before.jsonl و q_after.jsonl ذخیره کرده (نمونه reservoir تا ۵۰هزار مقدار)."""
    import numpy as np
    before = _load_jsonl_field(os.path.join(OUT_DIR, "q_before.jsonl"), "q")
    after = _load_jsonl_field(os.path.join(OUT_DIR, "q_after.jsonl"), "q")
    if not before or not after:
        print("داده واقعی Q(t) یافت نشد. ابتدا pipeline.py را روی داده واقعی اجرا کنید.")
        return
    mean_before = sum(before) / len(before)
    mean_after = sum(after) / len(after)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(before, bins=bins, alpha=0.65, label=f"پیش از چارچوب (میانگین {mean_before:.2f})", color="#d9534f", edgecolor="white")
    ax.hist(after, bins=bins, alpha=0.65, label=f"پس از چارچوب (میانگین {mean_after:.2f})", color="#5cb85c", edgecolor="white")
    ax.set_xlabel("امتیاز کیفیت Q(t)")
    ax.set_ylabel("فراوانی (نمونه reservoir)")
    ax.set_title("توزیع کیفیت داده‌های توییتر فارسی (داده واقعی)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "quality_dist.png")


def make_funnel():
    """رسم نمودار قیفی از شمارش واقعی مراحل که pipeline.py در tools/real_results.json
    ذخیره کرده - هیچ عددی دستی/فرضی نیست."""
    results = _load_json(RESULTS_PATH)
    if not results or "funnel" not in results:
        print("داده واقعی funnel یافت نشد. ابتدا pipeline.py را روی داده واقعی اجرا کنید.")
        return
    f = results["funnel"]
    stages = [
        ("توییت‌های خام ورودی", f["raw"]),
        ("پس از حذف ربات و نویز (الف)", f["after_bot_noise_filter"]),
        ("پس از فیلتر کیفیت (ب)", f["after_quality_filter"]),
        ("پس از تعدیل سوگیری (ج)", f["after_bias_mitigation"]),
    ]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    total = values[0]
    # چون افت واقعی داده بسیار تندتر از نسخه قبلی (فرضی) است، برچسب‌ها همیشه بیرون
    # میله (باالای آن) قرار می‌گیرند تا برای میله‌های باریک هم بریده نشوند.
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (lab, val) in enumerate(zip(labels, values)):
        width = max((val / total) * 0.8, 0.02)
        x0 = 0.5 - width / 2
        ax.barh(i, width, left=x0, height=0.5, color="#2e86c1", edgecolor="white")
        pct = val / total * 100
        ax.text(0.5, i - 0.42, f"{lab}\n{val:,}  ({pct:.1f}٪)",
                ha="center", va="bottom", color="#1a1a1a", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([])
    ax.set_ylim(len(stages) - 0.3, -1.1)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("نمودار قیفی (Funnel Chart) ریزش داده‌ها")
    for sp in ["top", "right", "bottom", "left"]:
        ax.spines[sp].set_visible(False)
    _save(fig, "funnel.png")


def make_wordcloud():
    """ابر کلمات از فراوانی واقعی توکن‌های فارسی که pipeline.py حین اجرا روی داده واقعی
    در article/out/word_freq_before.json و word_freq_after.json ذخیره کرده."""
    from wordcloud import WordCloud
    import arabic_reshaper
    from bidi.algorithm import get_display
    from PIL import Image, ImageDraw, ImageFont

    freq_before_raw = _load_json(os.path.join(OUT_DIR, "word_freq_before.json"))
    freq_after_raw = _load_json(os.path.join(OUT_DIR, "word_freq_after.json"))
    if not freq_before_raw or not freq_after_raw:
        print("داده واقعی فراوانی کلمات یافت نشد. ابتدا pipeline.py را روی داده واقعی اجرا کنید.")
        return
    # ۳۰ واژه پرتکرار برای خوانایی ابر کلمات
    freq_before_raw = dict(sorted(freq_before_raw.items(), key=lambda kv: -kv[1])[:30])
    freq_after_raw = dict(sorted(freq_after_raw.items(), key=lambda kv: -kv[1])[:30])

    def process_frequencies(freq_dict):
        out = {}
        for word, weight in freq_dict.items():
            out[get_display(arabic_reshaper.reshape(word))] = weight
        return out

    freq_before = process_frequencies(freq_before_raw)
    freq_after = process_frequencies(freq_after_raw)

    font_path = r"C:\Windows\Fonts\IRANSans.ttf"
    common = dict(
        font_path=font_path, width=800, height=600, background_color="white",
        margin=8, max_words=30, min_font_size=14, random_state=42,
    )
    img_before = WordCloud(**common, colormap="Reds").generate_from_frequencies(freq_before).to_image()
    img_after = WordCloud(**common, colormap="Greens").generate_from_frequencies(freq_after).to_image()

    def render_text(text, size, fill, bg=(255, 255, 255)):
        bidi_text = get_display(arabic_reshaper.reshape(text))
        font = ImageFont.truetype(font_path, size)
        tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        bbox = tmp.textbbox((0, 0), bidi_text, font=font)
        w = bbox[2] - bbox[0] + 20
        h = bbox[3] - bbox[1] + 20
        im = Image.new("RGB", (w, h), bg)
        ImageDraw.Draw(im).text((10 - bbox[0], 10 - bbox[1]), bidi_text, font=font, fill=fill)
        return im

    sup = render_text("ابر کلمات مقایسه‌ای پیکره پیش و پس از اعمال چارچوب", 30, (20, 20, 20))
    sub_l = render_text("الف) داده‌های خام (پیش از چارچوب)", 24, (192, 57, 43))
    sub_r = render_text("ب) داده‌های پالایش‌شده (پس از چارچوب)", 24, (30, 132, 73))

    gap = 30
    pw, ph = img_before.size
    W = pw * 2 + gap
    H = sup.height + 20 + ph + 20 + max(sub_l.height, sub_r.height)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(sup, ((W - sup.width) // 2, 0))
    canvas.paste(img_before, (0, sup.height + 20))
    canvas.paste(img_after, (pw + gap, sup.height + 20))
    canvas.paste(sub_l, ((pw - sub_l.width) // 2, sup.height + 20 + ph + 10))
    canvas.paste(sub_r, (pw + gap + (pw - sub_r.width) // 2, sup.height + 20 + ph + 10))

    out = os.path.join(FIG_DIR, "wordcloud.png")
    canvas.save(out, dpi=(300, 300))
    print("saved:", out)


if __name__ == "__main__":
    print("font:", _FONT)
    make_flowchart()
    make_quality_hist()
    make_funnel()
    make_wordcloud()
    print("all figures generated.")