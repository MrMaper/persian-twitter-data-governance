"""
تحلیل حساسیت پارامترها (پاسخ به داور ۲، بند ۳): اجرای چارچوب با ترکیب‌های مختلف
L_max و (α, β, γ) روی همان نمونه، و گزارش اثر هر ترکیب بر میانگین Q، نرخ نگه‌داری
و نرخ سوگیری - برای توجیه تجربی مقادیر پیش‌فرض انتخاب‌شده در مقاله.
"""
import json
import os
import random

import pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = p.DATA_PATH
OUT_PATH = os.path.join(HERE, "out", "param_sensitivity.json")

L_MAX_GRID = [10, 15, 20, 25, 30]
WEIGHT_GRID = [
    (0.5, 0.3, 0.2),  # مقدار پیش‌فرض مقاله
    (0.7, 0.2, 0.1),
    (0.3, 0.5, 0.2),
    (0.4, 0.2, 0.4),
    (0.33, 0.33, 0.34),
]


def run_once(data_path, l_max, alpha, beta, gamma):
    p.L_MAX = l_max
    p.ALPHA, p.BETA, p.GAMMA = alpha, beta, gamma
    rng = random.Random(p.SEED)
    result = p.pass1(rng, data_path=data_path, save_records=False)
    total = result["total"]
    kept = result["kept"]
    return {
        "L_max": l_max, "alpha": alpha, "beta": beta, "gamma": gamma,
        "mean_Q_before": round(result["q_before_sum"] / max(total, 1), 4),
        "mean_Q_after": round(result["q_after_sum"] / max(kept, 1), 4),
        "keep_ratio": round(kept / max(total, 1), 4),
        "bias_ratio_before": round(result["bias_before_flagged"] / max(total, 1), 4),
        "bias_ratio_after": round(result["bias_after_flagged"] / max(kept, 1), 4),
    }


def main(data_path=DEFAULT_DATA_PATH):
    orig_lmax, orig_a, orig_b, orig_g = p.L_MAX, p.ALPHA, p.BETA, p.GAMMA
    rows = []

    print("--- grid روی L_max (با وزن‌های پیش‌فرض) ---", flush=True)
    for l_max in L_MAX_GRID:
        row = run_once(data_path, l_max, orig_a, orig_b, orig_g)
        row["sweep"] = "L_max"
        rows.append(row)
        print(row, flush=True)

    print("--- grid روی (alpha, beta, gamma) (با L_max پیش‌فرض) ---", flush=True)
    for alpha, beta, gamma in WEIGHT_GRID:
        row = run_once(data_path, orig_lmax, alpha, beta, gamma)
        row["sweep"] = "weights"
        rows.append(row)
        print(row, flush=True)

    p.L_MAX, p.ALPHA, p.BETA, p.GAMMA = orig_lmax, orig_a, orig_b, orig_g

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
