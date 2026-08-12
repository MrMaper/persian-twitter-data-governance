"""
An alternative experiment for stage (c): instead of probabilistic reweighting
(the 0.2 multiplicative penalty in pipeline.py), bias-flagged records are
removed completely and deterministically (hard filtering), then frequency-
smoothed weighted resampling is applied to what remains.

Goal: check whether hard removal closes the gap observed in the exploratory
generation-fairness evaluation, compared to probabilistic reweighting. Output
is saved to a separate file so pipeline.py's main results stay untouched.
"""
import json
import math
import os
import random

import pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
KEPT_PATH = os.path.join(OUT_DIR, "kept_records.jsonl")
OUT_PATH = os.path.join(OUT_DIR, "balanced_set_hardfilter.jsonl")
SEED = 42


def main():
    records = []
    with open(KEPT_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    n_total = len(records)

    kept = [r for r in records if not r["bias_flagged"]]
    n_removed = n_total - len(kept)
    print(f"total kept-quality records: {n_total}")
    print(f"bias-flagged (hard-removed): {n_removed} ({n_removed/n_total*100:.3f}%)")
    print(f"remaining after hard filter: {len(kept)}")

    # content-token frequency over the set after hard filtering (for frequency smoothing)
    freq_after = {}
    for r in kept:
        for t in r["tokens"]:
            if p.is_content_token(t):
                freq_after[t] = freq_after.get(t, 0) + 1

    rng = random.Random(SEED)
    n = len(kept)
    weights = [0.0] * n
    for i, r in enumerate(kept):
        toks = [t for t in r["tokens"] if t in freq_after]
        avg_freq = (sum(freq_after[t] for t in toks) / len(toks)) if toks else 1.0
        weights[i] = max(1.0 / (1.0 + math.log1p(avg_freq)), 1e-9)

    # Same target ratio as the main pipeline (90%), but this time relative to
    # the *original* pre-filter set, so the final corpus size stays comparable
    # to the weighted-resampling version.
    target_size = int(n_total * p.BIAS_RESAMPLE_KEEP_RATIO)
    target_size = min(target_size, n)
    keys = [(rng.random() ** (1.0 / weights[i]), i) for i in range(n)]
    keys.sort(reverse=True)
    selected_idx = {i for _, i in keys[:target_size]}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i in selected_idx:
            f.write(json.dumps(kept[i], ensure_ascii=False) + "\n")

    print(f"final hard-filtered balanced set: {len(selected_idx)} records")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
