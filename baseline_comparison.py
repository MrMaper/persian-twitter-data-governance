"""
Compares three preprocessing backends on the same sample and the same Q(t)/B(t)
formula: only the normalization/tokenization step changes; everything else
(thresholds, coefficients, bias lexicon) is held fixed, so the comparison is a
true apples-to-apples test.

(a) Regex/Naive: whitespace split, no Arabic/Persian character normalization.
(b) Hazm: Normalizer + WordTokenizer (what the main pipeline.py uses).
(c) Parsivar: Normalizer + Tokenizer.

Virastar (a Ruby tool) is not available in this project's Python ecosystem and
is explicitly noted as a limitation in the paper.
"""
import json
import os
import re
import time

import pipeline as p
from hazm import Normalizer as HazmNormalizer, WordTokenizer as HazmTokenizer
from parsivar import Normalizer as ParsivarNormalizer, Tokenizer as ParsivarTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "out", "baseline_comparison.json")

_hazm_n = HazmNormalizer()
_hazm_t = HazmTokenizer()
_parsivar_n = ParsivarNormalizer(statistical_space_correction=False)
_parsivar_t = ParsivarTokenizer()


def backend_regex(text):
    cleaned, n_noise = p.strip_noise(text)
    tokens = cleaned.split()
    return cleaned, tokens, n_noise


def backend_hazm(text):
    cleaned, n_noise = p.strip_noise(text)
    normalized = _hazm_n.normalize(cleaned)
    tokens = _hazm_t.tokenize(normalized)
    return normalized, tokens, n_noise


def backend_parsivar(text):
    cleaned, n_noise = p.strip_noise(text)
    normalized = _parsivar_n.normalize(cleaned)
    tokens = _parsivar_t.tokenize_words(normalized)
    return normalized, tokens, n_noise


BACKENDS = {
    "Regex/Naive": backend_regex,
    "Hazm": backend_hazm,
    "Parsivar": backend_parsivar,
}


def evaluate_backend(name, fn, data_path):
    total = 0
    kept = 0
    q_sum = 0.0
    valid_tok_sum = 0
    t0 = time.time()
    for raw in p.stream_objects(data_path):
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        text = rec.get("text", "") or ""
        if not text:
            continue
        total += 1
        raw_token_count = max(len(text.split()), 1)
        normalized, tokens, n_noise = fn(text)
        q = p.quality_score(tokens, n_noise, raw_token_count, normalized)
        q_sum += q
        valid_tok_sum += len([t for t in tokens if p.is_content_token(t)])
        if q >= p.TAU_Q:
            kept += 1
    elapsed = time.time() - t0
    return {
        "backend": name,
        "total": total,
        "mean_Q": round(q_sum / max(total, 1), 4),
        "keep_ratio": round(kept / max(total, 1), 4),
        "mean_valid_tokens": round(valid_tok_sum / max(total, 1), 2),
        "elapsed_seconds": round(elapsed, 1),
        "records_per_second": round(total / max(elapsed, 1e-9), 1),
    }


def main(data_path):
    rows = []
    for name, fn in BACKENDS.items():
        print(f"--- evaluating backend: {name} ---", flush=True)
        row = evaluate_backend(name, fn, data_path)
        rows.append(row)
        print(row, flush=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else p.DATA_PATH
    main(data_path)
