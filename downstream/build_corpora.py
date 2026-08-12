"""
Builds two text corpora for the downstream evaluation:

- RAW: raw tweet text, before passing through the proposed framework.
- GOVERNED: final text after all three stages (cleaning, quality filtering,
  bias mitigation) — exactly out/balanced_set.jsonl as produced by pipeline.py.

Both corpora are sampled to the same size (by record count) so the perplexity
comparison is fair (training-data volume is controlled), and any difference in
outcome comes purely from text quality/cleanliness, not from a volume
difference.

A validation set with no overlap with either training corpus is also held out
from a separate slice of balanced_set.jsonl.
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

VAL_SIZE = 20000  # size of the shared validation set (fully separate from both training corpora)


def stream_raw_texts(path, limit=None):
    """Streams raw text from the source JSON file (no processing at all) — for the RAW corpus."""
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
        print("balanced_set.jsonl not found. Run pipeline.py to completion first.")
        return

    governed = load_governed_records(BALANCED_PATH)
    rng = random.Random(SEED)
    rng.shuffle(governed)

    val_records = governed[:VAL_SIZE]
    train_governed = governed[VAL_SIZE:]
    target_train_size = len(train_governed)

    print(f"governed total={len(governed)} val={len(val_records)} train_governed={target_train_size}")

    # RAW corpus: sample the same number of records as train_governed, from raw
    # text with no filtering at all. Since RAW may include records that also
    # survived into GOVERNED, we simply draw an independent, equal-size random
    # sample from the whole raw file (not just the excluded subset) — because
    # the goal is comparing "ordinary raw text" against "governed text," not
    # producing two mutually exclusive sets.
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
