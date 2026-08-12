"""
Downloads the "overall" subset of the ParsiNLU sentiment dataset (food +
movie) directly from the official persiannlp/parsinlu GitHub repository (the
HF `datasets` library no longer supports this dataset's script-based loader,
so the JSONL files are downloaded directly). The 7-point labels (-3 to 3) are
mapped to three classes: negative/neutral/positive.
"""
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "sentiment_data")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/persiannlp/parsinlu/master/data/sentiment-analysis/"
FILES = {
    "food_train": "food_train.jsonl",
    "food_dev": "food_dev.jsonl",
    "food_test": "food_test.jsonl",
    "movie_train": "movie_train.jsonl",
    "movie_dev": "movie_dev.jsonl",
    "movie_test": "movie_test.jsonl",
}


def label_to_class(label):
    v = int(label)
    if v <= -1:
        return 0  # negative
    if v == 0:
        return 1  # neutral
    return 2  # positive


def download(name, fname):
    dest = os.path.join(OUT_DIR, fname)
    if not os.path.exists(dest):
        print(f"downloading {fname}...", flush=True)
        urllib.request.urlretrieve(BASE_URL + fname, dest)
    return dest


def load_overall(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("aspect") == "کلی":
                rows.append({"text": r["review"], "label": label_to_class(r["label"])})
    return rows


def build_split(split):
    rows = []
    for domain in ["food", "movie"]:
        path = download(f"{domain}_{split}", FILES[f"{domain}_{split}"])
        rows.extend(load_overall(path))
    return rows


def main():
    for split in ["train", "dev", "test"]:
        rows = build_split(split)
        out_path = os.path.join(OUT_DIR, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples -> {out_path}")


if __name__ == "__main__":
    main()
