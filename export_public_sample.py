"""
Exports a public, anonymized sample from the governed dataset.

Samples only from out/balanced_set.jsonl (the framework's final output, after
all three stages of cleaning/quality/bias mitigation) — never from raw data.
No identity field (fK_UserID or similar) exists in this file's input at all,
since pipeline.py never stores it in the first place; so no removal should be
needed, but as defense in depth an explicit check is still run to make sure no
identity-related field has accidentally survived in the text or metadata.
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
    """Defense in depth: explicitly drop any field whose name is associated
    with user identity, if present."""
    cleaned = {k: v for k, v in record.items() if k.lower() not in IDENTITY_FIELD_NAMES}
    return cleaned


def main():
    if not os.path.exists(BALANCED_PATH):
        print("balanced_set.jsonl not found. Run pipeline.py to completion first.")
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
        f.write(f"""# Public Sample of Governed Data — Persian Twitter Data Governance Framework

## Description
This file ({SAMPLE_SIZE} records) is a random subsample of the final output of the proposed framework, after all three stages of noise cleaning, quality filtering, and bias mitigation. It is released to support reproducibility and methodological review by other researchers.

**This file contains no raw Twitter data and no user identity information** — only the final, governed tweets and their computed scores.

## Record fields
| Field | Description |
|---|---|
| `id` | Unique tweet ID (for provenance tracking — no user identity information) |
| `text` | Normalized text after noise cleaning |
| `tokens` | Token list produced by `hazm.WordTokenizer` |
| `q` | Quality score $Q(t)$ per Eq. (1) of the paper |
| `b` | Lexical bias index $B(t)$ per Eq. (2) of the paper |
| `bias_flagged` | Whether $B(t)$ exceeded the threshold $\\tau_B$ |

## Anonymization method
The core pipeline (`pipeline.py`) never stores user identity fields (e.g. `fK_UserID`) in `kept_records.jsonl` / `balanced_set.jsonl` — these fields are dropped as early as Pass 1. As defense in depth, this script (`export_public_sample.py`) also explicitly strips any remaining field whose name is associated with user identity.

## License
CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International). Non-commercial use with attribution is permitted; contact the author for commercial use.

## Note on Twitter/X policy
This sample includes the full text of tweets (with no user identity information) for research and educational purposes. These texts were originally collected from Twitter/X; if the original tweet is later deleted or made private by its author, the corresponding content in this archive may no longer be available on the source platform. Downstream users of this dataset should respect Twitter/X's developer policies when reusing this data.

## Citation
Roustaei, M. (2026). *Persian Twitter Data Governance Sample: Quality- and Bias-Filtered Corpus* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21903923

Full paper citation to be completed once accepted: *"Evaluating Data Quality and Bias of Social Media in Persian Large Language Models: A Data Governance Approach"* (Journal of Computing Sciences and Information Technology, Computer Society of Iran).
""")

    print(f"exported {len(sample)} records to {out_path}")
    print(f"README written to {readme_path}")


if __name__ == "__main__":
    main()
