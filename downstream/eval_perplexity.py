"""
Measures a model's perplexity on a shared validation set (fully separate from
both the RAW and GOVERNED training corpora).

Usage:
    python eval_perplexity.py --model models/raw
    python eval_perplexity.py --model models/governed
"""
import argparse
import math
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the xet protocol hangs on some networks

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_PATH = os.path.join(HERE, "corpora", "validation.txt")
BLOCK_SIZE = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", default=VAL_PATH)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    with open(args.val, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_loss = 0.0
    total_tokens = 0
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=BLOCK_SIZE,
                             padding=True, return_tensors="pt").to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels)
            n_tok = (labels != -100).sum().item()
            total_loss += out.loss.item() * n_tok
            total_tokens += n_tok
            if i % (batch_size * 50) == 0:
                print(f"  processed {i}/{len(lines)} lines...", flush=True)

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss)
    print(f"model={args.model} tokens={total_tokens} mean_loss={mean_loss:.4f} perplexity={ppl:.3f}")
    return ppl


if __name__ == "__main__":
    main()
