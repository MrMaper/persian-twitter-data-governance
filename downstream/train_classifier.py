"""
Fine-tunes a classification head (3 classes: negative/neutral/positive) on top
of each of the two continually pretrained models (RAW/GOVERNED), using the
ParsiNLU sentiment dataset (food + movie, "overall" subset), and compares
accuracy on the test set.

Usage:
    python train_classifier.py --model models/raw       --out clf/raw
    python train_classifier.py --model models/governed  --out clf/governed
"""
import argparse
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the xet protocol hangs on some networks

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "sentiment_data")
SEED = 42
MAX_LEN = 128


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_dataset(rows):
    return Dataset.from_dict({
        "text": [r["text"] for r in rows],
        "label": [r["label"] for r in rows],
    })


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(SEED)
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_rows = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    dev_rows = load_jsonl(os.path.join(DATA_DIR, "dev.jsonl"))
    test_rows = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")

    train_ds = to_dataset(train_rows).map(tok_fn, batched=True)
    dev_ds = to_dataset(dev_rows).map(tok_fn, batched=True)
    test_ds = to_dataset(test_rows).map(tok_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no",
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_metrics = trainer.evaluate(test_ds)
    print("TEST METRICS:", test_metrics, flush=True)

    with open(os.path.join(args.out, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
