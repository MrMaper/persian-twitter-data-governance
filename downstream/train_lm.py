"""
ادامه‌آموزش (Continued Pretraining) gpt2-persian روی پیکره خام یا پالایش‌شده (فاز ۳،
پاسخ به داور ۲ درباره ارزیابی downstream). دو مدل با تنظیمات کاملاً یکسان (seed،
تعداد گام، نرخ یادگیری) روی دو پیکره متفاوت آموزش داده می‌شوند تا تنها متغیر مستقل،
کیفیت داده آموزشی باشد.

اجرا:
    python train_lm.py --corpus corpora/train_raw.txt       --out models/raw
    python train_lm.py --corpus corpora/train_governed.txt  --out models/governed
"""
import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # پروتکل xet روی برخی شبکه‌ها هنگ می‌کند

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE_MODEL = "flax-community/gpt2-persian-question-answering"
# یادداشت: bolbolzaban/gpt2-persian تنها به‌صورت pytorch_model.bin (pickle) منتشر شده و
# transformers/torch جدید بارگذاری آن را به دلایل امنیتی (CVE-2025-32434) بدون torch>=2.6
# مسدود می‌کند؛ این مدل جایگزین (بر پایه gpt2-persian اصلی، فاین‌تیون‌شده برای QA) دارای
# فرمت امن‌تر safetensors است.
SEED = 42
BLOCK_SIZE = 128
MAX_STEPS = 3000  # بودجه محاسباتی ثابت و یکسان برای هر دو مدل (کنترل حجم آموزش)


def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_dataset(path, tokenizer):
    texts = load_lines(path)
    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=BLOCK_SIZE)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    set_seed(SEED)
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_ds = build_dataset(args.corpus, tokenizer)
    print(f"loaded {len(train_ds)} training examples from {args.corpus}", flush=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.out,
        max_steps=args.max_steps,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="no",
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("saved model to", args.out)


if __name__ == "__main__":
    main()
