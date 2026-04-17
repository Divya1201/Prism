from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

label_map = {
    "fabricated": 0,
    "satire": 1,
    "false_context": 2,
    "false_connection": 3,
    "imposter": 4,
    "manipulated": 5,
    "astroturfing": 6,
    "sponsored": 7,
    "unknown": 8,
}

MODEL_NAME = "distilbert-base-uncased"


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Argument Parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train misinformation classifier (BERT-based)"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/bert_model"),
    )
    return parser.parse_args()


# -----------------------------
# Dataset Loader
# -----------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("Only CSV or JSON supported")

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'label'")

    return df


# -----------------------------
# Metrics (important)
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = (preds == labels).mean()

    return {
        "accuracy": accuracy,
    }


# -----------------------------
# Main Training Logic
# -----------------------------
def main():
    set_seed()

    args = parse_args()
    df = load_dataset(args.dataset)

    # Clean data
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str)

    # Label encoding
    labels = sorted(df["label"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    df["label_id"] = df["label"].map(label2id)

    # Convert dataset
    dataset = Dataset.from_pandas(df[["text", "label_id"]])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    dataset = dataset.map(tokenize, batched=True)

    # Train-test split
    dataset = dataset.train_test_split(test_size=0.1)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    print("\n📊 Evaluation:", metrics)

    # Save model
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    # 🔥 Save label mapping (VERY IMPORTANT)
    with open(args.output / "labels.json", "w") as f:
        json.dump(label2id, f)

    print(f"\n✅ Model + labels saved to: {args.output}")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
