from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"


# -----------------------------
# Argument Parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train misinformation classifier (BERT-based)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset (CSV or JSON) with 'text' and 'label'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/bert_model"),
        help="Directory to save trained model",
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
        raise ValueError("Only CSV or JSON datasets are supported")

    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Dataset missing columns: {missing}")

    return df


# -----------------------------
# Main Training Logic
# -----------------------------
def main():
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

    # Convert to HuggingFace dataset
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
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save model + tokenizer
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    print(f"\n✅ Model trained and saved to: {args.output}")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
