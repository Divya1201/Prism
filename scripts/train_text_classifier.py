from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from services.text_analysis import DEFAULT_MODEL_PATH, build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fake-news text classifier")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to CSV dataset containing 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to save trained model (default: {DEFAULT_MODEL_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.dataset)
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    texts = df["text"].fillna("").astype(str)
    labels = df["label"].astype(str)

    pipeline = build_pipeline()
    pipeline.fit(texts, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)

    print(f"Model trained and saved to: {args.output}")


if __name__ == "__main__":
    main()
