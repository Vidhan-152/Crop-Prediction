from pathlib import Path

import pandas as pd


def load_dataset(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV datasets are supported.")
    return pd.read_csv(path)


def split_features_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"
        )

    cleaned_df = df.drop_duplicates().copy()
    cleaned_df = cleaned_df.dropna(subset=[target_column])

    X = cleaned_df.drop(columns=[target_column])
    y = cleaned_df[target_column].astype(str)
    return X, y
