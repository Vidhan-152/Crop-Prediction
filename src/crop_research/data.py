from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TARGET_COLUMN = "label"
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing_columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return df.copy()


def get_feature_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(str).copy()
    return X, y


def build_dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "num_features": len(FEATURE_COLUMNS),
        "num_classes": int(df[TARGET_COLUMN].nunique()),
        "class_distribution": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "describe": json.loads(df[FEATURE_COLUMNS].describe().round(4).to_json()),
    }


def save_dataset_profile(df: pd.DataFrame, output_path: str | Path) -> dict:
    output_path = Path(output_path)
    profile = {}
    for col in FEATURE_COLUMNS:
        profile[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
        }
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile
