from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the best crop using the saved research model.")
    parser.add_argument("--model-path", default="artifacts/research_run/best_model/best_pipeline.joblib")
    parser.add_argument("--N", type=float, required=True)
    parser.add_argument("--P", type=float, required=True)
    parser.add_argument("--K", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument("--rainfall", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run run_research_pipeline.py first.")

    model = joblib.load(model_path)
    X = pd.DataFrame(
        [
            {
                "N": args.N,
                "P": args.P,
                "K": args.K,
                "temperature": args.temperature,
                "humidity": args.humidity,
                "ph": args.ph,
                "rainfall": args.rainfall,
            }
        ]
    )
    prediction = model.predict(X)[0]
    print(f"Recommended crop: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_
        top3 = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:3]
        print("Top 3 candidates:")
        for name, prob in top3:
            print(f"  {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
