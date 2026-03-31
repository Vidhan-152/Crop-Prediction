from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path("data/demo_crop_recommendation.csv")


def build_crop_profiles():
    return {
        "rice": {
            "N": 88,
            "P": 40,
            "K": 40,
            "temperature": 27,
            "humidity": 84,
            "ph": 6.2,
            "rainfall": 220,
            "soil_type": "clayey",
            "season": "kharif",
            "region": "coastal",
        },
        "maize": {
            "N": 78,
            "P": 48,
            "K": 36,
            "temperature": 24,
            "humidity": 65,
            "ph": 6.4,
            "rainfall": 90,
            "soil_type": "loamy",
            "season": "kharif",
            "region": "plains",
        },
        "cotton": {
            "N": 115,
            "P": 46,
            "K": 20,
            "temperature": 29,
            "humidity": 62,
            "ph": 6.8,
            "rainfall": 75,
            "soil_type": "black",
            "season": "kharif",
            "region": "deccan",
        },
        "wheat": {
            "N": 72,
            "P": 46,
            "K": 32,
            "temperature": 20,
            "humidity": 55,
            "ph": 6.7,
            "rainfall": 65,
            "soil_type": "alluvial",
            "season": "rabi",
            "region": "north",
        },
        "chickpea": {
            "N": 38,
            "P": 68,
            "K": 78,
            "temperature": 22,
            "humidity": 42,
            "ph": 7.1,
            "rainfall": 55,
            "soil_type": "sandy",
            "season": "rabi",
            "region": "semi-arid",
        },
        "banana": {
            "N": 102,
            "P": 78,
            "K": 52,
            "temperature": 28,
            "humidity": 80,
            "ph": 6.0,
            "rainfall": 150,
            "soil_type": "loamy",
            "season": "zaid",
            "region": "tropical",
        },
        "coffee": {
            "N": 98,
            "P": 26,
            "K": 30,
            "temperature": 23,
            "humidity": 72,
            "ph": 6.1,
            "rainfall": 190,
            "soil_type": "laterite",
            "season": "kharif",
            "region": "hills",
        },
        "apple": {
            "N": 24,
            "P": 134,
            "K": 198,
            "temperature": 17,
            "humidity": 88,
            "ph": 6.5,
            "rainfall": 115,
            "soil_type": "silty",
            "season": "temperate",
            "region": "hills",
        },
    }


def sample_row(rng, crop_name, profile):
    return {
        "N": max(0, rng.normal(profile["N"], 10)),
        "P": max(0, rng.normal(profile["P"], 10)),
        "K": max(0, rng.normal(profile["K"], 12)),
        "temperature": rng.normal(profile["temperature"], 1.8),
        "humidity": np.clip(rng.normal(profile["humidity"], 6), 20, 100),
        "ph": np.clip(rng.normal(profile["ph"], 0.35), 3.5, 9.0),
        "rainfall": max(0, rng.normal(profile["rainfall"], 20)),
        "soil_type": profile["soil_type"],
        "season": profile["season"],
        "region": profile["region"],
        "crop": crop_name,
    }


def main():
    rng = np.random.default_rng(42)
    profiles = build_crop_profiles()

    rows = []
    for crop_name, profile in profiles.items():
        for _ in range(120):
            rows.append(sample_row(rng, crop_name, profile))

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved demo dataset to {OUTPUT_PATH} with {len(df)} rows.")


if __name__ == "__main__":
    main()
