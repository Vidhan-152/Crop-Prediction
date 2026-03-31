from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """Add domain-inspired agronomic features while preserving DataFrame semantics."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                X,
                columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"][: X.shape[1]],
            )

        X_out = X.copy()
        epsilon = 1e-6

        X_out["npk_sum"] = X_out["N"] + X_out["P"] + X_out["K"]
        X_out["npk_balance"] = X_out[["N", "P", "K"]].std(axis=1)
        X_out["rainfall_temperature_ratio"] = X_out["rainfall"] / (X_out["temperature"] + epsilon)
        X_out["humidity_temperature_interaction"] = X_out["humidity"] * X_out["temperature"]
        X_out["ph_distance_neutral"] = (X_out["ph"] - 7.0).abs()
        X_out["nutrient_density"] = X_out["npk_sum"] / (X_out["rainfall"] + 1.0)
        X_out["climate_stress_index"] = (X_out["temperature"] * X_out["humidity"]) / (X_out["rainfall"] + 1.0)

        return X_out


class MahalanobisOutlierFlagger:
    def __init__(self, threshold_quantile=0.997):
        self.threshold_quantile = threshold_quantile

