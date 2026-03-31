from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import chi2, friedmanchisquare, kruskal
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .data import FEATURE_COLUMNS
from .transformers import DomainFeatureGenerator


class EMImputer(BaseEstimator, TransformerMixin):
    """Simple Gaussian EM-style imputer for numeric data."""

    def __init__(self, max_iter=30, tol=1e-4, regularization=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        missing_mask = np.isnan(X)
        X_filled = X.copy()
        col_means = np.nanmean(X_filled, axis=0)
        inds = np.where(missing_mask)
        X_filled[inds] = np.take(col_means, inds[1])

        for _ in range(self.max_iter):
            previous = X_filled.copy()
            mean = X_filled.mean(axis=0)
            covariance = np.cov(X_filled, rowvar=False) + self.regularization * np.eye(X_filled.shape[1])

            for row_idx in range(X_filled.shape[0]):
                miss = missing_mask[row_idx]
                obs = ~miss
                if not miss.any():
                    continue
                if obs.sum() == 0:
                    X_filled[row_idx, miss] = mean[miss]
                    continue

                sigma_oo = covariance[np.ix_(obs, obs)]
                sigma_mo = covariance[np.ix_(miss, obs)]
                x_obs = X_filled[row_idx, obs]
                mu_obs = mean[obs]
                mu_miss = mean[miss]

                try:
                    conditional = mu_miss + sigma_mo @ np.linalg.pinv(sigma_oo) @ (x_obs - mu_obs)
                except np.linalg.LinAlgError:
                    conditional = mu_miss
                X_filled[row_idx, miss] = conditional

            delta = np.linalg.norm(X_filled - previous) / (np.linalg.norm(previous) + 1e-9)
            if delta < self.tol:
                break

        self.statistics_ = {
            "mean": X_filled.mean(axis=0),
            "covariance": np.cov(X_filled, rowvar=False) + self.regularization * np.eye(X_filled.shape[1]),
        }
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        missing_mask = np.isnan(X)
        X_filled = X.copy()
        mean = self.statistics_["mean"]
        covariance = self.statistics_["covariance"]

        col_means = np.nanmean(X_filled, axis=0)
        col_means = np.where(np.isnan(col_means), mean, col_means)
        inds = np.where(missing_mask)
        X_filled[inds] = np.take(col_means, inds[1])

        for row_idx in range(X_filled.shape[0]):
            miss = missing_mask[row_idx]
            obs = ~miss
            if not miss.any():
                continue
            if obs.sum() == 0:
                X_filled[row_idx, miss] = mean[miss]
                continue
            sigma_oo = covariance[np.ix_(obs, obs)]
            sigma_mo = covariance[np.ix_(miss, obs)]
            x_obs = X_filled[row_idx, obs]
            mu_obs = mean[obs]
            mu_miss = mean[miss]
            try:
                conditional = mu_miss + sigma_mo @ np.linalg.pinv(sigma_oo) @ (x_obs - mu_obs)
            except np.linalg.LinAlgError:
                conditional = mu_miss
            X_filled[row_idx, miss] = conditional
        return X_filled


def get_imputers():
    return {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "knn": KNNImputer(n_neighbors=5, weights="distance"),
        "iterative_regression": IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            random_state=42,
        ),
        "em_gaussian": EMImputer(max_iter=30),
    }


def benchmark_imputation_methods(X: pd.DataFrame, y: pd.Series, output_dir: str | Path, random_state=42):
    output_dir = Path(output_dir)
    rng = np.random.default_rng(random_state)
    methods = get_imputers()
    repeats = 5
    mask_fraction = 0.08
    baseline_model = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    records = []
    rmse_by_method = {name: [] for name in methods}
    f1_by_method = {name: [] for name in methods}

    X_values = X.to_numpy(dtype=float)
    total_cells = X_values.size
    mask_count = int(total_cells * mask_fraction)

    for repeat in range(repeats):
        mask = np.zeros_like(X_values, dtype=bool)
        flat_indices = rng.choice(total_cells, size=mask_count, replace=False)
        mask.flat[flat_indices] = True

        masked = X_values.copy()
        masked[mask] = np.nan

        for method_name, imputer in methods.items():
            reconstructed = clone(imputer).fit_transform(masked)
            rmse = float(np.sqrt(np.mean((reconstructed[mask] - X_values[mask]) ** 2)))
            pipeline = Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000)),
                ]
            )
            scores = cross_val_score(
                pipeline,
                reconstructed,
                y,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1,
            )
            rmse_by_method[method_name].append(rmse)
            f1_by_method[method_name].append(float(scores.mean()))
            records.append(
                {
                    "repeat": repeat + 1,
                    "method": method_name,
                    "rmse": rmse,
                    "macro_f1": float(scores.mean()),
                }
            )

    table = (
        pd.DataFrame(records)
        .groupby("method", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .sort_values(["macro_f1_mean", "rmse_mean"], ascending=[False, True])
    )
    table.to_csv(output_dir / "imputation_benchmark.csv", index=False)

    friedman_rmse = friedmanchisquare(*[rmse_by_method[name] for name in methods])
    friedman_f1 = friedmanchisquare(*[f1_by_method[name] for name in methods])
    stats_result = {
        "mask_fraction": mask_fraction,
        "repeats": repeats,
        "friedman_rmse_statistic": float(friedman_rmse.statistic),
        "friedman_rmse_pvalue": float(friedman_rmse.pvalue),
        "friedman_f1_statistic": float(friedman_f1.statistic),
        "friedman_f1_pvalue": float(friedman_f1.pvalue),
        "best_method": table.iloc[0]["method"],
    }
    (output_dir / "imputation_stats.json").write_text(json.dumps(stats_result, indent=2), encoding="utf-8")
    return table, stats_result


def detect_outliers(X: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)
    z_scores = np.abs(stats.zscore(X, nan_policy="omit"))
    z_flags = (z_scores > 3).any(axis=1)

    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    iqr_flags = ((X < (q1 - 1.5 * iqr)) | (X > (q3 + 1.5 * iqr))).any(axis=1)

    mean = X.mean().to_numpy()
    covariance = np.cov(X.to_numpy().T) + 1e-6 * np.eye(X.shape[1])
    inverse_cov = np.linalg.pinv(covariance)
    centered = X.to_numpy() - mean
    mahal = np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse_cov, centered))
    threshold = np.sqrt(chi2.ppf(0.997, df=X.shape[1]))
    mahal_flags = mahal > threshold

    outlier_df = X.copy()
    outlier_df["zscore_outlier"] = z_flags
    outlier_df["iqr_outlier"] = iqr_flags
    outlier_df["mahalanobis_distance"] = mahal
    outlier_df["mahalanobis_outlier"] = mahal_flags
    outlier_df.to_csv(output_dir / "outlier_flags.csv", index=False)

    summary = pd.DataFrame(
        [
            {"method": "zscore", "count": int(z_flags.sum()), "percentage": float(z_flags.mean() * 100)},
            {"method": "iqr", "count": int(iqr_flags.sum()), "percentage": float(iqr_flags.mean() * 100)},
            {"method": "mahalanobis", "count": int(mahal_flags.sum()), "percentage": float(mahal_flags.mean() * 100)},
            {
                "method": "intersection_all_three",
                "count": int((z_flags & iqr_flags & mahal_flags).sum()),
                "percentage": float((z_flags & iqr_flags & mahal_flags).mean() * 100),
            },
        ]
    )
    summary.to_csv(output_dir / "outlier_summary.csv", index=False)
    return summary


def correlation_and_covariance_analysis(df: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)
    pearson = df[FEATURE_COLUMNS].corr(method="pearson")
    spearman = df[FEATURE_COLUMNS].corr(method="spearman")
    covariance = df[FEATURE_COLUMNS].cov()

    pearson.to_csv(output_dir / "pearson_correlation.csv")
    spearman.to_csv(output_dir / "spearman_correlation.csv")
    covariance.to_csv(output_dir / "covariance_matrix.csv")

    vif_data = []
    X_values = df[FEATURE_COLUMNS].astype(float).to_numpy()
    for idx, feature in enumerate(FEATURE_COLUMNS):
        vif_data.append({"feature": feature, "vif": float(variance_inflation_factor(X_values, idx))})
    vif_df = pd.DataFrame(vif_data).sort_values("vif", ascending=False)
    vif_df.to_csv(output_dir / "vif.csv", index=False)

    hypothesis_rows = []
    for feature in FEATURE_COLUMNS:
        groups = [group[feature].to_numpy() for _, group in df.groupby("label")]
        anova_stat, anova_p = stats.f_oneway(*groups)
        kw_stat, kw_p = kruskal(*groups)
        hypothesis_rows.append(
            {
                "feature": feature,
                "anova_f_stat": float(anova_stat),
                "anova_p_value": float(anova_p),
                "kruskal_stat": float(kw_stat),
                "kruskal_p_value": float(kw_p),
            }
        )
    hypothesis_df = pd.DataFrame(hypothesis_rows).sort_values("anova_p_value")
    hypothesis_df.to_csv(output_dir / "hypothesis_tests.csv", index=False)

    return {
        "pearson": pearson,
        "spearman": spearman,
        "covariance": covariance,
        "vif": vif_df,
        "hypothesis_tests": hypothesis_df,
    }


def benchmark_scalers(X: pd.DataFrame, y: pd.Series, output_dir: str | Path):
    output_dir = Path(output_dir)
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "knn": __import__("sklearn.neighbors").neighbors.KNeighborsClassifier(n_neighbors=5),
        "svm": __import__("sklearn.svm").svm.SVC(kernel="rbf", probability=True),
        "mlp": __import__("sklearn.neural_network").neural_network.MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for scaler_name, scaler in scalers.items():
        for model_name, model in models.items():
            pipe = Pipeline(
                steps=[
                    ("domain", DomainFeatureGenerator()),
                    ("scale", scaler),
                    ("model", model),
                ]
            )
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
            rows.append(
                {
                    "scaler": scaler_name,
                    "model": model_name,
                    "macro_f1_mean": float(scores.mean()),
                    "macro_f1_std": float(scores.std()),
                }
            )

    result = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    result.to_csv(output_dir / "scaler_benchmark.csv", index=False)
    return result


def benchmark_dimensionality_reduction(X: pd.DataFrame, y: pd.Series, output_dir: str | Path):
    output_dir = Path(output_dir)
    transformed = DomainFeatureGenerator().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    reducers = {
        "none": "passthrough",
        "pca_95": PCA(n_components=0.95, random_state=42),
        "lda": LinearDiscriminantAnalysis(n_components=min(transformed.shape[1], y.nunique() - 1)),
    }
    rows = []

    for name, reducer in reducers.items():
        pipe = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("reduce", reducer),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        )
        scores = cross_val_score(pipe, transformed, y, cv=cv, scoring="f1_macro", n_jobs=1)
        rows.append(
            {
                "reducer": name,
                "macro_f1_mean": float(scores.mean()),
                "macro_f1_std": float(scores.std()),
            }
        )

    pca = PCA().fit(StandardScaler().fit_transform(transformed))
    explained = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    result = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    result.to_csv(output_dir / "dimensionality_reduction_benchmark.csv", index=False)
    return result, explained
