import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def infer_feature_types(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(X):
    numeric_features, categorical_features = infer_feature_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def get_model_specs():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=1,
        ),
        "SVM": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=600,
            random_state=42,
        ),
    }


def build_pipeline(preprocessor, estimator):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def compare_models(X, y, test_size=0.2, cv_folds=5):
    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    model_specs = get_model_specs()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    rows = []
    best_result = None

    for model_name, estimator in model_specs.items():
        pipeline = build_pipeline(preprocessor, estimator)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        pipeline.fit(X_train, y_train)
        holdout_pred = pipeline.predict(X_test)
        holdout_f1 = f1_score(y_test, holdout_pred, average="macro")

        result = {
            "model_name": model_name,
            "pipeline": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            "cv_f1_macro_std": float(np.std(scores["test_f1_macro"])),
            "cv_precision_macro_mean": float(np.mean(scores["test_precision_macro"])),
            "cv_recall_macro_mean": float(np.mean(scores["test_recall_macro"])),
            "holdout_f1_macro": float(holdout_f1),
        }
        rows.append(
            {
                "model": model_name,
                "cv_accuracy_mean": result["cv_accuracy_mean"],
                "cv_accuracy_std": result["cv_accuracy_std"],
                "cv_f1_macro_mean": result["cv_f1_macro_mean"],
                "cv_f1_macro_std": result["cv_f1_macro_std"],
                "cv_precision_macro_mean": result["cv_precision_macro_mean"],
                "cv_recall_macro_mean": result["cv_recall_macro_mean"],
                "holdout_f1_macro": result["holdout_f1_macro"],
            }
        )

        if best_result is None or result["cv_f1_macro_mean"] > best_result["cv_f1_macro_mean"]:
            best_result = result

    leaderboard_df = pd.DataFrame(rows).sort_values(
        by=["cv_f1_macro_mean", "holdout_f1_macro"],
        ascending=False,
    )

    feature_summary = {
        "all_features": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    return leaderboard_df, best_result, feature_summary


def evaluate_pipeline(best_result):
    pipeline = best_result["pipeline"]
    X_test = best_result["X_test"]
    y_test = best_result["y_test"]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    return metrics, y_test, y_pred


def write_classification_report(output_path, y_true, y_pred):
    report = classification_report(y_true, y_pred, zero_division=0)
    Path(output_path).write_text(report, encoding="utf-8")


def build_training_summary(args, dataset, feature_summary, best_result, leaderboard, model_path):
    return {
        "data_path": args.data,
        "target": args.target,
        "rows": int(len(dataset)),
        "columns": dataset.columns.tolist(),
        "feature_summary": feature_summary,
        "cv_folds": args.cv_folds,
        "test_size": args.test_size,
        "best_model": best_result["model_name"],
        "best_model_path": str(model_path),
        "leaderboard": json.loads(leaderboard.to_json(orient="records")),
    }
