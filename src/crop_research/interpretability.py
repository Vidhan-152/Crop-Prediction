from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance


def generate_interpretability_outputs(best_model, X_train, X_test, y_test, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_permutation_importance(best_model, X_test, y_test, output_dir)
    save_shap_outputs(best_model, X_train, X_test, output_dir)
    save_lime_output(best_model, X_train, X_test, output_dir)


def save_permutation_importance(best_model, X_test, y_test, output_dir: Path):
    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=15,
        random_state=42,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output_dir / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"], importance_df["importance_mean"], xerr=importance_df["importance_std"], color="#4C956C")
    ax.invert_yaxis()
    ax.set_title("Permutation Importance on Test Set")
    ax.set_xlabel("Mean macro-F1 drop")
    fig.tight_layout()
    fig.savefig(output_dir / "permutation_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_shap_outputs(best_model, X_train, X_test, output_dir: Path):
    background = X_train.sample(min(120, len(X_train)), random_state=42)
    sample = X_test.sample(min(40, len(X_test)), random_state=42)

    explainer = shap.Explainer(best_model.predict_proba, background)
    explanation = explainer(sample)

    values = explanation.values
    if values.ndim == 3:
        mean_abs = np.mean(np.abs(values), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(values), axis=0)

    shap_df = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    shap_df.to_csv(output_dir / "shap_global_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(shap_df["feature"], shap_df["mean_abs_shap"], color="#3D5A80")
    ax.invert_yaxis()
    ax.set_title("Mean Absolute SHAP Value")
    fig.tight_layout()
    fig.savefig(output_dir / "shap_global_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_lime_output(best_model, X_train, X_test, output_dir: Path):
    class_names = list(best_model.classes_) if hasattr(best_model, "classes_") else None
    explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=list(X_train.columns),
        class_names=class_names,
        mode="classification",
        random_state=42,
    )
    instance = X_test.iloc[0]
    explanation = explainer.explain_instance(instance.to_numpy(), best_model.predict_proba, num_features=len(X_train.columns))
    explanation.save_to_file(str(output_dir / "lime_explanation.html"))

    lime_df = pd.DataFrame(explanation.as_list(), columns=["feature_condition", "weight"])
    lime_df.to_csv(output_dir / "lime_explanation.csv", index=False)
