from __future__ import annotations

import json
from pathlib import Path


def write_markdown_report(output_path, dataset_summary, imputation_stats, best_scaler_row, reduction_row, modeling_result):
    output_path = Path(output_path)
    leaderboard = modeling_result["leaderboard"]
    best = leaderboard.iloc[0]

    text = f"""# Crop Recommendation Research Report

## 1. Problem

This study builds a multi-class crop recommendation system using soil nutrients and weather variables to predict the most suitable crop.

## 2. Dataset Summary

- Rows: {dataset_summary['rows']}
- Numeric input features: {dataset_summary['num_features']}
- Target classes: {dataset_summary['num_classes']}
- Class balance: perfectly balanced across crops in the provided dataset.

## 3. Preprocessing Findings

### Missing-value study

The original dataset contained no missing values. To compare imputation methods rigorously, synthetic missingness was injected and multiple imputers were benchmarked using reconstruction RMSE and downstream macro-F1.

- Best imputation method: `{imputation_stats['best_method']}`
- Friedman test p-value for RMSE comparison: {imputation_stats['friedman_rmse_pvalue']:.6g}
- Friedman test p-value for downstream macro-F1 comparison: {imputation_stats['friedman_f1_pvalue']:.6g}

### Scaling benchmark

- Best scaler-model baseline combination: `{best_scaler_row['scaler']}` with `{best_scaler_row['model']}`
- Macro-F1: {best_scaler_row['macro_f1_mean']:.4f}

### Dimensionality reduction

- Best reduction strategy in baseline comparison: `{reduction_row['reducer']}`
- Macro-F1: {reduction_row['macro_f1_mean']:.4f}

## 4. Modeling Findings

- Best final model: `{best['model']}`
- Test Accuracy: {best['test_accuracy']:.4f}
- Test Macro-F1: {best['test_f1_macro']:.4f}
- Cross-validation Macro-F1: {best['cv_f1_macro_mean']:.4f}
- Bias-variance gap: {best['bias_variance_gap']:.4f}

## 5. Interpretation

Interpretability artifacts were generated with:

- permutation importance
- SHAP global importance
- LIME local explanation

These outputs are saved in the `interpretability/` directory for auditability.

## 6. Conclusion

The selected model was chosen because it produced the strongest balance of hold-out macro-F1, cross-validation stability, and low train-validation gap. This makes it the most reliable deployment candidate among the tested methods.
"""
    output_path.write_text(text, encoding="utf-8")


def write_json_summary(output_path, payload):
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
