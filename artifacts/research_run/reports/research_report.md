# Crop Recommendation Research Report

## 1. Problem

This study builds a multi-class crop recommendation system using soil nutrients and weather variables to predict the most suitable crop.

## 2. Dataset Summary

- Rows: 2200
- Numeric input features: 7
- Target classes: 22
- Class balance: perfectly balanced across crops in the provided dataset.

## 3. Preprocessing Findings

### Missing-value study

The original dataset contained no missing values. To compare imputation methods rigorously, synthetic missingness was injected and multiple imputers were benchmarked using reconstruction RMSE and downstream macro-F1.

- Best imputation method: `knn`
- Friedman test p-value for RMSE comparison: 0.000771849
- Friedman test p-value for downstream macro-F1 comparison: 0.00656719

### Scaling benchmark

- Best scaler-model baseline combination: `robust` with `mlp`
- Macro-F1: 0.9877

### Dimensionality reduction

- Best reduction strategy in baseline comparison: `lda`
- Macro-F1: 0.9840

## 4. Modeling Findings

- Best final model: `SoftVoting`
- Test Accuracy: 0.9932
- Test Macro-F1: 0.9932
- Cross-validation Macro-F1: 0.9909
- Bias-variance gap: 0.0034

## 5. Interpretation

Interpretability artifacts were generated with:

- permutation importance
- SHAP global importance
- LIME local explanation

These outputs are saved in the `interpretability/` directory for auditability.

## 6. Conclusion

The selected model was chosen because it produced the strongest balance of hold-out macro-F1, cross-validation stability, and low train-validation gap. This makes it the most reliable deployment candidate among the tested methods.
