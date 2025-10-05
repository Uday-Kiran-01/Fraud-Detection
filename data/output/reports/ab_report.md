# A/B Comparison — Model vs Baseline

- Threshold for models: **0.75**

- Evaluation computed directly from each `preds_test.csv` (uses its `y_true`).

- Baseline included: **No (no aligned row_id)**


## Results

| model    |   accuracy |   precision |   recall |        f1 |   roc_auc |   flag_rate |
|:---------|-----------:|------------:|---------:|----------:|----------:|------------:|
| logistic |   0.974093 |   0.0437481 | 0.644011 | 0.0819306 |  0.901429 |    0.026424 |
| rf       |   0.998704 |   0.908347  | 0.309192 | 0.461347  |  0.91867  |    0.000611 |
| lgbm     |   0.991165 |   0.145733  | 0.806685 | 0.246867  |  0.963436 |    0.009936 |


*Saved:* `data/output/reports/model_comparison.csv`, `model_comparison.png`.
