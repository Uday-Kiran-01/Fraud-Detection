# 🚀 Ad Click Fraud Detection

End-to-end ML pipeline to detect fraudulent ad clicks.  
Includes EDA, feature engineering, model training (Logistic, Random Forest, LightGBM), batch/stream inference, A/B comparison, threshold sweep, and reporting.

---

## ✨ What this does

- **EDA**: explore click patterns and volumes  
- **Features**: build supervised features from raw clicks  
- **Train**: fit Logistic, RF, LGBM and save metrics + predictions  
- **Infer**: score new data in batch or simulated stream  
- **A/B**: compare models head-to-head (and optional rules)  
- **Threshold sweep**: pick an operating point by F1 or flag budget  
- **Report**: daily Markdown report + charts

---

## 📦 Repo structure
Fraud Detection/
├── configs/
│ └── config.yaml
├── data/
│ ├── input/
│ │ └── train_sample.csv # source data (TalkingData-style)
│ └── output/
│ ├── eda/
│ ├── features/
│ ├── models/
│ │ ├── logistic/
│ │ ├── rf/
│ │ └── lgbm/
│ ├── reports/ # ab_report.md, model_comparison., threshold_.png/csv, ...
│ └── stream/
├── src/
│ ├── eda.py ├── infer.py ├── report.py
│ ├── features.py ├── stream_simulator.py
│ ├── train.py └── evaluation/
│ ├── ab_test.py
│ └── threshold_sweep.py
├── tests/
│ ├── conftest.py
│ └── test_features.py
├── main.py
└── requirements.txt

## 🛠️ Setup

```powershell
# from repo root (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

⚙️ Configuration

Key knobs in configs/config.yaml (example):

paths:
  input_csv: data/input/train_sample.csv
  out_dir: data/output

training:
  model_type: lgbm  # or rf / logistic

evaluation:
  threshold: 0.75   # final operating point (≈1% flag rate)

baseline:
  clicks_per_ip_per_5min: 50
  unique_channels_per_5min: 40


We ship LightGBM @ threshold=0.75 as default (≈1% flags).

🚦 Quickstart (common commands)
# 1) EDA
python .\main.py eda

# 2) Feature engineering
python .\main.py featurize

# 3) Train a model (choose one)
python .\main.py train --model logistic
python .\main.py train --model rf
python .\main.py train --model lgbm

# 4) Batch inference (LightGBM @ 0.75 or choose your own threshold)
python .\main.py infer --model lgbm --threshold 0.75
# or by budget (e.g., 1%)
python .\main.py infer --model lgbm --flag-rate 0.01

# 5) Simulated streaming
python .\main.py simulate --model lgbm --rate 50 --batch 5000 --max-rows 200000

# 6) Daily report
python .\main.py report

# 7) A/B comparison (reads each model’s preds_test.csv)
python .\main.py abtest

# 8) Threshold sweep for LightGBM (creates plots + CSV)
python -m src.evaluation.threshold_sweep

📈 Current evaluation (from abtest)

Threshold = 0.5 comparison (auto-generated):

Model	Accuracy	Precision	Recall	F1	ROC AUC	Flag Rate
Logistic	0.9741	0.0437	0.6440	0.0819	0.9014	2.6424%
RF	0.9987	0.9083	0.3092	0.4613	0.9187	0.0611%
LightGBM	0.9912	0.1457	0.8067	0.2469	0.9634	0.9936%

Operating point selected: LightGBM @ threshold = 0.75 (≈1% flag budget)
(from threshold_sweep_lgbm.csv)

Metric	Value
Precision	0.145733
Recall	0.806685
F1	0.246867
Flag Rate	0.9936%

Rationale: Best overall ranking (ROC AUC) with high recall and manageable review volume.

Artifacts live in data/output/reports/:

ab_report.md, model_comparison.csv, model_comparison.png

threshold_sweep_lgbm.csv, threshold_f1_lgbm.png, threshold_pr_lgbm.png, threshold_flagrate_lgbm.png

Daily report.md and plots

🧪 Tests
pytest


We include a small thresholding test and a tests/conftest.py that adds the repo root to PYTHONPATH.

🧩 Baseline rules (optional in A/B)

If you want the baseline row in abtest, export a shared row_id in each model’s preds_test.csv:

# when saving predictions
preds = pd.DataFrame({
  "row_id": df_test.reset_index().index,  # or your true click_id
  "y_true": y_true,
  "y_score": y_proba
})
preds.to_csv(model_dir / "preds_test.csv", index=False)


Then:

python .\main.py abtest

🧭 Model choice: RF vs LGBM

RF at 0.5 was ultra-precise with tiny flag rate (good for strict triage).

LGBM has the best ROC AUC (ranking). With threshold tuning (0.75), it achieves high recall at ~1% flags.
Default: LightGBM @ 0.75. Keep RF as a “precision mode” if needed.

📚 Train/Valid/Test split (recommended)

For demos/internal use, current results are fine.
For production/rigor:

Use time-based splits (train on earlier days, validate on middle, test on latest) to reduce leakage.

Or perform k-fold CV on sessions/users/IPs (avoid row-level leakage).

Configure via configs/config.yaml or add CLI args in train.py.

🛡️ Model card & compliance

See MODEL_CARD.md for data, metrics, intended use, limitations, guardrails, and privacy notes.
We avoid PII; identifiers are technical (ip, device, channel). Hash/anonymize in reports as needed.

🗺️ Roadmap

Add baseline rules to A/B (shared row_id)

PR/ROC plots for LGBM (archive in reports)

Weekly human-review sample to estimate live precision

Drift monitoring (score distribution + top features)

CI: run pytest on PRs


---

If you want, I can also add a tiny **PR/ROC plotting** helper and a **time-based split** option next.

> 💡 Due to local hardware limits, this version uses a single holdout validation.  
> For larger compute setups, I can easily extend this to time-based splits or full k-fold cross-validation for deeper statistical rigor.
