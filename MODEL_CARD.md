# Model Card — Ad Click Fraud Detection (v1.0)

**Date:** 2025-10-05  
**Owner:** Uday Kiran  
**Repository:** Fraud Detection  
**Models evaluated:** Logistic Regression, Random Forest, LightGBM

## 1) Overview
Binary classification to detect fraudulent ad clicks from clickstream data. The pipeline ingests raw CSV, engineers features, trains models, supports batch/stream inference, and produces daily reports. This card summarizes offline evaluation and the chosen operating threshold.

## 2) Data
- **Source:** `data/input/train_sample.csv`
- **Features:** ip, app, device, channel, click_time (+ engineered temporal/agg features)
- **Label:** `is_attributed` (1 = fraud)
- **Splits:** internal train/test from `src/train.py`

## 3) Intended Use
- Prioritize suspicious IPs/devices/channels for analyst review.
- Optional auto-blocking under guardrails once calibrated.
- Batch (`src/infer.py`) and streaming (`src/stream_simulator.py`) modes.

## 4) Model Comparison (threshold = 0.5)
(From `python main.py abtest`)

| Model    | Accuracy | Precision | Recall | F1     | ROC AUC | Flag Rate |
|----------|----------|-----------|--------|--------|---------|-----------|
| Logistic | 0.9741   | 0.0437    | 0.6440 | 0.0819 | 0.9014  | 2.6424%   |
| RF       | 0.9987   | 0.9083    | 0.3092 | 0.4613 | 0.9187  | 0.0611%   |
| LightGBM | 0.9912   | 0.1457    | 0.8067 | 0.2469 | **0.9634** | 0.9936%   |

**Takeaway:** LightGBM is the strongest ranker (highest ROC AUC). RF is ultra-precise but low-recall at this threshold.

## 5) Final Operating Point (LightGBM @ threshold = 0.75)
Chosen to meet ~**1%** review budget.

| Metric     | Value    |
|-----------:|---------:|
| Precision  | 0.145733 |
| Recall     | 0.806685 |
| F1         | 0.246867 |
| Flag Rate  | 0.9936%  |

Artifacts: `data/output/reports/threshold_sweep_lgbm.csv`, `threshold_*_lgbm.png`, and A/B reports in `data/output/reports/`.

## 6) Risks & Limitations
- Legitimate high-burst campaigns can trigger false positives.
- Label and traffic drift require periodic threshold recalibration.
- IPs can be dynamic/CGNAT; combine with device/channel signals.

## 7) Guardrails & Monitoring
- Guardrails: daily flag-rate cap (≤1%), max auto-blocks/day, instant rollback.
- Monitoring: score distribution, top features, weekly drift.
- Rollout: shadow → 5% canary → 50/50 → full.

## 8) Privacy / GDPR
- Uses non-PII technical identifiers (ip, device, channel).
- Support retention controls and hashed identifiers in reports.
- No geolocation or personal profiling.

## 9) Reproducibility
- Configured in `configs/config.yaml` (`evaluation.threshold: 0.75`).
- Deterministic seeds where applicable.
- One-command eval: `python main.py abtest` → artifacts in `data/output/reports/`.

## 10) Next Steps
- (Optional) Include a **baseline rules** row by exporting a shared `row_id` in each `preds_test.csv`, then rerun `abtest`.
- Add PR/ROC plots for LightGBM to archive operating-point tradeoffs.
- Add a weekly human-review sample (e.g., n=200 flags) to estimate live precision.
