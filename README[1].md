# E-Commerce Fraud Detection with Cost-Sensitive Threshold Optimization

## Overview

This project detects fraudulent e-commerce transactions under class imbalance,
using the [Fraudulent E-Commerce Transactions dataset](https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions)
(1.47M transactions, ~5% fraud rate).

It shares its core framing with [Loshanya/fraud-detection](https://github.com/Loshanya/fraud-detection)
(imbalanced binary fraud classification, multi-model comparison, PR-AUC as the
primary metric, threshold tuning instead of trusting 0.5) but differs in
**domain, feature set, and what the threshold is optimized for**.

## What's different from the reference project

| | Reference (Loshanya) | This project |
|---|---|---|
| Domain | Bank transfers (PaySim-style) | E-commerce transactions |
| Fraud rate | 0.129% (extreme) | ~5% (still imbalanced, less extreme) |
| Key features | Balance-delta (`orig_error`, `dest_balance_change`) | Account age, transaction hour, amount — plus engineered interactions |
| Threshold chosen by | Recall constraint (>95% recall) | **Expected rupee cost** (cost of missed fraud vs. cost of false flag) |
| Best PR-AUC | 0.996 (near-perfect — balance features make fraud almost trivially separable) | 0.38 (much harder — weak, noisy signal, closer to real-world fraud detection) |

## EDA findings (on the real data, not assumed)

Before engineering features, I checked which raw columns actually carried
fraud signal:

- **No signal**: Payment Method, Device Used, Product Category, Quantity,
  Customer Age, shipping/billing address mismatch — all within ~0.5 points
  of the 5% base fraud rate. IP addresses are also essentially unique per
  transaction (no reuse to exploit).
- **Real signal**:
  - **Transaction Amount** — fraud averages ₹548 vs ₹210 for legit (corr 0.27)
  - **Account Age Days** — fraud skews toward much newer accounts (median 61
    days vs 183 for legit)
  - **Transaction Hour** — 55% of fraud happens 12am–5am vs only 25% of
    legitimate transactions

This directly shaped the feature set — no point one-hot-encoding four
categorical columns that carry zero information.

## Engineered features

- `log_amount` — log-transform of the (right-skewed) transaction amount
- `is_new_account` — binary flag, account age ≤ 60 days
- `is_night_hour` — binary flag, transaction between 12am–5am
- `amount_x_new_account`, `amount_x_night` — interaction terms; a large
  purchase from a brand-new account, or at 3am, is a much stronger combined
  signal than either alone (confirmed by feature importance below)

## Models compared

| Model | PR-AUC | ROC-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | Train time |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.380** | 0.817 | 0.152 | 0.692 | 0.249 | 14s |
| Random Forest | 0.377 | 0.817 | 0.149 | 0.697 | 0.245 | 218s |
| Logistic Regression | 0.255 | 0.790 | 0.121 | 0.715 | 0.206 | 2s |

XGBoost wins on PR-AUC while training ~15x faster than Random Forest for
essentially the same score — a clear practical choice.

## Feature importance (XGBoost)

```
Account Age Days       0.320
amount_x_night         0.282
log_amount             0.206
Transaction Hour       0.140
amount_x_new_account   0.039
is_new_account         0.013
is_night_hour          0.000
```

The engineered interaction term (`amount_x_night`) outranks the raw
`Transaction Hour` feature — evidence the interaction genuinely adds signal
rather than just duplicating an existing column.

## Cost-sensitive threshold optimization

Instead of picking a threshold by F1 or a recall floor, this project defines
an explicit cost matrix and picks the threshold that **minimizes expected
rupee cost** on the test set:

- `cost_fn` (missing a real fraud) = ~2x the average fraud transaction
  amount (₹1,095.64), approximating the loss plus chargeback/penalty overhead
- `cost_fp` (wrongly flagging a genuine order) = ₹50, the cost of routing an
  order to manual review rather than an outright block

**Result:** cost-optimal threshold = **0.455** (barely below the naive 0.5),
giving a **1.2% reduction in expected cost** on the test set (₹94,057 saved
of ₹7.84M naive cost) — 76% fraud recall at 12.6% precision.

The honest takeaway: on this dataset, the fraud/non-fraud probability
distributions aren't well-separated enough for threshold placement to matter
much — the model itself (not the threshold) is the bottleneck. This is a
useful negative result and a natural bridge to future work below, rather
than an oversold "we saved millions" claim.

## Future work

- Try gradient-boosted feature interactions beyond the two engineered here
  (e.g. amount-per-account-age-day rate)
- SHAP-based explainability for individual flagged transactions
- Since the threshold barely moved cost, the more promising direction is
  better features (e.g. sequential/behavioral signals if the dataset
  supported repeat customers — this dataset has near-unique customer/IP
  values, limiting velocity-style features)
- Calibration check (Platt/isotonic) before treating raw XGBoost scores as
  cost-weighted probabilities

## Files

- `pipeline.py` — feature engineering + cost-sensitive threshold logic (importable, testable)
- `train.py` — full training script: loads data, trains 3 models, runs cost optimization, prints results
- `model_comparison_results.csv` — output metrics table
- `data/` — place `Fraudulent_E-Commerce_Transaction_Data.csv` here (not committed — too large for git)

## Tech stack

Python, Pandas, NumPy, Scikit-learn, XGBoost
