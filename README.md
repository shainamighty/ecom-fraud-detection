    # E-Commerce Fraud Detection with Cost-Sensitive Threshold Optimization

## Overview

An end-to-end fraud detection system for e-commerce transactions, built on the
[Fraudulent E-Commerce Transactions dataset](https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions)
(1.47M transactions, ~5% fraud rate). The project covers EDA-driven feature
engineering, multi-model comparison, and — the core focus — choosing a
decision threshold based on actual rupee cost instead of a generic metric
like F1.

Most fraud detection writeups stop at "we got 97% F1" and pick threshold=0.5
without asking what a false positive or false negative actually costs a
business. This project treats that as the central question: given a cost
matrix (cost of missing a fraud vs. cost of wrongly flagging a genuine
order), what threshold actually minimizes total cost — and does tuning it
even matter?

## EDA — what actually predicts fraud here

Before building any features, I checked which raw columns carried real
signal rather than assuming:

- **No signal**: Payment Method, Device Used, Product Category, Quantity,
  Customer Age, shipping/billing address mismatch — all sat within ~0.5
  points of the 5% base fraud rate regardless of value. IP addresses were
  also essentially unique per transaction, so there's no reuse pattern to
  exploit.
- **Real signal**:
  - **Transaction Amount** — fraud averages ₹548 vs ₹210 for legitimate
    transactions (correlation 0.27)
  - **Account Age Days** — fraud skews toward much newer accounts (median
    61 days vs 183 for legit)
  - **Transaction Hour** — 55% of fraud happens 12am–5am vs only 25% of
    legitimate transactions

This directly shaped which features made it into the model — no point
one-hot-encoding four categorical columns that carry zero information.

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
essentially the same score — the clear practical choice.

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

Rather than picking a threshold by F1 or a fixed recall floor, this project
defines an explicit cost matrix and picks the threshold that **minimizes
expected rupee cost** on the test set:

- `cost_fn` (missing a real fraud) = ~2x the average fraud transaction
  amount (₹1,095.64), approximating the loss plus chargeback/penalty overhead
- `cost_fp` (wrongly flagging a genuine order) = ₹50, the cost of routing an
  order to manual review rather than an outright block

**Result:** cost-optimal threshold = **0.455** (barely below the naive 0.5),
giving a **1.2% reduction in expected cost** on the test set (₹94,057 saved
of ₹7.84M naive cost) — 76% fraud recall at 12.6% precision.

The honest takeaway: on this dataset, the fraud/non-fraud probability
distributions aren't well-separated enough for threshold placement alone to
move the needle much — the model itself (not the cutoff) is the bottleneck.
That's a useful finding in its own right, and a natural bridge to the future
work below, rather than an oversold "we saved millions" claim.

## Future work

- Try additional feature interactions beyond the two engineered here (e.g.
  amount-per-account-age-day rate)
- SHAP-based explainability for individual flagged transactions
- Since the threshold barely moved cost, the more promising direction is
  better features — e.g. sequential/behavioral velocity signals, though this
  dataset's near-unique customer/IP values limit that here
- Calibration check (Platt/isotonic) before treating raw XGBoost scores as
  cost-weighted probabilities

## Files

- `pipeline.py` — feature engineering + cost-sensitive threshold logic (importable, testable)
- `train.py` — full training script: loads data, trains 3 models, runs cost optimization, prints results
- `model_comparison_results.csv` — output metrics table
- `data/` — place `Fraudulent_E-Commerce_Transaction_Data.csv` here (not committed — too large for git)

## Tech stack

Python, Pandas, NumPy, Scikit-learn, XGBoost


Author : 
Shaina Srujitha

    
