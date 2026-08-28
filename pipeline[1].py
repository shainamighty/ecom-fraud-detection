"""
Core feature engineering + cost-sensitive evaluation logic for the
e-commerce fraud detection project.

Core idea kept from the reference project (Loshanya/fraud-detection):
  - severe class imbalance framing
  - multi-model comparison (LogReg, RF, XGBoost)
  - PR-AUC as primary ranking metric
  - threshold tuning rather than trusting the default 0.5

What's different here:
  - e-commerce transaction dataset instead of bank transfers -> new,
    domain-specific engineered features (device/IP reuse, account age,
    category risk, transaction-hour risk) instead of balance-delta features
  - threshold is chosen by minimizing EXPECTED RUPEE COST, not F1 -- i.e.
    an actual cost matrix (cost of a missed fraud vs. cost of wrongly
    blocking a genuine customer) drives the operating point, which is the
    business-facing version of what the reference repo only listed as a
    "future improvement"
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add e-commerce-specific fraud signals to the raw transaction table.

    NOTE: these features were chosen based on actual EDA on the real
    shriyashjagtap/fraudulent-e-commerce-transactions dataset (1.47M rows),
    not assumed. Payment Method, Device Used, Product Category, Quantity,
    Customer Age, and shipping/billing address mismatch all showed ~zero
    fraud-rate difference in this dataset and were DROPPED as features --
    including them would just add noise. The real signal lives in
    Transaction Amount, Account Age Days, and Transaction Hour.
    """
    out = df.copy()

    # Data quality fix: a handful of Customer Age values are negative
    # (data entry artifact in the synthetic generator) -- clip to a sane floor
    out["Customer Age"] = out["Customer Age"].clip(lower=0)

    # 1. Log-transform amount: right-skewed, and the single strongest
    #    predictor in this dataset (fraud mean ~₹548 vs ~₹210 for legit)
    out["log_amount"] = np.log1p(out["Transaction Amount"])

    # 2. New-account risk: fraud clusters in newer accounts
    #    (fraud median account age ~61 days vs ~183 for legit)
    out["is_new_account"] = (out["Account Age Days"] <= 60).astype(int)

    # 3. Night-time transaction: 55% of fraud happens 12am-5am vs 25% of
    #    legit transactions -- one of the strongest categorical splits found
    out["is_night_hour"] = out["Transaction Hour"].between(0, 5).astype(int)

    # 4. Amount x new-account interaction: a large purchase from a brand-new
    #    account is a much stronger combined signal than either alone
    out["amount_x_new_account"] = out["log_amount"] * out["is_new_account"]

    # 5. Amount x night-hour interaction: same logic for large + late-night
    out["amount_x_night"] = out["log_amount"] * out["is_night_hour"]

    return out


# Payment Method / Product Category / Device Used / Quantity / Customer Age /
# address-mismatch are intentionally excluded -- EDA showed no fraud signal.
CATEGORICAL_COLS: list = []
NUMERIC_COLS = [
    "log_amount", "Account Age Days", "Transaction Hour",
    "is_new_account", "is_night_hour",
    "amount_x_new_account", "amount_x_night",
]


# ---------------------------------------------------------------------------
# Cost-sensitive threshold optimization
# ---------------------------------------------------------------------------

def expected_cost(y_true: np.ndarray, y_proba: np.ndarray, threshold: float,
                   cost_fn: float, cost_fp: float) -> float:
    """
    Total expected cost at a given threshold.

    cost_fn: rupee cost of missing an actual fraud (chargeback + fraud amount + penalty)
    cost_fp: rupee cost of wrongly blocking/flagging a genuine customer
             (lost order value * conversion hit + support/manual-review cost)
    """
    y_pred = (y_proba >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return fn * cost_fn + fp * cost_fp


def find_cost_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                                 cost_fn: float, cost_fp: float,
                                 n_steps: int = 199) -> dict:
    """
    Scans thresholds in (0, 1) and returns the one that minimizes total
    expected rupee cost, plus the cost curve for plotting.
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    costs = [expected_cost(y_true, y_proba, t, cost_fn, cost_fp) for t in thresholds]
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_cost": float(costs[best_idx]),
        "thresholds": thresholds,
        "costs": costs,
    }
