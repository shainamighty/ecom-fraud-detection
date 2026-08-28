import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

from pipeline import engineer_features, NUMERIC_COLS, find_cost_optimal_threshold

print("Loading data...")
df = pd.read_csv("data/Fraudulent_E-Commerce_Transaction_Data.csv")
df = engineer_features(df)

X = df[NUMERIC_COLS]
y = df["Is Fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}, fraud rate test: {y_test.mean():.4f}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric="logloss", n_jobs=-1, random_state=42,
                              scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()),
}

results = []
probas = {}
for name, model in models.items():
    t0 = time.time()
    if name == "Logistic Regression":
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
    dt = time.time() - t0
    probas[name] = proba
    pred_05 = (proba >= 0.5).astype(int)
    results.append({
        "Model": name,
        "PR-AUC": average_precision_score(y_test, proba),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Precision@0.5": precision_score(y_test, pred_05),
        "Recall@0.5": recall_score(y_test, pred_05),
        "F1@0.5": f1_score(y_test, pred_05),
        "train_time_s": round(dt, 1),
    })
    print(f"{name} done in {dt:.1f}s")

results_df = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# Best model = XGBoost (expected). Now do cost-sensitive threshold optimization.
best_proba = probas["XGBoost"]

# Cost assumptions (documented, adjustable):
#   cost_fn: missing a fraud -> lose the full average fraud transaction amount
#            plus chargeback/penalty overhead. Using ~2x the mean fraud amount
#            (~₹548) as a conservative estimate of true loss incl. penalties.
#   cost_fp: wrongly flagging a genuine order -> cost of manual review +
#            fraction of customers abandoning due to friction. Using a small
#            fixed review cost since false positives here just mean "route to
#            manual review," not an outright block.
fraud_mean_amount = df.loc[df["Is Fraudulent"] == 1, "Transaction Amount"].mean()
COST_FN = round(fraud_mean_amount * 2, 2)   # ~ full loss + penalty overhead
COST_FP = 50.0                              # manual review cost per flagged order

cost_result = find_cost_optimal_threshold(y_test.values, best_proba, cost_fn=COST_FN, cost_fp=COST_FP)
best_t = cost_result["best_threshold"]

print(f"\n=== Cost-Sensitive Threshold Optimization (XGBoost) ===")
print(f"cost_fn (missed fraud) = ₹{COST_FN}, cost_fp (false flag / review) = ₹{COST_FP}")
print(f"Cost-optimal threshold: {best_t:.3f}, expected total cost on test set: ₹{cost_result['best_cost']:,.0f}")

# Compare against the naive 0.5 threshold and F1-optimal threshold cost
naive_cost = cost_result["costs"][np.argmin(np.abs(cost_result["thresholds"] - 0.5))]
print(f"Cost at naive threshold 0.5: ₹{naive_cost:,.0f}")
print(f"Savings from cost-optimal threshold: ₹{naive_cost - cost_result['best_cost']:,.0f} ({(1 - cost_result['best_cost']/naive_cost)*100:.1f}% reduction)")

pred_best_t = (best_proba >= best_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_best_t).ravel()
print(f"\nConfusion matrix @ cost-optimal threshold {best_t:.3f}:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"  Precision={tp/(tp+fp):.3f}  Recall={tp/(tp+fn):.3f}")

# Feature importance from XGBoost
importances = pd.Series(models["XGBoost"].feature_importances_, index=NUMERIC_COLS).sort_values(ascending=False)
print("\n=== XGBoost Feature Importance ===")
print(importances.to_string())

results_df.to_csv("model_comparison_results.csv", index=False)
np.save("cost_curve_thresholds.npy", cost_result["thresholds"])
np.save("cost_curve_costs.npy", cost_result["costs"])
print("\nSaved model_comparison_results.csv and cost curve arrays.")
