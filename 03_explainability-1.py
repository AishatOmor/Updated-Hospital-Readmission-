"""
Hospital Readmission Risk Prediction
Step 3: Explainability

Same permutation-importance approach as the other two projects in this
portfolio, for methodological consistency.
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, precision_recall_curve

with open("/home/claude/readmission_analysis/primary_model.pkl", "rb") as f:
    model = pickle.load(f)

test_df = pd.read_csv("/home/claude/readmission_analysis/test_predictions.csv")
feature_cols = [c for c in test_df.columns if c not in ("Label", "PrimaryModelProba", "BaselineProba")]
X_test = test_df[feature_cols]
y_test = test_df["Label"]

with open("/home/claude/readmission_analysis/results_summary.json") as f:
    summary = json.load(f)

# ---------------------------------------------------------------------------
# 1. Permutation importance
# ---------------------------------------------------------------------------
perm_result = permutation_importance(
    model, X_test, y_test, n_repeats=15, random_state=42, scoring="roc_auc", n_jobs=-1
)
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
}).sort_values("importance_mean", ascending=False)

print("Top 12 features by permutation importance:")
print(importance_df.head(12).to_string(index=False))
importance_df.to_csv("/home/claude/readmission_analysis/feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 6))
top_n = importance_df.head(12).iloc[::-1]
ax.barh(top_n["feature"], top_n["importance_mean"], xerr=top_n["importance_std"], color="#2E5C8A")
ax.set_xlabel("Permutation Importance (mean decrease in ROC-AUC when shuffled)")
ax.set_title("Feature Importance — Hospital Readmission Risk Model")
plt.tight_layout()
plt.savefig("/home/claude/readmission_analysis/feature_importance.png", dpi=150)
print("\nSaved feature_importance.png")

# ---------------------------------------------------------------------------
# 2. ROC and PR curves
# ---------------------------------------------------------------------------
baseline_proba = test_df["BaselineProba"]
primary_proba = test_df["PrimaryModelProba"]

fpr_b, tpr_b, _ = roc_curve(y_test, baseline_proba)
fpr_p, tpr_p, _ = roc_curve(y_test, primary_proba)
prec_b, rec_b, _ = precision_recall_curve(y_test, baseline_proba)
prec_p, rec_p, _ = precision_recall_curve(y_test, primary_proba)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr_p, tpr_p, color="#2E5C8A", label=f"Primary (ROC-AUC={summary['primary_roc_auc']:.3f})")
axes[0].plot(fpr_b, tpr_b, color="#B0B0B0", label=f"Baseline (ROC-AUC={summary['baseline_roc_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "--", color="lightgray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("ROC Curve — Test Set")
axes[0].legend()

axes[1].plot(rec_p, prec_p, color="#2E5C8A", label=f"Primary (PR-AUC={summary['primary_pr_auc']:.3f})")
axes[1].plot(rec_b, prec_b, color="#B0B0B0", label=f"Baseline (PR-AUC={summary['baseline_pr_auc']:.3f})")
axes[1].axhline(y=summary["readmit_rate_overall"], linestyle="--", color="lightgray", label="No-skill baseline")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Test Set")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/claude/readmission_analysis/roc_pr_curves.png", dpi=150)
print("Saved roc_pr_curves.png")

# ---------------------------------------------------------------------------
# 3. Example: a genuinely high-risk-scored patient encounter, with reasons
# ---------------------------------------------------------------------------
train_means = X_test.mean()
test_df_sorted = test_df.sort_values("PrimaryModelProba", ascending=False)
example = test_df_sorted.iloc[0]

print("\n" + "=" * 70)
print("EXAMPLE: highest-risk-scored encounter in the test set")
print("=" * 70)
print(f"Model-assigned readmission probability: {example['PrimaryModelProba']:.3f}")
print(f"Actual outcome: {'Readmitted' if example['Label'] == 1 else 'Not readmitted'}")
print(f"Age midpoint: {example['AgeMidpoint']}, Time in hospital: {example['time_in_hospital']:.0f} days, "
      f"Prior inpatient visits: {example['n_inpatient']:.0f}, Prior emergency visits: {example['n_emergency']:.0f}")

print("\nDone. All outputs saved to /home/claude/readmission_analysis/")
