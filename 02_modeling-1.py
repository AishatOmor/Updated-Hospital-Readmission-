"""
Hospital Readmission Risk Prediction
Step 2: Baseline and Primary Model Training + Evaluation

Unlike the two fraud-detection projects, this dataset is well-balanced (47%
positive class), so no resampling is needed here -- a standard stratified
train/test split is appropriate and standard practice for this kind of
clinical prediction task.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, classification_report, accuracy_score
)

RANDOM_STATE = 42

df = pd.read_csv("/home/claude/readmission_analysis/readmission_features.csv")
feature_cols = [c for c in df.columns if c != "Label"]
X = df[feature_cols]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train):,} encounters ({y_train.mean():.4f} readmit rate)")
print(f"Test:  {len(X_test):,} encounters ({y_test.mean():.4f} readmit rate)")

# ---------------------------------------------------------------------------
# Baseline: Logistic Regression on standardized features
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
baseline.fit(X_train_scaled, y_train)
baseline_proba = baseline.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# Primary: Gradient Boosted Trees
# ---------------------------------------------------------------------------
primary = HistGradientBoostingClassifier(
    max_iter=300, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE
)
primary.fit(X_train, y_train)
primary_proba = primary.predict_proba(X_test)[:, 1]

# Random Forest robustness check
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
results = {}
for name, proba in [("Baseline (Logistic Regression)", baseline_proba),
                     ("Primary (Gradient Boosted Trees)", primary_proba),
                     ("Random Forest (robustness check)", rf_proba)]:
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    results[name] = {"ROC_AUC": auc, "PR_AUC": ap, "Accuracy": acc}
    print(f"\n{name}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC (Average Precision): {ap:.4f}")
    print(f"  Accuracy (0.5 threshold): {acc:.4f}")

print("\nFull classification report (primary model, 0.5 threshold):")
print(classification_report(y_test, (primary_proba >= 0.5).astype(int),
                             target_names=["Not Readmitted", "Readmitted"]))

cm = confusion_matrix(y_test, (primary_proba >= 0.5).astype(int))
print("Confusion matrix (primary model, 0.5 threshold):")
print(f"                      Predicted No   Predicted Yes")
print(f"Actual Not Readmit      {cm[0][0]:>6}          {cm[0][1]:>6}")
print(f"Actual Readmitted       {cm[1][0]:>6}          {cm[1][1]:>6}")

# ---------------------------------------------------------------------------
# Recall/FPR tradeoff scan, same transparent multi-point reporting as the
# other two projects in this portfolio
# ---------------------------------------------------------------------------
def fpr_at_recall(y_true, scores, target_recall):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = np.searchsorted(tpr, target_recall)
    idx = min(idx, len(tpr) - 1)
    return float(fpr[idx]), float(tpr[idx])

recall_scan = []
for target in [0.60, 0.70, 0.80, 0.90]:
    b_fpr, b_rec = fpr_at_recall(y_test, baseline_proba, target)
    p_fpr, p_rec = fpr_at_recall(y_test, primary_proba, target)
    rel = (b_fpr - p_fpr) / b_fpr * 100 if b_fpr > 0 else None
    recall_scan.append({
        "target_recall": target, "baseline_actual_recall": b_rec,
        "baseline_fpr_pct": b_fpr * 100, "primary_actual_recall": p_rec,
        "primary_fpr_pct": p_fpr * 100, "relative_fpr_reduction_pct": rel,
    })

print("\nFalse positive rate at matched recall (transparent multi-point scan):")
for r in recall_scan:
    print(f"  Recall={r['target_recall']:.2f}: baseline FPR={r['baseline_fpr_pct']:.2f}%, "
          f"primary FPR={r['primary_fpr_pct']:.2f}%, relative reduction={r['relative_fpr_reduction_pct']:.1f}%")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
summary = {
    "n_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "readmit_rate_overall": float(y.mean()),
    "baseline_roc_auc": float(results["Baseline (Logistic Regression)"]["ROC_AUC"]),
    "baseline_pr_auc": float(results["Baseline (Logistic Regression)"]["PR_AUC"]),
    "baseline_accuracy": float(results["Baseline (Logistic Regression)"]["Accuracy"]),
    "primary_roc_auc": float(results["Primary (Gradient Boosted Trees)"]["ROC_AUC"]),
    "primary_pr_auc": float(results["Primary (Gradient Boosted Trees)"]["PR_AUC"]),
    "primary_accuracy": float(results["Primary (Gradient Boosted Trees)"]["Accuracy"]),
    "rf_roc_auc": float(results["Random Forest (robustness check)"]["ROC_AUC"]),
    "recall_scan": recall_scan,
}
with open("/home/claude/readmission_analysis/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

with open("/home/claude/readmission_analysis/primary_model.pkl", "wb") as f:
    pickle.dump(primary, f)

X_test.assign(Label=y_test.values, PrimaryModelProba=primary_proba, BaselineProba=baseline_proba).to_csv(
    "/home/claude/readmission_analysis/test_predictions.csv", index=False
)
X_train.to_csv("/home/claude/readmission_analysis/X_train.csv", index=False)

print("\nSaved: results_summary.json, primary_model.pkl, test_predictions.csv, X_train.csv")
