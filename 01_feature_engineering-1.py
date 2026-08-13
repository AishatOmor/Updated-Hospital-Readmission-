"""
Hospital Readmission Risk Prediction
Step 1: Feature Engineering

Scope note: this project predicts 30-day patient readmission risk, a clinical
care-management and discharge-planning problem. It is evidence of production-quality
predictive modeling on complex, real-world, messy clinical data -- NOT a claims
fraud detection project. Readmission risk and billing fraud are different problems,
and this project is deliberately not described as fraud-detection evidence anywhere
in this pipeline or its documentation.

Dataset: "Hospital Readmissions" (derived from the UCI Diabetes 130-US hospitals
dataset), 25,000 patient encounters, binary readmission outcome.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("/mnt/user-data/uploads/hospital_readmissions.csv")
print(f"Total encounters: {len(df):,}")
print(f"Readmitted: {(df['readmitted']=='yes').sum():,} ({(df['readmitted']=='yes').mean()*100:.2f}%)")

# ---------------------------------------------------------------------------
# 1. Target
# ---------------------------------------------------------------------------
df["Label"] = (df["readmitted"] == "yes").astype(int)

# ---------------------------------------------------------------------------
# 2. Age: convert bracket string to numeric midpoint (ordinal, preserves the
#    real clinical signal that risk increases with age more faithfully than
#    treating age brackets as unordered categories)
# ---------------------------------------------------------------------------
age_midpoints = {
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
    "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}
df["AgeMidpoint"] = df["age"].map(age_midpoints)

# ---------------------------------------------------------------------------
# 3. Ordinal encode glucose_test and A1Ctest (no < normal < high is a real
#    clinical ordering, not an arbitrary category order)
# ---------------------------------------------------------------------------
ordinal_map = {"no": 0, "normal": 1, "high": 2}
df["GlucoseTestOrdinal"] = df["glucose_test"].map(ordinal_map)
df["A1CTestOrdinal"] = df["A1Ctest"].map(ordinal_map)

# ---------------------------------------------------------------------------
# 4. Binary encode change / diabetes_med
# ---------------------------------------------------------------------------
df["ChangeFlag"] = (df["change"] == "yes").astype(int)
df["DiabetesMedFlag"] = (df["diabetes_med"] == "yes").astype(int)

# ---------------------------------------------------------------------------
# 5. One-hot encode nominal categoricals (medical_specialty, diag_1/2/3 --
#    these have no natural order, unlike age/glucose/A1C above)
# ---------------------------------------------------------------------------
nominal_cols = ["medical_specialty", "diag_1", "diag_2", "diag_3"]
df_encoded = pd.get_dummies(df, columns=nominal_cols, prefix=nominal_cols)

# ---------------------------------------------------------------------------
# 6. Assemble final feature set
# ---------------------------------------------------------------------------
numeric_cols = [
    "AgeMidpoint", "time_in_hospital", "n_lab_procedures", "n_procedures",
    "n_medications", "n_outpatient", "n_inpatient", "n_emergency",
    "GlucoseTestOrdinal", "A1CTestOrdinal", "ChangeFlag", "DiabetesMedFlag",
]
dummy_cols = [c for c in df_encoded.columns if any(c.startswith(p + "_") for p in nominal_cols)]

feature_cols = numeric_cols + dummy_cols
final_df = df_encoded[feature_cols + ["Label"]].copy()
# Ensure dummy columns are numeric (0/1) not boolean, for downstream sklearn compatibility
for c in dummy_cols:
    final_df[c] = final_df[c].astype(int)

print(f"\nFinal feature set: {final_df.shape[0]:,} rows, {len(feature_cols)} features")
print(f"Label rate: {final_df['Label'].mean():.4f}")

final_df.to_csv("/home/claude/readmission_analysis/readmission_features.csv", index=False)
print("\nSaved to readmission_features.csv")
print("\nFeature columns:")
print(feature_cols)
