# Hospital Readmission Risk Prediction

A predictive model for identifying patients at elevated risk of hospital
readmission, built on real-world, messy clinical encounter data with mixed
categorical and numeric features.

## Scope note — please read before citing this project

**This is a clinical care-management and discharge-planning prediction
problem. It is not a claims fraud detection project**, and it is not
described as one anywhere in this repository. Readmission risk (will this
patient come back to the hospital) and billing fraud (is this claim
fraudulent) are different problems requiring different data, different
labels, and different modeling objectives. This project's purpose within a
broader portfolio is to demonstrate production-quality predictive modeling
on complex, real-world healthcare data — feature engineering across mixed
categorical/numeric/ordinal clinical variables, proper baseline comparison,
and honest performance reporting — which is directly transferable capability
to healthcare claims fraud modeling, but is not itself a fraud detection
result.

## Dataset

["Hospital Readmissions"](https://www.kaggle.com/datasets/dubradave/hospital-readmissions)
dataset (derived from the UCI Machine Learning Repository's "Diabetes
130-US hospitals for years 1999-2008" dataset) — 25,000 patient encounters
with a binary readmission outcome (47.0% readmitted), diagnosis categories,
medication and lab test information, and demographic/utilization history.

## Methodology

### Feature engineering (`01_feature_engineering.py`)
- Age brackets converted to numeric midpoints (preserves the real ordinal
  clinical relationship between age and risk)
- Glucose test and A1C test results ordinally encoded (no < normal < high)
- Diagnosis categories (three fields) and medical specialty one-hot encoded
- Final feature set: 43 features across 25,000 encounters, no missing data

### Modeling (`02_modeling.py`)
- **Baseline:** Logistic Regression on standardized features
- **Primary:** Gradient Boosted Trees (scikit-learn `HistGradientBoostingClassifier`)
- **Robustness check:** Random Forest
- Standard stratified 75/25 train/test split (the dataset is well-balanced
  at 47% positive class, so — unlike the fraud-detection projects in this
  portfolio — no resampling was necessary or used)

### Explainability (`03_explainability.py`)
Permutation importance identifies which features drive the model's risk
scores. See the SHAP note in the fraud-detection projects in this portfolio;
the same offline-environment limitation applies here.

## Results — reported honestly, including the limits

| Model | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|
| Baseline: Logistic Regression | 0.652 | 0.622 | 0.617 |
| **Primary: Gradient Boosted Trees** | **0.657** | **0.627** | **0.618** |
| Random Forest (robustness check) | 0.660 | 0.628 | — |

**This is a genuinely modest result, and it is reported as such rather than
inflated.** Predicting hospital readmission is a well-documented hard
problem in the clinical machine learning literature — published research on
this exact dataset and closely related versions of it typically reports
ROC-AUC in the 0.60–0.68 range, because readmission is driven substantially
by factors this kind of encounter-level dataset does not capture (social
determinants of health, post-discharge care access, medication adherence,
housing stability, and other variables outside the hospital record).

The value of this result is not a claim of high predictive accuracy. It is
that:
1. **The model's top predictive feature is prior inpatient visit count**,
   which is independently well-established in the clinical literature as
   one of the strongest known predictors of readmission (it is a core
   component of established clinical risk tools such as the LACE index) —
   this is a meaningful sanity check that the model has learned a real,
   clinically valid signal rather than noise, even though overall
   discrimination is modest.
2. **The full pipeline — cleaning, encoding mixed categorical/ordinal/numeric
   clinical variables, proper train/test methodology, and honest, non-inflated
   performance reporting — demonstrates the same production-quality modeling
   discipline** this endeavor's healthcare claims fraud work depends on.

### Top predictive features
1. `n_inpatient` — number of prior inpatient visits (by a wide margin the
   strongest predictor)
2. `n_outpatient` — number of prior outpatient visits
3. `n_medications` — number of medications administered
4. `AgeMidpoint` — patient age
5. `n_emergency` — number of prior emergency visits

## Reproducing these results

```bash
pip install pandas numpy scikit-learn matplotlib
python 01_feature_engineering.py
python 02_modeling.py
python 03_explainability.py
```

Requires the "Hospital Readmissions" dataset (`hospital_readmissions.csv`),
available from the dataset page linked above.

## Limitations

- Modest overall discrimination (ROC-AUC ≈ 0.66), consistent with published
  literature on this task; this project should not be cited as evidence of
  a high-accuracy readmission predictor, but as evidence of sound modeling
  methodology on a genuinely difficult, real-world clinical prediction
  problem.
- This dataset does not include billing, claims, or reimbursement data, and
  this project makes no claims about fraud, upcoding, or billing anomaly
  detection.
- Readmission outcome definition (e.g., 30-day vs. any-time readmission) and
  patient de-identification are governed by the original dataset's
  publishers; no institutional patient data was used.
