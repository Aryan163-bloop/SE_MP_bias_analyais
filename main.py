import os
import warnings
warnings.filterwarnings("ignore")

# =========================
# CREATE OUTPUT FOLDER
# =========================

os.makedirs("outputs", exist_ok=True)

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("dataset/adult.csv")

print("\nDataset Loaded Successfully.\n")

print(df.head())

# =========================
# HANDLE MISSING VALUES
# =========================

df.replace("?", np.nan, inplace=True)

df.dropna(inplace=True)

# =========================
# PRINT COLUMN NAMES
# =========================

print("\nColumns in Dataset:\n")

print(df.columns)

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================

label_encoders = {}

categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:

    df[column] = df[column].astype(str)

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column])

    label_encoders[column] = le

print("\nCategorical Encoding Completed.\n")

# =========================
# FEATURES AND TARGET
# =========================

X = df.drop("income", axis=1)

y = df["income"]

# Sensitive feature
sensitive_feature = df["gender"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
    X,
    y,
    sensitive_feature,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed.\n")

# =========================
# TRAIN RANDOM FOREST MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed.\n")

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# MODEL ACCURACY
# =========================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")

# =========================
# CLASSIFICATION REPORT
# =========================

report = classification_report(y_test, y_pred)

print("\nClassification Report:\n")

print(report)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")

plt.close()

print("\nConfusion Matrix Saved.\n")

# =========================
# FAIRNESS METRICS
# =========================

dpd = demographic_parity_difference(
    y_test,
    y_pred,
    sensitive_features=sf_test
)

eod = equalized_odds_difference(
    y_test,
    y_pred,
    sensitive_features=sf_test
)

print("\n===== FAIRNESS METRICS =====\n")

print(f"Demographic Parity Difference: {dpd:.4f}")

print(f"Equalized Odds Difference: {eod:.4f}")

# =========================
# SAVE FAIRNESS METRICS
# =========================

with open("outputs/fairness_metrics.txt", "w") as f:

    f.write("FAIRNESS METRICS\n")
    f.write("========================\n\n")

    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Demographic Parity Difference: {dpd:.4f}\n")
    f.write(f"Equalized Odds Difference: {eod:.4f}\n")

print("\nFairness Metrics Saved.\n")

# =========================
# APPLY BIAS MITIGATION
# =========================

print("\nApplying Bias Mitigation...\n")

mitigator = ExponentiatedGradient(
    estimator=RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    constraints=DemographicParity()
)

mitigator.fit(
    X_train,
    y_train,
    sensitive_features=sf_train
)

mitigated_predictions = mitigator.predict(X_test)

# =========================
# FAIRNESS AFTER MITIGATION
# =========================

mitigated_dpd = demographic_parity_difference(
    y_test,
    mitigated_predictions,
    sensitive_features=sf_test
)

mitigated_eod = equalized_odds_difference(
    y_test,
    mitigated_predictions,
    sensitive_features=sf_test
)

print("\n===== AFTER MITIGATION =====\n")

print(f"Mitigated Demographic Parity Difference: {mitigated_dpd:.4f}")

print(f"Mitigated Equalized Odds Difference: {mitigated_eod:.4f}")

# =========================
# SAVE TRAINED MODEL
# =========================

joblib.dump(model, "bias_model.pkl")

print("\nModel Saved Successfully.\n")

# =========================
# GENERATE FINAL REPORT
# =========================

with open("outputs/bias_report.txt", "w") as f:

    f.write("BIAS ANALYSIS REPORT\n")
    f.write("============================\n\n")

    f.write(f"Initial Accuracy: {accuracy:.4f}\n\n")

    f.write("Before Bias Mitigation\n")
    f.write("----------------------------\n")

    f.write(f"Demographic Parity Difference: {dpd:.4f}\n")
    f.write(f"Equalized Odds Difference: {eod:.4f}\n\n")

    f.write("After Bias Mitigation\n")
    f.write("----------------------------\n")

    f.write(f"Mitigated Demographic Parity Difference: {mitigated_dpd:.4f}\n")
    f.write(f"Mitigated Equalized Odds Difference: {mitigated_eod:.4f}\n")

print("\nBias Report Saved.\n")

print("\nPROJECT EXECUTED SUCCESSFULLY.\n")

