# ==============================================
# DUAL HEART DISEASE PREDICTION MODELS
# ==============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, RocCurveDisplay
)
import joblib

# ==============================================
# MODEL 1: CLINICAL MODEL (14 Features)
# ==============================================
print("\n🚑 TRAINING CLINICAL MODEL...")
DATA_PATH_CLINICAL = "heart_disease_uci.csv"

if not os.path.exists(DATA_PATH_CLINICAL):
    raise FileNotFoundError("❌ Clinical dataset not found!")

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num"
]

df1 = pd.read_csv(DATA_PATH_CLINICAL, names=column_names, header=0, na_values='?')
df1 = df1.apply(pd.to_numeric, errors='coerce')
df1 = df1.fillna(df1.mean(numeric_only=True))

X1 = df1.drop("num", axis=1)
y1 = (df1["num"] > 0).astype(int)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

scaler1 = StandardScaler()
X_train1_scaled = scaler1.fit_transform(X_train1)
X_test1_scaled = scaler1.transform(X_test1)

rf1 = RandomForestClassifier(random_state=42)
rf1.fit(X_train1_scaled, y_train1)
y_pred1 = rf1.predict(X_test1_scaled)

print(f"✅ Clinical Model Accuracy: {accuracy_score(y_test1, y_pred1):.4f}")
print(classification_report(y_test1, y_pred1))

joblib.dump(rf1, "heart_rf_clinical.pkl")
joblib.dump(scaler1, "heart_scaler_clinical.pkl")
X1.head(1).to_csv("heart_user_template_clinical.csv", index=False)
print("💾 Clinical model and template saved!\n")

# ==============================================
# MODEL 2: LIFESTYLE MODEL (General / Self-Report)
# ==============================================
print("\n🏃 TRAINING LIFESTYLE MODEL...")
DATA_PATH_LIFESTYLE = "cardio_train.csv"

if not os.path.exists(DATA_PATH_LIFESTYLE):
    raise FileNotFoundError("❌ Lifestyle dataset 'cardio_train.csv' not found!")

# Load lifestyle dataset
df2 = pd.read_csv(DATA_PATH_LIFESTYLE, sep=';')

# Drop ID if present
if 'id' in df2.columns:
    df2 = df2.drop(columns=['id'])

# Feature-target split
if 'cardio' not in df2.columns:
    raise ValueError("❌ Target column 'cardio' missing in lifestyle dataset!")

X2 = df2.drop("cardio", axis=1)
y2 = df2["cardio"]

# Handle missing values (if any)
df2 = df2.fillna(df2.mean(numeric_only=True))

# Split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Scale
scaler2 = StandardScaler()
X_train2_scaled = scaler2.fit_transform(X_train2)
X_test2_scaled = scaler2.transform(X_test2)

# Train
rf2 = RandomForestClassifier(random_state=42)
rf2.fit(X_train2_scaled, y_train2)
y_pred2 = rf2.predict(X_test2_scaled)

print(f"✅ Lifestyle Model Accuracy: {accuracy_score(y_test2, y_pred2):.4f}")
print(classification_report(y_test2, y_pred2))

# Save
joblib.dump(rf2, "heart_rf_lifestyle.pkl")
joblib.dump(scaler2, "heart_scaler_lifestyle.pkl")
X2.head(1).to_csv("heart_user_template_lifestyle.csv", index=False)
print("💾 Lifestyle model and template saved!\n")

# ==============================================
# OPTIONAL VISUALIZATIONS
# ==============================================
plt.figure(figsize=(6, 4))
RocCurveDisplay.from_estimator(rf1, X_test1_scaled, y_test1)
plt.title("ROC Curve - Clinical Model")
plt.show()

plt.figure(figsize=(6, 4))
RocCurveDisplay.from_estimator(rf2, X_test2_scaled, y_test2)
plt.title("ROC Curve - Lifestyle Model")
plt.show()

print("✅ Both models trained and saved successfully!")
