# ==============================================
# HEART DISEASE PREDICTION (14-Feature Version)
# ==============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    RocCurveDisplay,
)
import joblib

# ----------------------------------------------
# Step 1: Load dataset
# ----------------------------------------------
DATA_PATH = "heart_disease_uci.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"❌ Dataset not found! Place 'heart_disease_uci.csv' in this directory or update DATA_PATH."
    )

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# Replace '?' with NaN while reading
df = pd.read_csv(DATA_PATH, names=column_names, header=0, na_values='?')

print("✅ Data loaded successfully!")
print(df.head())

# ----------------------------------------------
# Step 2: Convert data types
# ----------------------------------------------
# Convert all columns to numeric (if possible)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ----------------------------------------------
# Step 3: Handle missing data
# ----------------------------------------------
df = df.fillna(df.mean(numeric_only=True))
print("✅ Missing values handled successfully!")

# ----------------------------------------------
# Step 4: Correlation heatmap
# ----------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ----------------------------------------------
# Step 5: Feature-target split
# ----------------------------------------------
if "num" not in df.columns:
    raise ValueError("❌ Target column 'num' not found in dataset.")

X = df.drop("num", axis=1)
y = (df["num"] > 0).astype(int)  # Binary target (1 = disease, 0 = no disease)

print(f"✅ Using {X.shape[1]} input features")

# ----------------------------------------------
# Step 6: Train-test split
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------
# Step 7: Feature scaling
# ----------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------
# Step 8: Train Random Forest model
# ----------------------------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Random Forest Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# ROC curve
RocCurveDisplay.from_estimator(rf_model, X_test_scaled, y_test)
plt.title("ROC Curve - Random Forest (14 Features)")
plt.show()

# ----------------------------------------------
# Step 9: Confusion matrix and feature importance
# ----------------------------------------------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.show()

feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind="barh")
plt.title("Top Feature Importances (Random Forest)")
plt.show()

# ----------------------------------------------
# Step 10: Cross-validation
# ----------------------------------------------
for metric in ["accuracy", "precision", "recall", "f1"]:
    score = cross_val_score(
        rf_model, X_train_scaled, y_train, cv=5, scoring=metric
    ).mean()
    print(f"{metric.title()} (CV): {score:.4f}")

# ----------------------------------------------
# Step 11: Save model, scaler, and template
# ----------------------------------------------
joblib.dump(rf_model, "heart_rf_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
X.head(1).to_csv("heart_user_template.csv", index=False)
print("\n✅ Model, scaler, and template saved successfully!")

# ----------------------------------------------
# Step 12: Predict new data (optional test)
# ----------------------------------------------
user_file = input("\nEnter your patient CSV filename (or press Enter to skip): ").strip()

if user_file:
    if not os.path.exists(user_file):
        print(f"❌ File '{user_file}' not found.")
    else:
        user_df = pd.read_csv(user_file)
        user_df = user_df.fillna(user_df.mean(numeric_only=True))
        scaled = scaler.transform(user_df)
        preds = rf_model.predict(scaled)
        probs = rf_model.predict_proba(scaled)[:, 1]
        user_df["Prediction"] = preds
        user_df["Confidence"] = (probs * 100).round(2)
        print("\n✅ Prediction Results:")
        print(user_df)
        user_df.to_csv("heart_prediction_output.csv", index=False)
        print("\n📁 Results saved to 'heart_prediction_output.csv'")
else:
    print("➡️ Skipped prediction phase.")

