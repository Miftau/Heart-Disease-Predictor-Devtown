# ==============================================
# HEART DISEASE PREDICTION
# ==============================================

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler
import joblib

# ----------------------------------------------
# Step 1: Load dataset (ensure path is correct)
# ----------------------------------------------
DATA_PATH = "heart_disease_uci.csv"  # ✅ Change this if file is elsewhere

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"❌ Dataset not found! Place 'heart_disease_uci.csv' in this directory or update DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)

print("✅ Data loaded successfully!")
print(df.head())

# ----------------------------------------------
# Step 2: Basic exploration
# ----------------------------------------------
print("\nDataset Info:")
df.info()
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# ----------------------------------------------
# Step 3: Handle missing data
# ----------------------------------------------
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = df.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0]).astype("object")

# ----------------------------------------------
# Step 4: Correlation heatmap
# ----------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Numeric Feature Correlation")
plt.show()

# ----------------------------------------------
# Step 5: Data preparation
# ----------------------------------------------
if "num" not in df.columns:
    raise ValueError("❌ Target column 'num' not found in dataset.")

X = df.drop("num", axis=1)
y = (df["num"] > 0).astype(int)  # Convert to binary labels

# One-hot encode categorical features
X = pd.get_dummies(X, columns=categorical_cols)
print("✅ Final feature columns:", len(X.columns))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------
# Step 6: Train models
# ----------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
    plt.title(f"ROC Curve - {name}")
    plt.show()

# ----------------------------------------------
# Step 7: Evaluate Random Forest in detail
# ----------------------------------------------
rf_model = models["Random Forest"]
cm = confusion_matrix(y_test, rf_model.predict(X_test_scaled))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix (Random Forest)")
plt.show()

# Cross-validation metrics
for metric in ["accuracy", "precision", "recall", "f1"]:
    score = cross_val_score(
        rf_model, X_train_scaled, y_train, cv=5, scoring=metric
    ).mean()
    print(f"{metric.title()} (CV): {score:.4f}")

# Feature importance plot
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind="barh")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()

# ----------------------------------------------
# Step 8: Save model and scaler
# ----------------------------------------------
joblib.dump(rf_model, "heart_rf_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
X.head(1).to_csv("heart_user_template.csv", index=False)
print("✅ Model and template saved successfully!")

# ----------------------------------------------
# Step 9: Predict on new user data
# ----------------------------------------------
print("\n📂 To make predictions, place your CSV file (same columns as template) in this folder.")
user_file = input("Enter your patient CSV filename (or press Enter to skip): ").strip()

if user_file:
    if not os.path.exists(user_file):
        print(f"❌ File '{user_file}' not found.")
    else:
        user_df = pd.read_csv(user_file)
        user_df_encoded = pd.get_dummies(user_df, columns=categorical_cols)
        user_df_encoded = user_df_encoded.reindex(columns=X.columns, fill_value=0)
        scaler = joblib.load("heart_scaler.pkl")
        model = joblib.load("heart_rf_model.pkl")
        preds = model.predict(scaler.transform(user_df_encoded))
        user_df["Heart_Disease_Prediction"] = preds
        print("\n✅ Prediction Results:")
        print(user_df)
        user_df.to_csv("heart_prediction_output.csv", index=False)
        print("\n📁 Results saved to 'heart_prediction_output.csv'")
else:
    print("➡️ Skipped prediction phase.")
