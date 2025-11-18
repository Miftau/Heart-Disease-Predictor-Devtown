# ==============================================
# DUAL HEART DISEASE PREDICTION MODELS
# WITH DATA PROFILING + MULTIPLE MODELS
# ==============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    RocCurveDisplay, confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import xgboost as xgb
import joblib
from tabulate import tabulate

# ==============================================
# UTILITY: DATA PROFILING
# ==============================================
def profile_data(df, dataset_name):
    print(f"\n📊 DATA PROFILE: {dataset_name}")
    print("-" * 50)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Data Types:\n{df.dtypes.value_counts().to_string()}")
    
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    valid_cells = total_cells - missing_cells
    print(f"\nValid Cells: {valid_cells} ({100 * valid_cells / total_cells:.2f}%)")
    print(f"Missing/Invalid Cells: {missing_cells} ({100 * missing_cells / total_cells:.2f}%)")

    # Missing per column
    missing_per_col = df.isnull().sum()
    if missing_per_col.sum() > 0:
        print("\nMissing Values by Column:")
        missing_df = missing_per_col[missing_per_col > 0].to_frame(name='Missing Count')
        missing_df['% Missing'] = (missing_df['Missing Count'] / len(df)) * 100
        print(tabulate(missing_df, headers='keys', tablefmt='github', floatfmt=".2f"))

    # Target distribution (assume last column is target)
    target_col = df.columns[-1]
    if df[target_col].dtype in ['int64', 'float64']:
        target_dist = df[target_col].value_counts(normalize=True) * 100
        print(f"\nTarget Distribution ({target_col}):")
        for val, pct in target_dist.items():
            print(f"  Class {val}: {pct:.2f}%")

    # Feature histograms (numerical only)
    num_cols = df.select_dtypes(include=[np.number]).columns[:-1]  # exclude target
    if len(num_cols) > 0:
        n_cols = min(4, len(num_cols))
        n_rows = (len(num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        for i, col in enumerate(num_cols):
            df[col].hist(ax=axes[i], bins=20, alpha=0.7)
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.suptitle(f"Feature Distributions – {dataset_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # Missingness heatmap
    if missing_cells > 0:
        plt.figure(figsize=(10, 4))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title(f"Missing Value Heatmap – {dataset_name}")
        plt.show()


# ==============================================
# UTILITY: MODEL EVALUATION
# ==============================================
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    }
    return metrics, y_proba


# ==============================================
# MODEL 1: CLINICAL MODEL (14 Features)
# ==============================================
print("\n🚑 TRAINING CLINICAL MODELS (RF, XGB, MLP)...")
DATA_PATH_CLINICAL = "heart_disease_uci.csv"

if not os.path.exists(DATA_PATH_CLINICAL):
    raise FileNotFoundError("❌ Clinical dataset not found!")

# Load with inferred header (your file uses no header; first row is data)
df1 = pd.read_csv(DATA_PATH_CLINICAL, header=None)
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num"
]
df1.columns = column_names

# Replace '?' with NaN and convert
df1 = df1.replace('?', np.nan)
df1 = df1.apply(pd.to_numeric, errors='coerce')

# Profile data
profile_data(df1, "Clinical (UCI Heart Disease)")

# Handle missing values
df1 = df1.fillna(df1.mean(numeric_only=True))

X1 = df1.drop("num", axis=1)
y1 = (df1["num"] > 0).astype(int)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

scaler1 = StandardScaler()
X_train1_scaled = scaler1.fit_transform(X_train1)
X_test1_scaled = scaler1.transform(X_test1)

# Models
models1 = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, alpha=0.01
    )
}

results1 = []
trained_models1 = {}

for name, model in models1.items():
    print(f"  Training {name} (Clinical)...")
    model.fit(X_train1_scaled, y_train1)
    metrics, _ = evaluate_model(model, X_test1_scaled, y_test1, name)
    results1.append(metrics)
    trained_models1[name] = model

# Save
for name, model in trained_models1.items():
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"heart_{safe_name.lower()}_clinical.pkl")
joblib.dump(scaler1, "heart_scaler_clinical.pkl")
X1.head(1).to_csv("heart_user_template_clinical.csv", index=False)

# Display results
print("\n✅ Clinical Model Performance Comparison:")
print(tabulate(results1, headers="keys", tablefmt="github", floatfmt=".4f"))

# ROC Curve
plt.figure(figsize=(7, 5))
for name, model in trained_models1.items():
    RocCurveDisplay.from_estimator(model, X_test1_scaled, y_test1, name=name, ax=plt.gca())
plt.title("ROC Curves – Clinical Models")
plt.legend(loc="lower right")
plt.show()


# ==============================================
# MODEL 2: LIFESTYLE MODEL (General / Self-Report)
# ==============================================
print("\n🏃 TRAINING LIFESTYLE MODELS (RF, XGB, MLP)...")
DATA_PATH_LIFESTYLE = "cardio_train.csv"

if not os.path.exists(DATA_PATH_LIFESTYLE):
    raise FileNotFoundError("❌ Lifestyle dataset 'cardio_train.csv' not found!")

df2 = pd.read_csv(DATA_PATH_LIFESTYLE, sep=';')

if 'id' in df2.columns:
    df2 = df2.drop(columns=['id'])

if 'cardio' not in df2.columns:
    raise ValueError("❌ Target column 'cardio' missing!")

# Profile data
profile_data(df2, "Lifestyle (Cardio)")

X2 = df2.drop("cardio", axis=1)
y2 = df2["cardio"]

# Handle missing (should be none, but just in case)
df2 = df2.fillna(df2.mean(numeric_only=True))

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
X_train2_scaled = scaler2.fit_transform(X_train2)
X_test2_scaled = scaler2.transform(X_test2)

models2 = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, alpha=0.01
    )
}

results2 = []
trained_models2 = {}

for name, model in models2.items():
    print(f"  Training {name} (Lifestyle)...")
    model.fit(X_train2_scaled, y_train2)
    metrics, _ = evaluate_model(model, X_test2_scaled, y_test2, name)
    results2.append(metrics)
    trained_models2[name] = model

# Save
for name, model in trained_models2.items():
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"heart_{safe_name.lower()}_lifestyle.pkl")
joblib.dump(scaler2, "heart_scaler_lifestyle.pkl")
X2.head(1).to_csv("heart_user_template_lifestyle.csv", index=False)

# Display results
print("\n✅ Lifestyle Model Performance Comparison:")
print(tabulate(results2, headers="keys", tablefmt="github", floatfmt=".4f"))

# ROC Curve
plt.figure(figsize=(7, 5))
for name, model in trained_models2.items():
    RocCurveDisplay.from_estimator(model, X_test2_scaled, y_test2, name=name, ax=plt.gca())
plt.title("ROC Curves – Lifestyle Models")
plt.legend(loc="lower right")
plt.show()

print("\n✅ All models trained, evaluated, visualized, and saved successfully!")