import os
import uuid
import time
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pandas as pd
import numpy as np
import joblib

# CONFIG
MODEL_FILE = "heart_rf_model.pkl"      # trained RandomForest model
SCALER_FILE = "heart_scaler.pkl"       # fitted StandardScaler
TEMPLATE_FILE = "heart_user_template.csv"  # contains X.columns (one-hot encoded columns)
RESULTS_DIR = os.path.join("static", "results")
BASE_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")  # change in prod

# Load model, scaler, and feature columns
if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(TEMPLATE_FILE)):
    raise FileNotFoundError("Make sure heart_rf_model.pkl, heart_scaler.pkl and heart_user_template.csv exist in the app folder.")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Read template to get feature columns (one-hot encoded)
template_df = pd.read_csv(TEMPLATE_FILE)
FEATURE_COLUMNS = template_df.columns.tolist()   # final columns used by model

def prepare_and_predict(df_raw):
    """
    Takes a dataframe with base columns (or many columns),
    returns (df_with_predictions, saveable_df)
    - df_with_predictions: df including 'Prediction', 'Prob_Pos', 'Risk_Level'
    """
    # If df_raw has 14 columns (no headers in original data), assign base + num
    if df_raw.shape[1] == 14 and all(isinstance(c, int) for c in df_raw.columns):
        df_raw.columns = BASE_COLUMNS + ["num"]
    # If df_raw has 13 columns with integer column names -> assign base columns
    elif df_raw.shape[1] == 13 and all(isinstance(c, int) for c in df_raw.columns):
        df_raw.columns = BASE_COLUMNS

    # If uploaded CSV contains extra columns, try to keep the base columns
    if not set(BASE_COLUMNS).issubset(set(df_raw.columns)):
        # attempt to infer by position if no matching names
        if df_raw.shape[1] >= 13 and all(isinstance(c, int) for c in df_raw.columns):
            df_raw = df_raw.iloc[:, :13]
            df_raw.columns = BASE_COLUMNS
        else:
            # If columns exist but different names, try to rename heuristically (not foolproof)
            flash("Uploaded CSV doesn't contain the expected columns. Ensure file has 13 input columns (order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) or includes headers named accordingly.", "warning")

    # Keep original for output
    df_input = df_raw.copy()

    # Only keep base columns (if other columns exist)
    df_input = df_input.loc[:, [c for c in BASE_COLUMNS if c in df_input.columns]]

    # One-hot encode categorical fields, then align with FEATURE_COLUMNS
    df_encoded = pd.get_dummies(df_input, columns=df_input.select_dtypes(include=['object','category']).columns.tolist())
    df_encoded = df_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Scale numeric features (scaler expects numeric array)
    X_scaled = scaler.transform(df_encoded.values)

    # Predict probabilities and class
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]  # probability of positive class
    else:
        # fallback: use decision_function or predict (not as good)
        try:
            probs = model.decision_function(X_scaled)
            # convert to 0-1 via sigmoid
            probs = 1 / (1 + np.exp(-probs))
        except Exception:
            probs = model.predict(X_scaled)

    preds = model.predict(X_scaled)

    # create output dataframe (original columns + prediction info)
    out_df = df_input.copy()
    out_df["Prediction"] = preds
    out_df["Prob_Pos"] = np.round(probs, 4)

    # Risk level mapping
    def risk_level(p):
        if p >= 0.66:
            return "High"
        elif p >= 0.33:
            return "Medium"
        else:
            return "Low"

    out_df["Risk_Level"] = out_df["Prob_Pos"].apply(risk_level)

    return out_df

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Case A: CSV upload
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename != "":
            # Read CSV. Let pandas infer header. If headerless, columns are numeric indices.
            df = pd.read_csv(uploaded_file, header=None)
            # If user included header row intentionally, try to detect - simple heuristic:
            # if header row values are strings and not numeric, re-read with header=0
            header_candidate = pd.read_csv(uploaded_file, nrows=1)
            # but uploaded_file is consumed; re-read from beginning
            uploaded_file.stream.seek(0)
            try:
                # attempt read with header infer
                df_try = pd.read_csv(uploaded_file)
                # Heuristic: if df_try has 13 or 14 columns and first row values are not numeric, assume header present
                numeric_count = df_try.dtypes.apply(lambda x: np.issubdtype(x, np.number)).sum()
                if df_try.shape[1] >= 13 and numeric_count >= 5:
                    df = df_try
                else:
                    # keep original no-header df already read
                    uploaded_file.stream.seek(0)
                    df = pd.read_csv(uploaded_file, header=None)
            except Exception:
                uploaded_file.stream.seek(0)
                df = pd.read_csv(uploaded_file, header=None)

            results_df = prepare_and_predict(df)

            # Save CSV with timestamp + uuid
            fname = f"pred_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.csv"
            save_path = os.path.join(RESULTS_DIR, fname)
            results_df.to_csv(save_path, index=False)

            download_url = url_for("static", filename=f"results/{fname}")
            # render result page (table + download link)
            return render_template("result.html", tables=[results_df.to_html(classes="table-auto w-full text-sm", index=False, justify="center")], download_link=download_url)

        else:
            missing = []
            data = {}
            for col in BASE_COLUMNS:
                val = request.form.get(col)
                if val is None or val == "":
                    missing.append(col)
                else:
                    data[col] = float(val)
            if missing:
                flash(f"Missing inputs for: {', '.join(missing)}. Please fill all fields.", "danger")
                return redirect(url_for("index"))

            user_df = pd.DataFrame([data], columns=BASE_COLUMNS)
            results_df = prepare_and_predict(user_df)

            # Save single-result CSV
            fname = f"pred_manual_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.csv"
            save_path = os.path.join(RESULTS_DIR, fname)
            results_df.to_csv(save_path, index=False)
            download_url = url_for("static", filename=f"results/{fname}")

            # For single-record, show summary result (result string) and table
            single = results_df.iloc[0]
            readable_result = "No Heart Disease Detected" if single["Prediction"] == 0 else "Heart Disease Detected"
            prob = single["Prob_Pos"]
            risk = single["Risk_Level"]

            return render_template("result.html",
                                   result=readable_result,
                                   prob=prob,
                                   risk=risk,
                                   tables=[results_df.to_html(classes="table-auto w-full text-sm", index=False)],
                                   download_link=download_url)
    except Exception as e:
        # Safe error message
        return render_template("index.html", error=f"Error processing request: {str(e)}")

@app.route("/download/<path:filename>")
def download_file(filename):
    # optional: if you want a custom route for downloads
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
