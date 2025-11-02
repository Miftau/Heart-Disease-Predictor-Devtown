import os
import uuid
import time
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pandas as pd
import numpy as np
import joblib
from flask_mail import Mail, Message
from dotenv import load_dotenv

# Load environment variables (if .env present)
load_dotenv()

# -------------------------
# CONFIG
# -------------------------
# Model files - ensure these exist
CLINICAL_MODEL_FILE = "heart_rf_clinical.pkl"
CLINICAL_SCALER_FILE = "heart_scaler_clinical.pkl"
CLINICAL_TEMPLATE_FILE = "heart_user_template_clinical.csv"

LIFESTYLE_MODEL_FILE = "heart_rf_lifestyle.pkl"
LIFESTYLE_SCALER_FILE = "heart_scaler_lifestyle.pkl"
LIFESTYLE_TEMPLATE_FILE = "heart_user_template_lifestyle.csv"

RESULTS_DIR = os.path.join("static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Base columns for forms (clinical = the 13 input cols used previously)
BASE_COLUMNS_CLINICAL = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Lifestyle columns according to your cardio_train.csv (without id and target)
# Provided earlier: id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio
BASE_COLUMNS_LIFESTYLE = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]

# Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# === Flask-Mail Configuration ===
app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", 587))
app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS", "True") == "True"
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = (os.getenv("MAIL_SENDER_NAME", "HeartPredict Contact"), os.getenv("MAIL_USERNAME"))

mail = Mail(app)

@app.context_processor
def inject_now():
    """Add current year to all templates."""
    return {"current_year": datetime.now().year}

# -------------------------
# Load models & templates
# -------------------------
missing = []
for path in [CLINICAL_MODEL_FILE, CLINICAL_SCALER_FILE, CLINICAL_TEMPLATE_FILE,
             LIFESTYLE_MODEL_FILE, LIFESTYLE_SCALER_FILE, LIFESTYLE_TEMPLATE_FILE]:
    if not os.path.exists(path):
        missing.append(path)

if missing:
    raise FileNotFoundError(f"Missing required model/template files: {missing}")

# Load clinical model + scaler + feature columns (these are one-hot encoded columns)
clinical_model = joblib.load(CLINICAL_MODEL_FILE)
clinical_scaler = joblib.load(CLINICAL_SCALER_FILE)
clinical_template_df = pd.read_csv(CLINICAL_TEMPLATE_FILE)
CLINICAL_FEATURE_COLUMNS = clinical_template_df.columns.tolist()

# Load lifestyle model + scaler + feature columns (likely raw numeric columns)
lifestyle_model = joblib.load(LIFESTYLE_MODEL_FILE)
lifestyle_scaler = joblib.load(LIFESTYLE_SCALER_FILE)
lifestyle_template_df = pd.read_csv(LIFESTYLE_TEMPLATE_FILE)
LIFESTYLE_FEATURE_COLUMNS = lifestyle_template_df.columns.tolist()

# -------------------------
# Helper: prepare and predict
# -------------------------
def prepare_and_predict(df_raw: pd.DataFrame, model_type: str):
    """
    Preprocess input dataframe and predict using the selected model_type ('clinical'|'lifestyle').
    Returns: out_df (original input fields + Prediction, Prob_Pos (0-1), Risk_Level)
    """
    if model_type not in ("clinical", "lifestyle"):
        raise ValueError("model_type must be 'clinical' or 'lifestyle'")

    # Work on a copy
    df = df_raw.copy()

    # If headerless CSV gives integer column names, use positional mapping
    if all(isinstance(c, int) for c in df.columns):
        if model_type == "clinical":
            # clinical expects 13 input cols (no target), or 14 with target
            if df.shape[1] >= 13:
                df = df.iloc[:, :13]
                df.columns = BASE_COLUMNS_CLINICAL
        else:
            # lifestyle expects 11 input columns (age..active)
            if df.shape[1] >= len(BASE_COLUMNS_LIFESTYLE):
                df = df.iloc[:, :len(BASE_COLUMNS_LIFESTYLE)]
                df.columns = BASE_COLUMNS_LIFESTYLE

    # Keep original for output
    out_input = df.copy()

    if model_type == "clinical":
        # Keep only base clinical columns
        df = df.loc[:, [c for c in BASE_COLUMNS_CLINICAL if c in df.columns]]

        # One-hot encode categorical cols (if any) then align with CLINICAL_FEATURE_COLUMNS
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        df_encoded = pd.get_dummies(df, columns=cat_cols)
        df_encoded = df_encoded.reindex(columns=CLINICAL_FEATURE_COLUMNS, fill_value=0)

        # Scale
        X = clinical_scaler.transform(df_encoded.values)
        model = clinical_model

    else:  # lifestyle
        # Keep only lifestyle base cols
        df = df.loc[:, [c for c in BASE_COLUMNS_LIFESTYLE if c in df.columns]]

        # Ensure numeric, coerce errors to NaN then fill with column mean
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(df.mean(numeric_only=True))

        # Reindex to match LIFESTYLE_FEATURE_COLUMNS (which should be numeric feature columns)
        df_reindexed = df.reindex(columns=LIFESTYLE_FEATURE_COLUMNS, fill_value=0)
        df_encoded = df_reindexed  # no one-hot here (assuming lifestyle model trained on numeric features)

        # Scale
        X = lifestyle_scaler.transform(df_encoded.values)
        model = lifestyle_model

    # Predict probabilities and classes
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        try:
            df_dec = model.decision_function(X)
            probs = 1 / (1 + np.exp(-df_dec))
        except Exception:
            probs = model.predict(X)

    preds = model.predict(X)

    # Build output dataframe
    out_df = out_input.copy()
    out_df["Prediction"] = preds
    out_df["Prob_Pos"] = np.round(probs, 4)

    def risk_level(p):
        if p >= 0.66:
            return "High"
        elif p >= 0.33:
            return "Medium"
        else:
            return "Low"

    out_df["Risk_Level"] = out_df["Prob_Pos"].apply(risk_level)
    return out_df

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")

        if not name or not email or not message:
            flash("Please fill in all fields.", "danger")
            return redirect(url_for("contact"))

        try:
            msg = Message(
                subject=f"New Contact Form Message from {name}",
                recipients=[os.getenv("MAIL_DEFAULT_RECEIVER")],
                body=f"""
You have received a new message from your website contact form.

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}
"""
            )
            mail.send(msg)
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            flash(f"Error sending message: {str(e)}", "danger")

        return redirect(url_for("contact"))

    return render_template("contact.html")

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route("/form")
def form():
    return render_template(
        "form.html",
        BASE_COLUMNS_CLINICAL=BASE_COLUMNS_CLINICAL,
        BASE_COLUMNS_LIFESTYLE=BASE_COLUMNS_LIFESTYLE
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept common frontend model names and map them to internal names
        raw_model_type = (request.form.get("model_type") or request.args.get("model_type") or "clinical").lower().strip()
        model_map = {
            "heart": "clinical",
            "clinical": "clinical",
            "cardio": "lifestyle",
            "lifestyle": "lifestyle"
        }
        model_type = model_map.get(raw_model_type, None)

        if model_type is None:
            flash("Invalid model selection. Choose 'Clinical' or 'Lifestyle'.", "danger")
            return redirect(url_for("form"))

        print(f"[INFO] Prediction requested. Model type (raw): {raw_model_type} → mapped: {model_type}")

        # Case A: CSV upload
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename != "":
            # Read CSV (start with headerless read)
            uploaded_file.stream.seek(0)
            df = pd.read_csv(uploaded_file, header=None)

            # try to detect headerful CSV
            uploaded_file.stream.seek(0)
            try:
                df_try = pd.read_csv(uploaded_file)
                uploaded_file.stream.seek(0)
                # If df_try has enough numeric columns, prefer it
                numeric_count = df_try.dtypes.apply(lambda x: np.issubdtype(x, np.number)).sum()
                if df_try.shape[1] >= 5 and numeric_count >= 3:
                    df = df_try
                else:
                    uploaded_file.stream.seek(0)
                    df = pd.read_csv(uploaded_file, header=None)
            except Exception:
                uploaded_file.stream.seek(0)
                df = pd.read_csv(uploaded_file, header=None)

            results_df = prepare_and_predict(df, model_type=model_type)

            # Save results with model name
            fname = f"pred_{model_type}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.csv"
            save_path = os.path.join(RESULTS_DIR, fname)
            results_df.to_csv(save_path, index=False)

            download_url = url_for("static", filename=f"results/{fname}")
            return render_template("result.html",
                                   tables=[results_df.to_html(classes="table-auto w-full text-sm", index=False)],
                                   download_link=download_url,
                                   model_type=model_type)

        # Case B: Manual form input
        else:
            missing = []
            data = {}
            # pick proper base columns
            base_cols = BASE_COLUMNS_CLINICAL if model_type == "clinical" else BASE_COLUMNS_LIFESTYLE

            # map 'sex' -> 'gender' for lifestyle if user uses 'sex' in form
            for col in base_cols:
                # prefer form keys: direct name, but allow aliases
                val = None
                if col in request.form:
                    val = request.form.get(col)
                else:
                    # alias handling:
                    if col == "gender" and "sex" in request.form:
                        val = request.form.get("sex")
                    # sometimes frontend uses 'alco' vs 'alcohol'
                    elif col == "alco" and "alcohol" in request.form:
                        val = request.form.get("alcohol")
                    elif col == "smoke" and "smoking" in request.form:
                        val = request.form.get("smoking")
                    else:
                        val = request.form.get(col)

                if val is None or val == "":
                    missing.append(col)
                else:
                    # Coerce numeric types where possible, but keep strings for categorical
                    try:
                        data[col] = float(val)
                    except Exception:
                        data[col] = val

            if missing:
                # Give helpful message listing which inputs were missing for selected model
                flash(f"Missing inputs for model '{model_type}': {', '.join(missing)}. Please fill all required fields.", "danger")
                return redirect(url_for("form"))

            # Build dataframe and predict
            user_df = pd.DataFrame([data], columns=base_cols)
            results_df = prepare_and_predict(user_df, model_type=model_type)

            # Save single-result CSV
            fname = f"pred_manual_{model_type}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.csv"
            save_path = os.path.join(RESULTS_DIR, fname)
            results_df.to_csv(save_path, index=False)
            download_url = url_for("static", filename=f"results/{fname}")

            single = results_df.iloc[0]
            readable_result = None
            if model_type == "clinical":
                readable_result = "No Heart Disease Detected" if single["Prediction"] == 0 else "Heart Disease Detected"
            else:
                # lifestyle model target is 'cardio' — use more generic phrasing
                readable_result = "Low Cardiovascular Risk" if single["Prediction"] == 0 else "Elevated Cardiovascular Risk"

            prob = float(single["Prob_Pos"])
            risk = single["Risk_Level"]

            return render_template("result.html",
                                   result=readable_result,
                                   prob=prob,
                                   risk=risk,
                                   tables=[results_df.to_html(classes="table-auto w-full text-sm", index=False)],
                                   download_link=download_url,
                                   model_type=model_type)
    except Exception as e:
        # Log error in server logs (optional print)
        print("Error during prediction:", str(e))
        # show the form with an error message
        flash(f"Error processing request: {str(e)}", "danger")
        return redirect(url_for("form"))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

# Optional health endpoint for uptime monitors
@app.route("/health")
def health():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    # Note: consider using gunicorn for production
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
