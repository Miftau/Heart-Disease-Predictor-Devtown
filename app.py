import os
import uuid
import time
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, jsonify
)
import pandas as pd
import numpy as np
import joblib
from flask_mail import Mail, Message
from dotenv import load_dotenv
import requests

# ============================================================
# APP INITIALIZATION & CONFIG
# ============================================================
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# -------------------------------
# EMAIL CONFIGURATION
# -------------------------------
app.config.update(
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_USE_TLS=os.getenv("MAIL_USE_TLS", "True") == "True",
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_DEFAULT_SENDER=(
        os.getenv("MAIL_SENDER_NAME", "CardioGuard Contact"),
        os.getenv("MAIL_USERNAME")
    ),
)
mail = Mail(app)


@app.context_processor
def inject_now():
    """Inject current year in templates."""
    return {"current_year": datetime.now().year}


# ============================================================
# PATHS & MODEL FILES
# ============================================================
RESULTS_DIR = os.path.join("static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLINICAL_MODEL_FILE = "heart_rf_clinical.pkl"
CLINICAL_SCALER_FILE = "heart_scaler_clinical.pkl"
CLINICAL_TEMPLATE_FILE = "heart_user_template_clinical.csv"

LIFESTYLE_MODEL_FILE = "heart_rf_lifestyle.pkl"
LIFESTYLE_SCALER_FILE = "heart_scaler_lifestyle.pkl"
LIFESTYLE_TEMPLATE_FILE = "heart_user_template_lifestyle.csv"

REMOTE_MODEL_BASE = os.getenv("REMOTE_MODEL_BASE", "")


# ============================================================
# MODEL AUTO-DOWNLOAD HANDLER
# ============================================================
def ensure_model_files():
    """Ensure model files exist locally; download if not present."""
    required_files = [
        CLINICAL_MODEL_FILE, CLINICAL_SCALER_FILE, CLINICAL_TEMPLATE_FILE,
        LIFESTYLE_MODEL_FILE, LIFESTYLE_SCALER_FILE, LIFESTYLE_TEMPLATE_FILE,
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if not missing:
        return

    if REMOTE_MODEL_BASE:
        print(f"🛰️ Missing model files detected: {missing}")
        for fname in missing:
            try:
                url = f"{REMOTE_MODEL_BASE}/{fname}"
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                with open(fname, "wb") as f:
                    f.write(r.content)
                print(f"✅ Downloaded {fname}")
            except Exception as e:
                print(f"❌ Failed to download {fname}: {e}")
    else:
        raise FileNotFoundError(f"Missing required model/template files: {missing}")


ensure_model_files()


# ============================================================
# MODEL LOADING
# ============================================================
def load_models():
    print("🔄 Loading trained models...")

    clinical_model = joblib.load(CLINICAL_MODEL_FILE)
    clinical_scaler = joblib.load(CLINICAL_SCALER_FILE)
    clinical_template_df = pd.read_csv(CLINICAL_TEMPLATE_FILE)
    CLINICAL_FEATURE_COLUMNS = clinical_template_df.columns.tolist()

    lifestyle_model = joblib.load(LIFESTYLE_MODEL_FILE)
    lifestyle_scaler = joblib.load(LIFESTYLE_SCALER_FILE)
    lifestyle_template_df = pd.read_csv(LIFESTYLE_TEMPLATE_FILE)
    LIFESTYLE_FEATURE_COLUMNS = lifestyle_template_df.columns.tolist()

    print("✅ Models loaded successfully.")
    return (clinical_model, clinical_scaler, CLINICAL_FEATURE_COLUMNS,
            lifestyle_model, lifestyle_scaler, LIFESTYLE_FEATURE_COLUMNS)


(
    clinical_model, clinical_scaler, CLINICAL_FEATURE_COLUMNS,
    lifestyle_model, lifestyle_scaler, LIFESTYLE_FEATURE_COLUMNS
) = load_models()

# ============================================================
# MODEL INPUT COLUMNS
# ============================================================
BASE_COLUMNS_CLINICAL = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

BASE_COLUMNS_LIFESTYLE = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]

# ============================================================
# DIAGNOSTIC RULES (Heuristic)
# ============================================================
def get_likely_condition(age, cholesterol, resting_bp, max_heart_rate,
                         fasting_blood_sugar=None, exercise_angina=None,
                         chest_pain_type=None, oldpeak=None, st_slope=None,
                         sex=None, smoking=None, obesity=None,
                         alcohol=None, physical_activity=None):

    likely_condition = "Generalized Cardiac Risk"
    suggestions = []

    try:
        age = float(age)
        cholesterol = float(cholesterol)
        resting_bp = float(resting_bp)
        max_heart_rate = float(max_heart_rate)
        oldpeak = float(oldpeak) if oldpeak is not None else 0
        fasting_blood_sugar = float(fasting_blood_sugar or 0)
    except Exception:
        return likely_condition, ["Incomplete data to infer likely condition."]

    # --- Rule-based categories ---
    if (cholesterol > 240 and resting_bp > 140) or (st_slope == "down") or (oldpeak > 2.0):
        likely_condition = "Coronary Artery Disease (CAD)"
        suggestions = [
            "Adopt a low-fat, high-fiber diet.",
            "Consider a coronary CT or angiogram.",
            "Begin supervised cardio exercises."
        ]
    elif (max_heart_rate < 100 and age > 60) or (oldpeak > 1.5):
        likely_condition = "Heart Failure (HF)"
        suggestions = [
            "Limit salt and monitor daily weight.",
            "Request echocardiogram to assess heart function.",
            "Avoid heavy exertion."
        ]
    elif fasting_blood_sugar > 120 and cholesterol > 200:
        likely_condition = "Diabetic Cardiomyopathy"
        suggestions = [
            "Control blood sugar and avoid refined carbs.",
            "Get regular ECG checkups.",
            "Consult both cardiologist and endocrinologist."
        ]
    elif max_heart_rate > 180:
        likely_condition = "Cardiac Arrhythmia"
        suggestions = [
            "Avoid caffeine, stress, and alcohol.",
            "Schedule an ECG or Holter monitor test.",
            "Discuss rhythm management options."
        ]
    elif resting_bp > 160 and age > 40:
        likely_condition = "Hypertensive Heart Disease"
        suggestions = [
            "Monitor BP regularly.",
            "Avoid salt, smoking, and alcohol.",
            "Consider echocardiogram to assess wall thickening."
        ]
    elif smoking == "yes" and cholesterol > 200:
        likely_condition = "Smoking-related Coronary Risk"
        suggestions = [
            "Immediate smoking cessation is strongly advised.",
            "Perform lipid profile and stress ECG.",
            "Adopt heart-healthy habits."
        ]
    elif alcohol == "yes" and age > 35:
        likely_condition = "Alcoholic Cardiomyopathy"
        suggestions = [
            "Strict abstinence from alcohol.",
            "Take vitamin B complex, especially thiamine.",
            "Schedule cardiac evaluation."
        ]
    elif obesity == "yes" and (physical_activity in ["low", "none"]):
        likely_condition = "Obesity-related Cardiomyopathy"
        suggestions = [
            "Aim for gradual weight loss.",
            "Exercise 30 minutes 5x weekly.",
            "Adopt calorie-restricted diet."
        ]
    elif (cholesterol < 200 and resting_bp < 120 and fasting_blood_sugar < 100):
        likely_condition = "Low Cardiac Risk"
        suggestions = [
            "Maintain a balanced diet and regular exercise.",
            "Annual routine checkup is recommended.",
            "Continue heart-healthy lifestyle."
        ]

    return likely_condition, suggestions


# ============================================================
# PREDICTION HELPER
# ============================================================
def prepare_and_predict(df_raw, model_type):
    if model_type not in ("clinical", "lifestyle"):
        raise ValueError("Invalid model type.")
    df = df_raw.copy()

    if all(isinstance(c, int) for c in df.columns):
        if model_type == "clinical":
            df.columns = BASE_COLUMNS_CLINICAL[:df.shape[1]]
        else:
            df.columns = BASE_COLUMNS_LIFESTYLE[:df.shape[1]]

    if model_type == "clinical":
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols)
        df = df.reindex(columns=CLINICAL_FEATURE_COLUMNS, fill_value=0)
        X = clinical_scaler.transform(df.values)
        model = clinical_model
    else:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(df.mean(numeric_only=True))
        df = df.reindex(columns=LIFESTYLE_FEATURE_COLUMNS, fill_value=0)
        X = lifestyle_scaler.transform(df.values)
        model = lifestyle_model

    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    preds = model.predict(X)

    df_out = df_raw.copy()
    df_out["Prediction"] = preds
    df_out["Prob_Pos"] = np.round(probs, 4)
    df_out["Risk_Level"] = df_out["Prob_Pos"].apply(
        lambda p: "High" if p > 0.66 else ("Medium" if p > 0.33 else "Low")
    )

    return df_out


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")

        if not all([name, email, message]):
            flash("Please fill all fields.", "danger")
            return redirect(url_for("contact"))

        try:
            msg = Message(
                subject=f"[CardioGuard] {subject or 'New Message'} from {name}",
                recipients=[os.getenv("MAIL_DEFAULT_RECEIVER")],
                body=f"Name: {name}\nEmail: {email}\n\n{message}",
            )
            mail.send(msg)
            flash("✅ Message sent successfully!", "success")
        except Exception as e:
            flash(f"❌ Failed to send: {e}", "danger")

        return redirect(url_for("contact"))
    return render_template("contact.html")


@app.route("/form")
def form():
    return render_template(
        "form.html",
        BASE_COLUMNS_CLINICAL=BASE_COLUMNS_CLINICAL,
        BASE_COLUMNS_LIFESTYLE=BASE_COLUMNS_LIFESTYLE,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_type = (request.form.get("model_type") or "clinical").lower()
        model_map = {"heart": "clinical", "clinical": "clinical", "cardio": "lifestyle", "lifestyle": "lifestyle"}
        model_type = model_map.get(raw_type, "clinical")

        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename:
            df = pd.read_csv(uploaded_file)
            results = prepare_and_predict(df, model_type)
        else:
            base_cols = BASE_COLUMNS_CLINICAL if model_type == "clinical" else BASE_COLUMNS_LIFESTYLE
            user_data = {col: request.form.get(col, None) for col in base_cols}
            df = pd.DataFrame([user_data])
            results = prepare_and_predict(df, model_type)

        # Save results
        fname = f"{model_type}_pred_{uuid.uuid4().hex[:8]}.csv"
        save_path = os.path.join(RESULTS_DIR, fname)
        results.to_csv(save_path, index=False)
        download_link = url_for("static", filename=f"results/{fname}")

        single = results.iloc[0]
        prob = float(single["Prob_Pos"])
        risk = single["Risk_Level"]
        readable = (
            "Heart Disease Detected" if model_type == "clinical" and single["Prediction"] == 1
            else "Elevated Cardiovascular Risk" if model_type == "lifestyle" and single["Prediction"] == 1
            else "No Heart Disease Detected"
        )

        # --- Apply rule-based condition inference ---
        likely_condition, suggestions = get_likely_condition(
            age=single.get("age"),
            cholesterol=single.get("chol", single.get("cholesterol", 0)),
            resting_bp=single.get("trestbps", single.get("ap_hi", 0)),
            max_heart_rate=single.get("thalach", single.get("ap_lo", 0)),
            fasting_blood_sugar=single.get("fbs", single.get("gluc", 0)),
            sex=single.get("sex"),
            smoking=single.get("smoke"),
            alcohol=single.get("alco"),
            physical_activity=single.get("active"),
        )

        return render_template(
            "result.html",
            result=readable,
            prob=prob,
            risk=risk,
            likely_condition=likely_condition,
            suggestions=suggestions,
            tables=[results.to_html(classes="table table-striped", index=False)],
            download_link=download_link,
            model_type=model_type,
        )

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        flash(f"Error processing prediction: {e}", "danger")
        return redirect(url_for("form"))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": time.time()}), 200


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
