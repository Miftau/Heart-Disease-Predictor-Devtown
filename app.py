# app.py
"""
CardioGuard Flask app (full).
Features:
 - Clinical & Lifestyle predictions (CSV or manual)
 - Heuristic condition inference & suggestions
 - AI chat endpoint (Groq preferred, OpenAI fallback)
 - Optional Supabase chat-history persistence
 - Model auto-download from REMOTE_MODEL_BASE (if configured)
 - Health endpoint for uptime monitoring

Requirements (install in your venv):
pip install flask pandas numpy scikit-learn joblib python-dotenv flask-mail requests supabase py-sdk-openai groq flask-cors

Note: package names may vary slightly for groq or supabase clients; adjust to what you actually use:
 - supabase: "supabase" or "supabase-py" (depending on venv)
 - groq: "groq" (if you plan to use Groq)
 - openai: "openai" (fallback)
"""

import os
import bcrypt
import uuid
import time
import json
import requests
import markdown
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from groq import Groq
from supabase import create_client, Client
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, jsonify, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from flask_mail import Mail, Message
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import joblib

# Optional SDKs — imported inside try/except so app still runs without them
try:
    from supabase import create_client as create_supabase_client
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

try:
    # Groq client; if not installed, fallback to openai
    import groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ============================================================
# App initialization
# ============================================================
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
app.config["TEMPLATES_AUTO_RELOAD"] = True
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])

# Email config (optional)
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
    return {"current_year": datetime.now().year}

# ============================================================
# Paths, model filenames & optional remote base
# ============================================================
RESULTS_DIR = os.path.join("static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLINICAL_MODEL_FILE = os.getenv("CLINICAL_MODEL_FILE", "heart_rf_clinical.pkl")
CLINICAL_SCALER_FILE = os.getenv("CLINICAL_SCALER_FILE", "heart_scaler_clinical.pkl")
CLINICAL_TEMPLATE_FILE = os.getenv("CLINICAL_TEMPLATE_FILE", "heart_user_template_clinical.csv")

LIFESTYLE_MODEL_FILE = os.getenv("LIFESTYLE_MODEL_FILE", "heart_rf_lifestyle.pkl")
LIFESTYLE_SCALER_FILE = os.getenv("LIFESTYLE_SCALER_FILE", "heart_scaler_lifestyle.pkl")
LIFESTYLE_TEMPLATE_FILE = os.getenv("LIFESTYLE_TEMPLATE_FILE", "heart_user_template_lifestyle.csv")

REMOTE_MODEL_BASE = os.getenv("REMOTE_MODEL_BASE", "").rstrip("/")

# ============================================================
# Optional Supabase (chat history persistence)
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY and SUPABASE_AVAILABLE:
    try:
        supabase = create_supabase_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized.")
    except Exception as e:
        print("⚠️ Supabase init failed:", e)
        supabase = None
else:
    if SUPABASE_URL or SUPABASE_KEY:
        print("⚠️ Supabase credentials provided but 'supabase' package not available.")
    else:
        print("ℹ️ Supabase not configured; chat history persistence disabled.")

# ============================================================
# AI provider configuration (Groq preferred, OpenAI fallback)
# ============================================================
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

use_groq = bool(GROQ_API_KEY and GROQ_AVAILABLE)
use_openai = bool(OPENAI_API_KEY and OPENAI_AVAILABLE)

if use_groq:
    groq_client = groq.Groq(api_key=GROQ_API_KEY)
    print("✅ Groq configured for chat.")
elif use_openai:
    openai.api_key = OPENAI_API_KEY
    print("✅ OpenAI configured for chat fallback.")
else:
    print("⚠️ No AI provider configured (GROQ or OpenAI). Chat endpoint will return an error unless keys installed.")

groq_client = Groq(api_key=GROQ_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# ============================================================
# Utility: ensure model files exist (remote download option)
# ============================================================
def ensure_model_files(timeout=60):
    required = [
        CLINICAL_MODEL_FILE, CLINICAL_SCALER_FILE, CLINICAL_TEMPLATE_FILE,
        LIFESTYLE_MODEL_FILE, LIFESTYLE_SCALER_FILE, LIFESTYLE_TEMPLATE_FILE,
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if not missing:
        return
    if not REMOTE_MODEL_BASE:
        raise FileNotFoundError(f"Missing model/template files: {missing}. Set REMOTE_MODEL_BASE to auto-download them.")
    print("🛰️ Missing files detected:", missing)
    for fname in missing:
        url = f"{REMOTE_MODEL_BASE}/{fname}"
        try:
            print(f"Downloading {fname} from {url} ...")
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            with open(fname, "wb") as fh:
                fh.write(r.content)
            print("✅", fname)
        except Exception as e:
            print("❌ failed to download", fname, e)

# ensure files on startup
ensure_model_files()

# ============================================================
# Load models & templates
# ============================================================
def load_models():
    print("🔄 Loading models & templates...")
    clinical_model = joblib.load(CLINICAL_MODEL_FILE)
    clinical_scaler = joblib.load(CLINICAL_SCALER_FILE)
    clinical_template_df = pd.read_csv(CLINICAL_TEMPLATE_FILE)
    CLINICAL_FEATURE_COLUMNS = clinical_template_df.columns.tolist()

    lifestyle_model = joblib.load(LIFESTYLE_MODEL_FILE)
    lifestyle_scaler = joblib.load(LIFESTYLE_SCALER_FILE)
    lifestyle_template_df = pd.read_csv(LIFESTYLE_TEMPLATE_FILE)
    LIFESTYLE_FEATURE_COLUMNS = lifestyle_template_df.columns.tolist()

    print("✅ Models loaded")
    return (clinical_model, clinical_scaler, CLINICAL_FEATURE_COLUMNS,
            lifestyle_model, lifestyle_scaler, LIFESTYLE_FEATURE_COLUMNS)

(
    clinical_model, clinical_scaler, CLINICAL_FEATURE_COLUMNS,
    lifestyle_model, lifestyle_scaler, LIFESTYLE_FEATURE_COLUMNS
) = load_models()

# ============================================================
# Input columns (forms)
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
# Heuristic diagnostic rules (your rules, expanded)
# ============================================================
def get_likely_condition(
    age, cholesterol, resting_bp, max_heart_rate,
    fasting_blood_sugar=None, exercise_angina=None,
    chest_pain_type=None, oldpeak=None, st_slope=None,
    sex=None, smoking=None, obesity=None,
    alcohol=None, physical_activity=None
) -> Tuple[str, List[str]]:
    """Return (likely_condition, suggestions) based on simple heuristic rules."""
    likely_condition = "Generalized Cardiac Risk"
    suggestions: List[str] = []
    # normalize
    def to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    age_v = to_float(age, 0)
    chol_v = to_float(cholesterol, 0)
    bp_v = to_float(resting_bp, 0)
    mhr_v = to_float(max_heart_rate, 0)
    old_v = to_float(oldpeak, 0)
    fbs_v = to_float(fasting_blood_sugar, 0)

    # normalize some categorical inputs to lowercase strings for matching
    smoking_s = str(smoking).strip().lower() if smoking is not None else ""
    alcohol_s = str(alcohol).strip().lower() if alcohol is not None else ""
    physical_activity_s = str(physical_activity).strip().lower() if physical_activity is not None else ""
    st_slope_s = str(st_slope).strip().lower() if st_slope is not None else ""

    # Rule set (ordered — earlier matches stronger)
    if (chol_v > 240 and bp_v > 140) or (st_slope_s in ["down", "downsloping", "2"]) or (old_v > 2.0):
        likely_condition = "Coronary Artery Disease (CAD)"
        suggestions = [
            "Adopt a low-fat, high-fiber diet and reduce saturated fats.",
            "Ask your physician about coronary CT angiography or stress testing.",
            "Start supervised, low-to-moderate intensity cardiovascular exercise."
        ]
    elif (mhr_v < 100 and age_v > 60) or (old_v > 1.5 and bp_v > 130):
        likely_condition = "Heart Failure (HF) — possible reduced cardiac output"
        suggestions = [
            "Monitor weight daily and report rapid gains (>2kg in 2 days).",
            "Request an echocardiogram (echo) to evaluate ejection fraction.",
            "Limit high-intensity exertion until evaluated."
        ]
    elif fbs_v > 120 and chol_v > 200:
        likely_condition = "Diabetic Cardiomyopathy risk"
        suggestions = [
            "Tight glycemic control and diet to lower fasting blood sugar.",
            "Get regular ECG and consider echocardiography.",
            "Coordinate care with an endocrinologist and cardiologist."
        ]
    elif mhr_v > 180 or (exercise_angina in [1, "1", "yes", "true"] and mhr_v > 150):
        likely_condition = "Suspected Arrhythmia / Tachycardia"
        suggestions = [
            "Avoid stimulants (caffeine, amphetamines) and alcohol.",
            "Consider ECG, ambulatory Holter monitor or event recorder.",
            "If palpitations are associated with syncope, seek urgent care."
        ]
    elif bp_v >= 160 and age_v > 40:
        likely_condition = "Hypertensive Heart Disease"
        suggestions = [
            "Start/optimize antihypertensive therapy as advised by your clinician.",
            "Daily BP monitoring and salt restriction are recommended.",
            "Evaluate with echocardiography for left ventricular hypertrophy."
        ]
    elif smoking_s in ["1", "yes", "true", "y"] and chol_v > 200:
        likely_condition = "Smoking-related coronary risk"
        suggestions = [
            "Immediate smoking cessation; consider pharmacotherapy (NRT, varenicline).",
            "Full lipid profile and stress testing if symptomatic.",
            "Lifestyle changes: exercise, diet, and smoking cessation programs."
        ]
    elif alcohol_s in ["1", "yes", "true", "y"] and age_v > 35:
        likely_condition = "Alcohol-related cardiomyopathy risk"
        suggestions = [
            "Strict alcohol abstinence and clinical Cardio follow-up.",
            "Check liver function and vitamin B/thiamine levels.",
            "Consider echocardiogram and cardiology referral."
        ]
    elif obesity in ["1", "yes", "true", "y"] or (physical_activity_s in ["low", "none", "0"] and float(bp_v) > 130):
        likely_condition = "Obesity-related cardiometabolic risk"
        suggestions = [
            "Structured weight-loss program and increased physical activity.",
            "Dietary counseling and consider referral to nutritionist.",
            "Check for sleep apnea if obese; treat if present."
        ]
    elif (chol_v < 200 and bp_v < 120 and fbs_v < 100):
        likely_condition = "Low Cardiac Risk"
        suggestions = [
            "Maintain a balanced diet and regular exercise.",
            "Annual routine checkup and continue healthy lifestyle."
        ]
    else:
        likely_condition = "Generalized Cardiac Risk"
        suggestions = [
            "Discuss results with your primary care clinician.",
            "Consider targeted tests (lipid panel, ECG, echo) if concerned."
        ]

    return likely_condition, suggestions

# ============================================================
# Prediction helper (clinical & lifestyle)
# ============================================================
def prepare_and_predict(df_raw: pd.DataFrame, model_type: str) -> pd.DataFrame:
    if model_type not in ("clinical", "lifestyle"):
        raise ValueError("Invalid model_type")

    df = df_raw.copy()

    # If headerless (pandas will provide integer column names), map positional columns
    if all(isinstance(c, int) for c in df.columns):
        if model_type == "clinical":
            df = df.iloc[:, :len(BASE_COLUMNS_CLINICAL)]
            df.columns = BASE_COLUMNS_CLINICAL[:df.shape[1]]
        else:
            df = df.iloc[:, :len(BASE_COLUMNS_LIFESTYLE)]
            df.columns = BASE_COLUMNS_LIFESTYLE[:df.shape[1]]

    # Clinical pipeline: one-hot align with CLINICAL_FEATURE_COLUMNS then scale
    if model_type == "clinical":
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        df_enc = pd.get_dummies(df, columns=cat_cols)
        df_enc = df_enc.reindex(columns=CLINICAL_FEATURE_COLUMNS, fill_value=0)
        X = clinical_scaler.transform(df_enc.values)
        model = clinical_model
        df_out_base = df.copy()
    else:
        # Lifestyle pipeline: ensure numeric & fillna then align
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(df.mean(numeric_only=True))
        df_enc = df.reindex(columns=LIFESTYLE_FEATURE_COLUMNS, fill_value=0)
        X = lifestyle_scaler.transform(df_enc.values)
        model = lifestyle_model
        df_out_base = df.copy()

    # predict probabilities & classes
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        # fallback - sigmoid of decision_function if available
        try:
            df_dec = model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-df_dec))
        except Exception:
            probs = model.predict(X).astype(float)  # last resort

    preds = model.predict(X)

    # Build output
    out_df = df_out_base.copy()
    out_df["Prediction"] = preds
    out_df["Prob_Pos"] = np.round(probs, 4)
    out_df["Risk_Level"] = out_df["Prob_Pos"].apply(lambda p: "High" if p > 0.66 else ("Medium" if p > 0.33 else "Low"))
    return out_df

# ============================================================
# Chat helpers (Groq / OpenAI) + Supabase persistence
# ============================================================
def save_chat_message(user_id: str, role: str, message: str):
    if supabase:
        try:
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "role": role,
                "message": message
            }).execute()
        except Exception as e:
            # don't fail the whole request because of DB issues
            print("⚠️ Supabase insert failed:", e)

def call_groq_chat(user_message: str, system_prompt: Optional[str] = None) -> str:
    # Minimal Groq chat usage - adjust to your groq SDK version
    if not GROQ_API_KEY or not GROQ_AVAILABLE:
        raise RuntimeError("Groq not available/configured")
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        response = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama3-13b"),  # choose desired model
            messages=messages,
            temperature=float(os.getenv("GROQ_TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 512)),
        )
        # the exact response structure may vary — adapt to your groq client
        text = response.choices[0].message["content"]
        return text
    except Exception as e:
        print("Groq chat error:", e)
        raise

def call_openai_chat(user_message: str, system_prompt: Optional[str] = None) -> str:
    if not OPENAI_API_KEY or not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not available/configured")
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 512)),
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        print("OpenAI chat error:", e)
        raise

# ============================================================
# Routes - UI, predict, chat, health
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
                body=f"Name: {name}\nEmail: {email}\n\n{message}"
            )
            mail.send(msg)
            flash("✅ Message sent successfully!", "success")
        except Exception as e:
            print("Mail send error:", e)
            flash("❌ Failed to send message.", "danger")
        return redirect(url_for("contact"))
    return render_template("contact.html")

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
        raw_type = (request.form.get("model_type") or "clinical").lower()
        model_map = {"heart": "clinical", "clinical": "clinical", "cardio": "lifestyle", "lifestyle": "lifestyle"}
        model_type = model_map.get(raw_type, "clinical")

        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename:
            df = pd.read_csv(uploaded_file)
            results = prepare_and_predict(df, model_type)
        else:
            base_cols = BASE_COLUMNS_CLINICAL if model_type == "clinical" else BASE_COLUMNS_LIFESTYLE
            user_data = {}
            for c in base_cols:
                # alias mapping: allow 'sex' -> 'gender' etc.
                val = request.form.get(c)
                if val is None:
                    if c == "gender":
                        val = request.form.get("sex")
                    if c == "cholesterol":
                        val = request.form.get("chol")
                user_data[c] = val
            df = pd.DataFrame([user_data])
            results = prepare_and_predict(df, model_type)

        # Save results CSV
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

        # heuristic inference (use available fields; fallback to 0)
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
            oldpeak=single.get("oldpeak"),
            st_slope=single.get("slope")
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
            model_type=model_type
        )
    except Exception as e:
        print("Prediction error:", e)
        flash(f"Error processing prediction: {str(e)}", "danger")
        return redirect(url_for("form"))

# -------------------
# AI chat endpoint
# -------------------
@app.route("/consult", methods=["POST"])
def consult():
    """
    JSON input:
    { "user_id": "user-123", "message": "I feel chest pain when..." }

    Response:
    { "reply": "...", "saved": true/false }
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "missing 'message' in JSON body"}), 400

    user_msg = data["message"]
    user_id = data.get("user_id", f"anon-{uuid.uuid4().hex[:8]}")
    system_prompt = os.getenv("CHAT_SYSTEM_PROMPT", "You are CardioConsult, a medically informed assistant. Provide safe, conservative guidance and always advise seeing a clinician for definitive diagnosis.")

    # store user message (best-effort)
    try:
        save_chat_message(user_id, "user", user_msg)
    except Exception as e:
        print("Warning: save chat failed:", e)

    # call AI provider
    try:
        if use_groq:
            ai_reply = call_groq_chat(user_msg, system_prompt=system_prompt)
        elif use_openai:
            ai_reply = call_openai_chat(user_msg, system_prompt=system_prompt)
        else:
            return jsonify({"error": "No AI provider configured (set GROQ_API_KEY or OPENAI_API_KEY)."}), 500
    except Exception as e:
        print("AI call failed:", e)
        return jsonify({"error": "AI provider error", "details": str(e)}), 500

    # persist assistant reply
    try:
        save_chat_message(user_id, "assistant", ai_reply)
    except Exception as e:
        print("Warning: save chat failed:", e)

    return jsonify({"reply": ai_reply, "saved": bool(supabase)}), 200

@app.route("/chat", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def chat():
    if request.method == "GET":
        chat_log = session.get("chat_log", [])
        return render_template("chat.html", chat_log=chat_log)

    else:
        data = request.get_json()
        user_message = data.get("message", "")
        chat_log = session.get("chat_log", [])
        chat_log.append({"role": "user", "message": user_message})

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": (
                        "You are CardioConsult, a compassionate AI cardiologist providing preventive health advice. "
                        "Do not give medical diagnoses; only provide general wellness guidance based on cardiovascular science."
                    )},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens = 1000

            )
            reply = response.choices[0].message.content.strip()
            formatted_reply = markdown.markdown(reply, extensions=["fenced_code", "nl2br"])
        except Exception as e:
            print("Groq connection error:", e)
            reply = "⚠️ Sorry, I’m having trouble connecting to my heart consultation engine."

        chat_log.append({"role": "assistant", "message": formatted_reply})
        session["chat_log"] = chat_log[-10:]

        try:
            supabase.table("chat_logs").insert({
                "user_id": session.get("user_id", "guest"),
                "user_message": user_message,
                "bot_reply": formatted_reply,
            }).execute()
        except Exception as e:
            print("Logging error:", e)

        return jsonify({"reply": formatted_reply})

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"].lower()
        password = request.form["password"]

        # Hash the password using bcrypt
        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Check if user already exists
        existing = supabase.table("users").select("id").eq("email", email).execute()
        if existing.data:
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))

        # Insert new user
        supabase.table("users").insert({
            "name": name,
            "email": email,
            "password_hash": hashed_pw
        }).execute()

        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# ================================
# 📊 DASHBOARD
# ================================
@app.route("/user/dashboard")
def user_dashboard():
    if session.get("role") != "user":
        return redirect(url_for("login"))

    user_id = session.get("user_id")
    user_data = supabase.table("users").select("name, email").eq("id", user_id).execute().data[0]
    records = supabase.table("records").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data

    # Prepare data for charts
    if records:
        labels = [r["created_at"][:10] for r in records]
        scores = [r["health_score"] for r in records]
    else:
        labels, scores = [], []

    return render_template(
        "user_dashboard.html",
        user=user_data,
        records=records,
        chart_labels=json.dumps(labels),
        chart_scores=json.dumps(scores)
    )


@app.route("/admin/dashboard")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("login"))

    total_users = len(supabase.table("users").select("id").execute().data)
    total_records = len(supabase.table("records").select("id").execute().data)
    total_chats = len(supabase.table("chat_logs").select("id").execute().data)
    admins = supabase.table("admins").select("name, email").execute().data

    # Aggregate data for chart (records by consultation type)
    record_data = supabase.table("records").select("consultation_type").execute().data
    type_count = {}
    for r in record_data:
        ctype = r.get("consultation_type", "Unknown")
        type_count[ctype] = type_count.get(ctype, 0) + 1

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_records=total_records,
        total_chats=total_chats,
        admins=admins,
        record_type_labels=json.dumps(list(type_count.keys())),
        record_type_counts=json.dumps(list(type_count.values()))
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].lower()
        password = request.form["password"].encode("utf-8")

        # Try user first
        user_data = supabase.table("users").select("*").eq("email", email).execute()
        if user_data.data:
            user = user_data.data[0]
            stored_hash = user["password_hash"].encode("utf-8")
            if bcrypt.checkpw(password, stored_hash):
                session["user_id"] = user["id"]
                session["role"] = "user"
                flash("Welcome to CardioConsult!", "success")
                return redirect(url_for("user_dashboard"))

        # Try admin next
        admin_data = supabase.table("admins").select("*").eq("email", email).execute()
        if admin_data.data:
            admin = admin_data.data[0]
            stored_hash = admin["password_hash"].encode("utf-8")
            if bcrypt.checkpw(password, stored_hash):
                session["admin_id"] = admin["id"]
                session["role"] = "admin"
                flash("Admin login successful!", "success")
                return redirect(url_for("admin_dashboard"))

        flash("Invalid email or password", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully.")
    return redirect(url_for("login"))




    
@app.route("/api/chat", methods=["POST"])
@limiter.limit("5 per minute")  # tighter limit for chat API
def ai_chat():
    user_input = request.json.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Retrieve session chat
    history = session.get("chat_history", [])
    history.append({"role": "user", "content": user_input})

    # Prepare Groq request
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mixtral-8x7b-32768",  # free + hosted
                "messages": [
                    {"role": "system", "content": "You are CardioConsult, an AI cardiovascular consultant. Provide informative, safe health responses and always encourage users to see a doctor if symptoms persist."},
                    *history[-10:]  # only recent context for efficiency
                ],
                "temperature": 0.7
            },
            timeout=15
        )
        data = response.json()
        ai_reply = data["choices"][0]["message"]["content"]

        # Update chat session
        history.append({"role": "assistant", "content": ai_reply})
        session["chat_history"] = history

        # Basic logging (could log to DB instead)
        print(f"[CHAT LOG] User: {user_input}\nAI: {ai_reply}\n---")

        return jsonify({"reply": ai_reply})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "CardioConsult AI is currently unavailable. Please try again later."}), 500


# -----------------------------
# Utility: Clear chat
# -----------------------------
@app.route("/chat/clear")
def clear_chat():
    session.pop("chat_history", None)
    flash("Chat history cleared.")
    return redirect(url_for("chat"))

@app.route("/chat-history/<user_id>", methods=["GET"])
def chat_history(user_id):
    if not supabase:
        return jsonify({"error": "Chat history persistence is not configured."}), 400
    try:
        res = supabase.table("chat_history").select("*").eq("user_id", user_id).order("created_at", desc=False).execute()
        return jsonify({"history": res.data}), 200
    except Exception as e:
        print("Supabase fetch failed:", e)
        return jsonify({"error": "Failed to fetch history", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()}), 200

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    # For production use gunicorn (or fly/gunicorn)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "False") == "True")
