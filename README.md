# CardioGuard — Heart Disease & Lifestyle Risk Predictor (Dual-Model Flask App)

**HeartPredict** is a polished, production-ready Flask web application that performs **two types of cardiovascular risk predictions**:

* **Clinical model** — trained on clinical features (Cleveland-style dataset).
* **Lifestyle model** — trained on general / lifestyle features (`cardio_train.csv` style: height, weight, smoking, alcohol, activity, BP, etc.)

This repository contains the web app, training scripts, model artifacts, and UI templates (Tailwind) to upload CSVs or enter single-patient data, view downloadable results, and visualize risk/confidence. The project is built to be extended into dashboards, subscription/payments, and an AI CardioConsultant (RAG/agentic) in future releases.

---

## Table of contents

- [CardioGuard — Heart Disease \& Lifestyle Risk Predictor (Dual-Model Flask App)](#cardioguard--heart-disease--lifestyle-risk-predictor-dual-model-flask-app)
  - [Table of contents](#table-of-contents)
  - [Highlights](#highlights)
  - [Repository structure](#repository-structure)
  - [Quickstart — Run Locally](#quickstart--run-locally)
  - [Model training \& artifacts](#model-training--artifacts)
  - [Endpoints \& Input formats](#endpoints--input-formats)
  - [Deployment (Render) \& keep-alive](#deployment-render--keep-alive)
  - [Environment variables (`.env.example`)](#environment-variables-envexample)
  - [UI / Templates \& Static assets](#ui--templates--static-assets)
  - [Troubleshooting \& FAQs](#troubleshooting--faqs)
  - [Testing \& Sample Data](#testing--sample-data)
  - [Security, Privacy \& Compliance](#security-privacy--compliance)
  - [Roadmap — Upcoming Features (planned)](#roadmap--upcoming-features-planned)
    - [User-facing](#user-facing)
    - [AI \& Advanced](#ai--advanced)
  - [Contribution](#contribution)
  - [License \& Credits](#license--credits)
  - [Contact](#contact)
  - [Examples \& Quick Tips](#examples--quick-tips)

---

## Highlights

* Dual-model architecture (clinical & lifestyle) so your users can choose the most appropriate predictor.
* Upload CSVs (batch) or fill a single-person form (manual).
* Tailwind-based UI: `/`, `/form`, `/result`, `/about`, `/contact`.
* Results saved to `static/results/` and downloadable.
* Health endpoint for uptime monitors: `/health`.
* Designed for Render deployment (Procfile + gunicorn ready).
* Clear extension points for dashboards, subscriptions, payments, and AI assistants.

---

## Repository structure

```
heartpredict/
├── app.py                      # Flask app (loads both models, routes, predict logic)
├── train/
│   ├── train_clinical.py       # (optional) training script for clinical model
│   └── train_lifestyle.py      # (optional) training script for lifestyle model
├── models/
│   ├── heart_rf_clinical.pkl
│   ├── heart_scaler_clinical.pkl
│   ├── heart_user_template_clinical.csv
│   ├── heart_rf_lifestyle.pkl
│   ├── heart_scaler_lifestyle.pkl
│   └── heart_user_template_lifestyle.csv
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── form.html
│   ├── result.html
│   ├── about.html
│   ├── contact.html
│   └── resources.html
├── static/
│   ├── assets/
│   └── results/
├── requirements.txt
├── Procfile
├── runtime.txt (optional)
├── .env.example
└── README.md
```

---

## Quickstart — Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/Miftau/CardioGuard
cd CardioGuard
```

2. **Create & activate virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add model files**
   Ensure the `models/` directory contains:

* `heart_rf_clinical.pkl`, `heart_scaler_clinical.pkl`, `heart_user_template_clinical.csv`
* `heart_rf_lifestyle.pkl`, `heart_scaler_lifestyle.pkl`, `heart_user_template_lifestyle.csv`

(If you don't have them, run the training scripts in `heart_disease_predictor.py` or contact the maintainer.)

5. **Create `.env`** (or copy `.env.example`) and add secrets (see below).

6. **Run app**

```bash
python app.py
# or for production-like run locally:
gunicorn app:app --bind 0.0.0.0:5000
```

7. Visit `http://localhost:5000/form` to use the app.

---

## Model training & artifacts

* Training scripts are placed in `heart_disease_predictor.py`. They produce:

  * model `.pkl` (RandomForest or other)
  * scaler `.pkl` (StandardScaler)
  * template CSV with final columns used (helps align incoming CSVs).
* Filenames used in the app (change in `app.py` if you rename them):

  * Clinical: `heart_rf_clinical.pkl`, `heart_scaler_clinical.pkl`, `heart_user_template_clinical.csv`
  * Lifestyle: `heart_rf_lifestyle.pkl`, `heart_scaler_lifestyle.pkl`, `heart_user_template_lifestyle.csv`

**Training notes**

* Clinical model expects 13 feature inputs (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal).
* Lifestyle model expects columns like: `age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active`.
* Ensure you preprocess and align columns exactly as the app expects; training scripts should save the final columns (`template.csv`) used.

---

## Endpoints & Input formats

**Web UI**

* `/form` — unified form to choose model (Clinical / Lifestyle), choose CSV/manual, submit.
* `/predict` (POST) — receives form submission or CSV upload. Renders `result.html`.
* `/result` — rendered by `/predict`. Not a direct endpoint (result page returned).
* `/health` — returns `{"status": "ok"}` for uptime monitors.

**CSV format examples**

* **Clinical (no headers required; order is important if headerless)** — 13 columns:

```
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
```

* **Lifestyle (no headers required; order is important if headerless)**:

```
age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active
```

**API-style curl (manual single patient example)**

```bash
curl -X POST http://localhost:5000/predict \
  -F "model_type=clinical" \
  -F "age=63" -F "sex=1" -F "cp=1" -F "trestbps=145" \
  -F "chol=233" -F "fbs=1" -F "restecg=2" -F "thalach=150" \
  -F "exang=0" -F "oldpeak=2.3" -F "slope=3" -F "ca=0" -F "thal=6"
```

**CSV upload (curl)**

```bash
curl -X POST http://localhost:5000/predict \
  -F "model_type=cardio" \
  -F "file=@/path/to/cardio_test.csv"
```

---

## Deployment (Render) & keep-alive

**Procfile**

```
web: gunicorn app:app
```

**requirements.txt** should contain at least:

```
Flask
gunicorn
pandas
numpy
scikit-learn
joblib
flask-mail
python-dotenv
plotly
```

**Keep the app always-on**

* Add a `/health` endpoint (already included).
* Use an external uptime monitor (UptimeRobot, Cron-job.org) to ping `https://your-app.onrender.com` every 5–10 minutes. This is the most reliable method for free hosting.
* Optional: self-ping thread *may* be used but is less reliable on some hosts.

**Render-specific tips**

* Keep the model files in repo or on mounted storage. Large models may require storing on an object store (S3) and downloading on startup.
* Use `gunicorn` in Procfile.
* Configure environment variables in Render dashboard.

---

## Environment variables (`.env.example`)

```
FLASK_SECRET=super-secret-key
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=app-or-smtp-password
MAIL_DEFAULT_RECEIVER=admin@yourdomain.com
PORT=5000
```

> Never commit `.env` or secrets to git. Use the Render/host provider secret manager.

---

## UI / Templates & Static assets

* Templates are Jinja2 in `/templates`. They use Tailwind (CDN).
* Use `{{ url_for('static', filename='assets/my.png') }}` to reference images.
* Put static files in `static/assets/` and results in `static/results/`.

**Image issue on production?**

* Ensure `static` folder is at repo root same level as `app.py`.
* Reference via `url_for('static', filename='assets/your.png')`.
* Check Render logs for 404s (case-sensitive filenames on Linux).

---

## Troubleshooting & FAQs

**1. “Form reloads but no result”**

* Confirm `model_type` value from the form matches backend mapping. App accepts `heart|clinical` → clinical and `cardio|lifestyle` → lifestyle.
* Check server logs for printed messages or error stack traces.

**2. “StandardScaler expects N features”**

* This means incoming data columns do not match the scaler used in training. Use the template CSV produced in `/models` to align columns or ensure manual form matches expected feature set for selected model.

**3. “ValueError: could not convert string to float: '?'”**

* Replace `?` with `NaN` before numeric coercion. Training and preprocess scripts should include `na_values='?'` and numeric coercion with `pd.to_numeric(..., errors='coerce')`.

**4. Missing model files**

* Place your model artifacts into `/models` or update `app.py` configuration to point to correct locations.

---

## Testing & Sample Data

* Add a `tests/` folder and unit tests for `prepare_and_predict()` logic (pandas-based).
* Create sample CSVs:

  * `sample_clinical.csv` (one row) matching clinical columns
  * `sample_cardio.csv` matching lifestyle columns
* Use the saved `heart_user_template_*.csv` as canonical input format for users.

---

## Security, Privacy & Compliance

* **Data privacy**: Patient data is extremely sensitive. Do **not** log or store PII unnecessarily.
* **Transport**: Always serve the app over HTTPS in production.
* **Storage**: If storing results, ensure access controls and retention policies. Consider encrypting results at rest.
* **Regulatory**: This app is not medical advice. Add clear disclaimers. If you intend clinical usage, consult legal and regulatory frameworks (HIPAA, GDPR, local health regulations).

---

## Roadmap — Upcoming Features (planned)

> These features are **planned** and **not yet implemented**. They are included here so contributors and stakeholders know the direction.

### User-facing

* **User Dashboard** — per-user history of predictions, filterable with date ranges and downloadable reports.
* **Charts & Visualizations** — time-series, risk trends, feature importance per user (Plotly/Chart.js).
* **Account Management** — sign-up, sign-in, password reset (Auth via provider or custom).
* **Subscription Plans & Payments** — integrate Paystack / Stripe for freemium / paid tiers, recurring subscriptions.
* **Admin Dashboard** — manage users, view aggregated stats, moderate results, push notifications.
* **Notification System** — email alerts or SMS when risk crosses thresholds.

### AI & Advanced

* **Agentic CardioConsultant (RAG)** — use Retrieval-Augmented Generation to combine system knowledge, user history, and medical resources to answer user queries.

  * Integrate a vector DB (e.g., FAISS, Milvus, Pinecone) for RAG.
  * Use embeddings and a LLM to provide explanations, references, and personalized advice.
  * Provide audit trails of AI suggestions.
* **Voice & Conversational UI** — speech-to-text + text-to-speech, allowing voice queries and responses for CardioConsultant.
* **Model Explainability** — SHAP/LIME explanations for single predictions (feature contribution visualization).
* **Real-time Monitoring & Drift Detection** — monitor model performance and data drift on incoming data.

---

## Contribution

Contributions are welcome! Suggested workflow:

1. Fork repo
2. Create feature branch `feature/awesome`
3. Open PR with a clear description and tests
4. Follow commit & PR conventions

Please open issues for bugs or feature requests.

---

## License & Credits

* **License:** MIT (or choose your preferred OSS license).
* **Acknowledgements:** UCI Heart datasets, Kaggle Cardio dataset, Plotly, TailwindCSS, Flask community.

---

## Contact

If you want help extending this project, integrating payments, or building the CardioConsultant AI, reach out:

* Email: `support@heartpredict.ai` (configured via `.env`)
* GitHub: `<your-github-url>`

---

## Examples & Quick Tips

**1. Ensure frontend and backend model keys match**

* The `form` selects `heart` or `cardio`. `app.py` maps these into `clinical` and `lifestyle`. Keep these mappings consistent if you change names.

**2. Keep `heart_user_template_*.csv` with model artifacts**

* Use template CSV to generate sample upload files for users. This prevents scaler/feature mismatch errors.

**3. Keep results clean**

* `static/results` stores CSV outputs. Consider a scheduled job to purge older files or move to S3 for a production app.


