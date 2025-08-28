# Heart-Disease-Predictor-Devtown
This is a project given to me by DevTown India after a five day Bootcamp on Using AI for Heart-Disease-Prediction


## 📌 Project Title

**Heart Disease Prediction Model**

---

## 📖 Overview

This project uses machine learning techniques to predict the likelihood of heart disease in patients based on medical attributes.
It demonstrates the full pipeline of **data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment**.

---

## 🎯 Objectives

* Analyze patient health data.
* Identify key risk factors influencing heart disease.
* Build predictive models to classify patients at risk.
* Provide a reproducible framework for future healthcare-related ML projects.

---

## 📊 Dataset

* **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) (or any other source used).
* **Size:** 303 rows × 14 columns.
* **Features include:**

  * Age
  * Sex
  * Chest pain type
  * Resting blood pressure
  * Cholesterol level
  * Fasting blood sugar
  * Resting ECG results
  * Maximum heart rate
  * Exercise-induced angina
  * Oldpeak (ST depression)
  * Slope of ST segment
  * Number of major vessels colored by fluoroscopy
  * Thalassemia
  * Target (1 = heart disease, 0 = no disease)

---

## ⚙️ Tech Stack

* **Languages:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn and Scikit-Learn
* **Tools:** Google Colab

---

## 🔎 Exploratory Data Analysis (EDA)
* Correlation heatmap to find strong predictors of heart disease.
  

---

## 🤖 Model Building

* Preprocessing: handling missing values, encoding categorical features, scaling numerical data.
* Models tested: Logistic Regression, Random Forest, KNeighborsClassifier.
* Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
* Best performing model: (**RandomForestClassifier with 88% accuracy**).

---

## 🚀 How to Run the Project

Certainly! Here are the **numbered steps** to run your GitHub project (**ibnmikhail48/Heart-Disease-Predictor-Devtown**) on Google Colab:

---

### Steps to Run Your Project on Google Colab

1. **Open Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Start a new notebook.

2. **Clone Your Repository**
   - In the first cell, enter:
     ```python
     !git clone https://github.com/ibnmikhail48/Heart-Disease-Predictor-Devtown.git
     ```

3. **Change Directory to Your Project Folder**
   - In the next cell, enter:
     ```python
     %cd Heart-Disease-Predictor-Devtown
     ```

4. **Install Project Requirements**
   - If your repo has a `requirements.txt` file:
     ```python
     !pip install -r requirements.txt
     ```
   - Or, manually install required packages (if needed):
     ```python
     !pip install pandas numpy scikit-learn matplotlib seaborn
     ```

5. **Upload or Mount Data Files (if needed)**
   - To upload local files:
     ```python
     from google.colab import files
     files.upload()
     ```
   - Or mount Google Drive:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

6. **Run Your Project Code**
   - Find the main script (e.g., `main.py`, `app.py`, or relevant notebook).
   - To run a Python script:
     ```python
     !python main.py
     ```
   - Or, open and execute the notebook code cells.

7. **Experiment and Modify**
   - Add new cells, edit code, and interact with outputs as needed.

---

**You can copy and paste these steps into your Colab notebook for easy reference!**---

## 📈 Results

* The best model achieved:

  * **Accuracy:** 88%
  * **Precision:** 87%
  * **Recall:** 90%
  * **F1-Score:88%
  * **ROC-AUC:** 0.92%

* Important features influencing predictions: age, chest pain type, cholesterol, maximum heart rate, etc.

---

## 🏥 Use Cases

* Early detection of heart disease risk.
* Decision support system for healthcare professionals.
* Educational tool for machine learning in medicine.

---

## 📂 Project Structure

```heart-disease-prediction/
│── README.md            
│── heart-disease-predictor-devtown.ipynb          
```

---

## 📌 Future Improvements

* Collect more patient data for better generalization.
* Deploy the model as a full web or mobile application.
* Integrate with real-time hospital systems.
* Apply deep learning models for comparison.

---

## Contact and Support 
Developer: [Mikail Muhammed Adekunle]
* Email: ibnmikhail48@gmail.com
* Linkedln: Mikail Muhammed
* Github: ibnmikhail48
---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

