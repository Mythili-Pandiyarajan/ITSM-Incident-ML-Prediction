# 🛡️ ITSM Incident ML Prediction System

> Machine Learning system for IT Service Management — predicts incident priority, detects high-priority tickets before SLA breach, forecasts incident volume, and flags RFC-triggering tickets.

---

## 📌 Problem Statement

A mid-size IT organisation receives **22–25k incidents/year** managed under the ITIL framework. Despite mature processes, customer satisfaction with incident management was rated poor. This project applies ML across 4 key areas to improve ITSM operations:

| Task | Objective |
|------|-----------|
| **Task 1** | Predict High Priority (P1/P2) tickets for early escalation |
| **Task 2** | Forecast monthly incident volume for resource planning |
| **Task 3** | Auto-tag tickets with correct priority (P2–P5) |
| **Task 4** | Predict whether a ticket will trigger a Request for Change (RFC) |

---

## 🤖 Models & Results

| Task | Best Model | Accuracy | Key Metric |
|------|-----------|----------|------------|
| Task 1 — High Priority Detection | XGBoost | 0.98 | F1 Weighted: 0.98 |
| Task 2 — Volume Forecasting | Random Forest Regressor | — | R²: 0.31 · MAE: 206 |
| Task 3 — Priority Auto-Tag | XGBoost Baseline | 0.82 | F1 Macro: 0.71 |
| Task 4 — RFC Prediction | Random Forest | 0.98 | F1 Macro: 0.68 |

**Recommended for production:** XGBoost Baseline (Task 3) — highest F1 Macro across all classification models, with SHAP explainability for IT manager trust and auditability.

---

## 📂 Project Structure

```
ITSM-Incident-ML-Prediction/
│
├── PRCL_0012_ITSM_ML_corrected.ipynb   # Full ML notebook (EDA → Models → SHAP)
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
│
├── itsm_priority_model.pkl              # Task 3 — XGBoost Priority Auto-Tag
├── itsm_highpriority_model.pkl          # Task 1 — XGBoost High Priority Binary
├── itsm_scaler.pkl                      # StandardScaler (required for inference)
└── itsm_rfc_model.pkl                   # Task 4 — RFC Random Forest
```

---

## 🗂️ Dataset

- ~46,000 incident records from 2012–2014
- 26 fields: CI details, priority, impact, urgency, timestamps, closure codes, reassignment counts
- Source: MySQL ITSM database
- Key preprocessing: European comma decimal fix on `Handle_Time_hrs`, frequency encoding on `CI_Name`, datetime feature extraction, SMOTE for class imbalance

---

## 🔧 Feature Engineering

| Feature | Description |
|---------|-------------|
| `CI_Name_freq` | Frequency of CI in dataset — top SHAP feature |
| `Open_Hour` | Hour ticket was opened |
| `Is_BusinessHour` | 1 if opened 9am–6pm |
| `Is_Weekend` | 1 if Saturday/Sunday |
| `Handle_Time_hrs` | Cleaned from European comma decimal format |
| `No_of_Reassignments` | Strong signal for complex/high-priority tickets |
| `Closure_Code` | Encoded — second strongest SHAP feature |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/ITSM-Incident-ML-Prediction.git
cd ITSM-Incident-ML-Prediction

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 🖥️ Streamlit App — 4 Tabs

**🔮 Predict** — Enter ticket details and get real-time predictions from all 3 models simultaneously: high-priority alert with confidence score, priority auto-tag (P2–P5) with probability chart, and RFC detection.

**📊 Analyze** — Upload ITSM CSV for instant EDA: KPI cards, priority distribution, monthly trend, handle time analysis, and batch prediction across the full dataset.

**📈 Forecast** — Upload CSV to run the volume forecasting model with actual vs predicted chart, quarterly forecast table, and annual forecast summary.

**🗂️ Dashboard** — Project summary, all-model comparison charts, and actionable IT manager recommendations.

---

## 📊 Notebook Structure

The notebook follows a structured ML workflow across 19 sections:

1. Import Libraries
2. Load Dataset (MySQL connection)
3. Basic Checks
4. Exploratory Data Analysis (Univariate, Bivariate, Multivariate)
5. Data Preprocessing (Missing values, Outliers, Feature Engineering)
6. Task 1 — High Priority Binary Classification (4 models)
7. Task 3 — Priority Auto-Tag Multi-class (4 models)
8. Hyperparameter Tuning (GridSearchCV)
9. Cross Validation
10. Overfitting Check
11. Confusion Matrix
12. Feature Importance
13. SHAP Explainability
14. Task 2 — Volume Forecasting
15. Task 4 — RFC Prediction
16. Model Comparison Report
17. IT Manager Recommendations
18. Challenges & Solutions
19. Conclusion & Model Saving

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

- **ML:** XGBoost, Random Forest, Decision Tree, Logistic Regression
- **Imbalance handling:** SMOTE (imbalanced-learn)
- **Explainability:** SHAP TreeExplainer
- **UI:** Streamlit
- **Data:** MySQL (pymysql), pandas

---

## 💡 Key Findings

- **CI_Name_freq** is the strongest SHAP feature — frequently failing CIs predict priority level better than Impact/Urgency alone
- **Closure_Code** is the second strongest — how a ticket closes correlates strongly with its priority class
- **XGBoost Baseline beat all tuned models** — data quality is the ceiling, not hyperparameters
- **SMOTE with k_neighbors=1** was required for Priority 2 — only ~139 samples even after oversampling
- **Task 2 R² of 0.31** is expected — only 36 monthly data points (2012–2014); ARIMA/LSTM require 50+ points minimum

---

## 📋 IT Manager Recommendations

1. **Deploy Task 1 model at ticket creation** — auto-escalate P1/P2 predictions to on-call engineers before SLA breach
2. **Use Task 3 for auto-routing** — eliminates manual priority tagging and reduces reassignment delays
3. **Use Task 2 for quarterly staffing** — MAE of 206 tickets/month is sufficient for rough headcount planning
4. **Use Task 4 for proactive change management** — pre-prepare RFC workflow for flagged tickets
5. **Monitor high-frequency CIs** — top SHAP feature; these CIs should be flagged for problem management review
