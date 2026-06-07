# 🛡️ ITSM Incident ML Prediction System
An end-to-end Machine Learning project to predict IT incident priority, detect high-priority tickets, forecast incident volume, and flag RFC requests.

## 🚀 Live Demo
🎯 **Streamlit:** https://itsm-incident-ml-mythili.streamlit.app/

## 🔍 Project Overview
This project applies ML across **4 ITSM tasks** using a dataset of ~46k IT incidents (2012–2014):

- **Task 1** — Predict High Priority (P1/P2) tickets for early escalation
- **Task 2** — Forecast monthly incident volume for resource planning
- **Task 3** — Auto-tag tickets with correct priority (P2–P5)
- **Task 4** — Predict whether a ticket will trigger a Request for Change (RFC)

Models compared:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

## 📁 Project Structure

| File | Description |
|------|-------------|
| `PRCL_0012_ITSM_ML_corrected.ipynb` | ML notebook with EDA and model training |
| `app.py` | Streamlit web application |
| `itsm_priority_model.pkl` | Task 3 — XGBoost Priority Auto-Tag model |
| `itsm_highpriority_model.pkl` | Task 1 — High Priority binary classifier |
| `itsm_scaler.pkl` | StandardScaler for inference |
| `itsm_rfc_model.pkl` | Task 4 — RFC prediction model |
| `requirements.txt` | Required libraries |

## 🛠️ Libraries Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit
- Matplotlib, Seaborn
- SHAP

## 🚀 How to Run Locally
```bash
git clone https://github.com/Mythili-Pandiyarajan/itsm-incident-ml-prediction.git
cd itsm-incident-ml-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Model Performance

### Task 1 — High Priority Detection (Binary)
| Model | Accuracy | F1 Weighted |
|-------|----------|-------------|
| Logistic Regression | 0.95 | 0.85 |
| Decision Tree | 0.97 | 0.97 |
| Random Forest | 0.98 | 0.98 |
| XGBoost | 0.98 | 0.98 |

✅ **Best Model: XGBoost** with F1 Weighted of 0.98

### Task 3 — Priority Auto-Tag (Multi-class)
| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| Logistic Regression | 0.67 | 0.56 |
| Decision Tree | 0.78 | 0.66 |
| Random Forest | 0.80 | 0.68 |
| XGBoost | 0.82 | 0.71 |

✅ **Best Model: XGBoost Baseline** with Accuracy 0.82 and F1 Macro 0.71

### Task 4 — RFC Prediction (Binary)
| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| Random Forest | 0.98 | 0.68 |

### Task 2 — Volume Forecasting
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Random Forest Regressor | 205.93 | 237.94 | 0.31 |

## 📌 Dataset
- ~46,000 IT incident records from 2012–2014
- Source: MySQL ITSM Database
- 26 fields: CI details, priority, impact, urgency, timestamps, closure codes, reassignment counts

## 👩‍💻 Author
**Mythili Pandiyarajan** — [GitHub Profile](https://github.com/Mythili-Pandiyarajan)
