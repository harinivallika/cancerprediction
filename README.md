# 🧬 OncoPedict — Cancer Risk Prediction App

A full-stack ML web app for educational cancer risk prediction.

## Supported Cancer Types
| # | Type | Model | Features |
|---|------|-------|----------|
| 1 | 🎗️ Breast Cancer | Random Forest | 10 |
| 2 | 🫁 Lung Cancer | Gradient Boosting | 10 |
| 3 | 🩸 Diabetes / Pancreatic Risk | Logistic Regression | 8 |
| 4 | 🔬 Skin Cancer | Random Forest | 10 |
| 5 | 🩺 Cervical Cancer | Gradient Boosting | 10 |
| 6 | ⚕️ Prostate Cancer | Random Forest | 10 |

---

## 🚀 Quick Start (Windows)

### Step 1 — Start the Backend
```
cd backend
Double-click  run_backend.bat
```
This will:
- Create a Python virtual environment
- Install all dependencies (scikit-learn 1.2.2, Flask 2.3.3, etc.)
- Start the Flask API at http://localhost:5000

### Step 2 — Open the Frontend
```
Open  frontend/index.html  in Chrome, Firefox, or Edge
```
✅ No Node.js required — pure HTML/JS frontend!

---

## 📁 File Structure
```
cancer-prediction-app/
├── backend/
│   ├── app.py              ← Flask REST API + 6 trained ML models
│   ├── requirements.txt    ← Python dependencies (fixed versions)
│   └── run_backend.bat     ← One-click Windows startup script
├── frontend/
│   └── index.html          ← Complete standalone frontend (no build needed)
└── README.md
```

---

## 🔌 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | Check backend status |
| GET | /api/cancer-types | List all cancer types |
| GET | /api/features/<type> | Get input feature definitions |
| POST | /api/predict | Run ML prediction |
| GET | /api/datasets | Dataset information & links |

### Example POST /api/predict
```json
Request:
{
  "cancer_type": "breast",
  "features": [14.0, 19.0, 92.0, 655.0, 0.096, 0.104, 0.089, 0.049, 0.181, 0.063]
}

Response:
{
  "risk_score": 23.4,
  "risk_level": "Low",
  "risk_color": "#22c55e",
  "advice": "Your risk indicators appear low...",
  "top_features": [...],
  "disclaimer": "Educational model only..."
}
```

---

## 📊 Datasets
| Cancer | Dataset | Source |
|--------|---------|--------|
| Breast | Wisconsin Breast Cancer | sklearn built-in / UCI |
| Lung | Survey Lung Cancer | Kaggle |
| Diabetes | Pima Indians Diabetes | UCI / Kaggle |
| Skin | HAM10000 Skin Lesion | ISIC / Kaggle |
| Cervical | Cervical Cancer Risk Factors | UCI |
| Prostate | Prostate Cancer Dataset | Kaggle |

---

## ⚠️ Disclaimer
This app is strictly for **educational purposes only**.  
It is **NOT** a medical device and should **never** replace professional medical advice.
