# Gravity of Debt: Credit Risk Prediction Engine

A complete end-to-end Machine Learning and Data Science project for predicting credit risk using the LendingClub dataset. This project includes data generation, extensive feature engineering, model comparison, explainability with SHAP, a FastAPI backend, and a Streamlit dashboard.

## 📁 Project Structure

```text
gravity_of_debt/
├── data/
│   └── lending_club_sample.csv   ← Generated 300K row dataset
├── notebooks/
│   └── 01_eda_and_modeling.ipynb ← Exploratory Data Analysis & Modeling
├── src/
│   ├── generate_data.py          ← Script to generate realistic data
│   ├── preprocess.py             ← Scikit-learn preprocessing pipelines
│   ├── train.py                  ← Model training and evaluation
│   ├── explain.py                ← SHAP explainability and risk logic
│   └── utils.py                  ← Helper metrics and functions
├── models/
│   └── best_model.pkl            ← Best performing model (LightGBM/XGBoost)
├── main.py                       ← FastAPI application
├── dashboard.py                  ← Streamlit frontend
├── requirements.txt              ← Python dependencies
└── README.md                     ← Project documentation
```

## 🚀 Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the Dataset**
   ```bash
   python src/generate_data.py
   ```
   *Generates `data/lending_club_sample.csv` with realistic correlations and distributions.*

3. **Train the Models**
   ```bash
   python src/train.py
   ```
   *Trains Logistic Regression, Random Forest, XGBoost, and LightGBM. The best model is saved to `models/best_model.pkl`.*

4. **Run the API**
   ```bash
   python -m uvicorn main:app --reload
   ```

5. **Run the Dashboard**
   ```bash
   python -m streamlit run dashboard.py
   ```

## 🔌 API Usage Examples

**Health Check**
```bash
curl -X GET "http://localhost:8000/health"
```

**Predict Risk**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
          "loan_amnt": 15000,
          "int_rate": 12.0,
          "installment": 400,
          "grade": "B",
          "sub_grade": "B2",
          "emp_length": "5 years",
          "home_ownership": "MORTGAGE",
          "annual_inc": 75000,
          "verification_status": "Verified",
          "purpose": "debt_consolidation",
          "dti": 18.0,
          "delinq_2yrs": 0,
          "fico_range_low": 700,
          "fico_range_high": 704,
          "open_acc": 10,
          "pub_rec": 0,
          "revol_bal": 15000,
          "revol_util": 50.0,
          "total_acc": 20
        }'
```

**Expected Response**:
```json
{
  "default_probability": 0.0845,
  "risk_level": "LOW",
  "top_reasons": [
    "fico_avg decreased the risk score by 0.354",
    "dti_bin_Medium increased the risk score by 0.125",
    "annual_inc decreased the risk score by 0.098",
    "revol_util increased the risk score by 0.054",
    "grade_B decreased the risk score by 0.045"
  ],
  "shap_values": {
    "fico_avg": -0.354,
    "dti_bin_Medium": 0.125,
    ...
  }
}
```

## 🖼️ Dashboard Screenshots

*(Placeholder for dashboard screenshot)*

---
*Built with Scikit-Learn, XGBoost, LightGBM, SHAP, FastAPI, and Streamlit.*
