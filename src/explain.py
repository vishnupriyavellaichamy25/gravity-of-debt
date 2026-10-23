

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "credit_risk_model.joblib"


# --------------------------------------------------
# 2. INPUT FEATURES
# --------------------------------------------------

NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]

CATEGORICAL_FEATURES = [
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# 3. TRAIN DEMONSTRATION MODEL
# --------------------------------------------------

def train_demo_model():
    """
    Train a demonstration model using synthetic data.

    This model is intended for software testing only.
    It must not be used for actual lending decisions.
    """

    rng = np.random.default_rng(42)

    n = 3000

    data = {
        "loan_amnt": rng.integers(1000, 40000, n),
        "int_rate": rng.uniform(5, 30, n),
        "installment": rng.uniform(50, 1500, n),
        "annual_inc": rng.uniform(15000, 150000, n),
        "dti": rng.uniform(0, 45, n),
        "delinq_2yrs": rng.integers(0, 5, n),
        "fico_range_low": rng.integers(600, 850, n),
        "fico_range_high": rng.integers(605, 855, n),
        "open_acc": rng.integers(1, 30, n),
        "pub_rec": rng.integers(0, 4, n),
        "revol_bal": rng.uniform(0, 50000, n),
        "revol_util": rng.uniform(0, 120, n),
        "total_acc": rng.integers(2, 60, n),

        "grade": rng.choice(list("ABCDEFG"), n),

        "sub_grade": rng.choice(
            [
                f"{grade}{number}"
                for grade in "ABCDEFG"
                for number in range(1, 6)
            ],
            n,
        ),

        "emp_length": rng.choice(
            [
                "< 1 year",
                "1 year",
                "2 years",
                "3 years",
                "4 years",
                "5 years",
                "6 years",
                "7 years",
                "8 years",
                "9 years",
                "10+ years",
            ],
            n,
        ),

        "home_ownership": rng.choice(
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
            n,
        ),

        "verification_status": rng.choice(
            ["Verified", "Source Verified", "Not Verified"],
            n,
        ),

        "purpose": rng.choice(
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "small_business",
                "car",
                "medical",
                "other",
            ],
            n,
        ),
    }

    df = pd.DataFrame(data)

    # Artificial target-generation formula for demonstration only.
    score = (
        (df["int_rate"] - 15) * 0.08
        + (df["dti"] - 20) * 0.04
        - (df["fico_range_low"] - 700) * 0.025
        + df["delinq_2yrs"] * 0.4
    )

    probability = 1 / (1 + np.exp(-score))

    target = rng.binomial(1, probability)

    # Numeric preprocessing
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # Complete machine-learning pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model.fit(X_train, y_train)

    # Evaluate the demonstration model
    try:
        predictions = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, predictions)

        print(f"Demo model ROC-AUC: {auc:.3f}")
        print("Evaluation uses synthetic data only.")

    except ValueError:
        print("ROC-AUC could not be calculated.")

    # Save trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")


# --------------------------------------------------
# 4. LOAD MODEL
# --------------------------------------------------

def load_pipeline():
    """
    Load the saved model.
    Train the demonstration model if no saved model exists.
    """

    if not MODEL_PATH.exists():
        print(
            "Trained model not found. "
            "Creating a demonstration model..."
        )

        train_demo_model()

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 5. FEATURE REFERENCE VALUES
# --------------------------------------------------

REFERENCE_VALUES = {
    "loan_amnt": 10000,
    "int_rate": 12.0,
    "installment": 350,
    "annual_inc": 50000,
    "dti": 18.0,
    "delinq_2yrs": 0,
    "fico_range_low": 700,
    "fico_range_high": 704,
    "open_acc": 6,
    "pub_rec": 0,
    "revol_bal": 12000,
    "revol_util": 45.0,
    "total_acc": 10,
    "grade": "B",
    "sub_grade": "B3",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
}


# --------------------------------------------------
# 6. PREDICTION AND FEATURE CONTRIBUTIONS
# --------------------------------------------------

def explain_prediction(applicant_data, pipeline_artifact):
    """
    Predict default probability and estimate feature sensitivity.

    Each feature is changed to a reference value individually.
    The difference between the original and modified predictions
    is an approximate sensitivity indicator.

    These indicators are NOT SHAP values and are NOT causal
    explanations.
    """

    # Build input DataFrame in the correct feature order
    applicant = pd.DataFrame(
        [applicant_data],
        columns=FEATURES,
    )

    # Predict original probability
    baseline_probability = float(
        pipeline_artifact.predict_proba(applicant)[0, 1]
    )

    # Assign illustrative risk category
    if baseline_probability < 0.30:
        risk_level = "LOW"

    elif baseline_probability < 0.60:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    # Calculate feature sensitivity
    feature_contributions = {}

    for feature in FEATURES:

        modified_applicant = applicant.copy()

        modified_applicant.loc[0, feature] = (
            REFERENCE_VALUES[feature]
        )

        modified_probability = float(
            pipeline_artifact.predict_proba(
                modified_applicant
            )[0, 1]
        )

        # Positive value means the original prediction is higher
        # than the prediction after replacing this feature.
        contribution = (
            baseline_probability - modified_probability
        )

        feature_contributions[feature] = contribution

    # Select the 10 largest absolute changes
    top_features = sorted(
        feature_contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:10]

    # Dictionary for the dashboard chart
    feature_values = {
        feature: round(float(value), 6)
        for feature, value in top_features
    }

    # Explanatory text for the dashboard
    top_reasons = [
        (
            f"{feature}: estimated probability difference "
            f"{value:+.2%} compared with the reference value"
        )
        for feature, value in top_features[:5]
    ]

    return {
        "default_probability": baseline_probability,
        "risk_level": risk_level,
        "top_reasons": top_reasons,
        "shap_values": feature_values,
    }
