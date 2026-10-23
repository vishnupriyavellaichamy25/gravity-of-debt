import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Gravity of Debt",
    page_icon="💸",
    layout="wide"
)

st.title("💸 Gravity of Debt: Credit Risk Prediction Engine")
st.markdown(
    "A demonstration application for evaluating loan applicants."
)

tab1, tab2 = st.tabs(["Single Applicant", "Batch Prediction"])

# Sidebar: Applicant Profile
st.sidebar.header("Applicant Profile")


def get_user_input():
    loan_amnt = st.sidebar.number_input(
        "Loan Amount ($)", min_value=1000.0,
        max_value=40000.0, value=15000.0
    )

    int_rate = st.sidebar.slider(
        "Interest Rate (%)", 5.0, 35.0, 12.0
    )

    installment = st.sidebar.number_input(
        "Monthly Installment ($)", min_value=30.0,
        max_value=1500.0, value=400.0
    )

    grade = st.sidebar.selectbox(
        "Loan Grade", list("ABCDEFG"), index=1
    )

    sub_grade = st.sidebar.selectbox(
        "Sub-Grade",
        [f"{grade}{i}" for i in range(1, 6)],
        index=2
    )

    emp_length = st.sidebar.selectbox(
        "Employment Length",
        [
            "< 1 year", "1 year", "2 years", "3 years",
            "4 years", "5 years", "6 years", "7 years",
            "8 years", "9 years", "10+ years"
        ],
        index=5
    )

    home_ownership = st.sidebar.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"],
        index=2
    )

    annual_inc = st.sidebar.number_input(
        "Annual Income ($)", min_value=5000.0,
        max_value=1000000.0, value=75000.0
    )

    verification_status = st.sidebar.selectbox(
        "Verification Status",
        ["Not Verified", "Source Verified", "Verified"],
        index=1
    )

    purpose = st.sidebar.selectbox(
        "Purpose",
        [
            "debt_consolidation", "credit_card",
            "home_improvement", "major_purchase",
            "small_business", "car", "medical", "other"
        ]
    )

    dti = st.sidebar.slider(
        "Debt-to-Income Ratio (%)", 0.0, 50.0, 18.0
    )

    delinq_2yrs = st.sidebar.number_input(
        "Delinquencies (Past 2 yrs)",
        min_value=0, max_value=20, value=0
    )

    fico_base = st.sidebar.slider(
        "FICO Score", 600, 850, 700
    )

    open_acc = st.sidebar.number_input(
        "Open Accounts", min_value=2, max_value=50, value=10
    )

    pub_rec = st.sidebar.number_input(
        "Public Records", min_value=0, max_value=10, value=0
    )

    revol_bal = st.sidebar.number_input(
        "Revolving Balance ($)", min_value=0.0,
        max_value=100000.0, value=15000.0
    )

    revol_util = st.sidebar.slider(
        "Revolving Utilization (%)", 0.0, 150.0, 50.0
    )

    total_acc = st.sidebar.number_input(
        "Total Accounts", min_value=2, max_value=100, value=20
    )

    return {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "fico_range_low": float(fico_base),
        "fico_range_high": float(fico_base + 4),
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc
    }


user_data = get_user_input()

# Single Applicant
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Predict Risk")

        if st.button(
            "Evaluate Applicant",
            type="primary",
            use_container_width=True
        ):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=user_data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    prob = result["default_probability"]
                    risk = result["risk_level"]

                    color = (
                        "green" if risk == "LOW"
                        else "orange" if risk == "MEDIUM"
                        else "red"
                    )

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            title={
                                "text": "Default Probability (%)"
                            },
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": color},
                                "steps": [
                                    {
                                        "range": [0, 20],
                                        "color": "lightgreen"
                                    },
                                    {
                                        "range": [20, 50],
                                        "color": "lightyellow"
                                    },
                                    {
                                        "range": [50, 100],
                                        "color": "lightpink"
                                    }
                                ]
                            }
                        )
                    )

                    st.plotly_chart(
                        fig, use_container_width=True
                    )

                    st.markdown(f"### Risk Level: **{risk}**")
                    st.write("**Top Reasons:**")

                    for reason in result.get("top_reasons", []):
                        st.write(f"- {reason}")

                    st.subheader("Feature Sensitivity Analysis")

                    feature_values = result.get("shap_values", {})

                    if feature_values:
                        sorted_features = sorted(
                            feature_values.items(),
                            key=lambda item: abs(item[1]),
                            reverse=True
                        )[:10]

                        sorted_features.sort(
                            key=lambda item: item[1]
                        )

                        features = [
                            item[0] for item in sorted_features
                        ]
                        values = [
                            item[1] for item in sorted_features
                        ]

                        colors = [
                            "red" if value > 0 else "green"
                            for value in values
                        ]

                        fig2 = go.Figure(
                            go.Bar(
                                x=values,
                                y=features,
                                orientation="h",
                                marker_color=colors,
                                hovertemplate=(
                                    "Feature: %{y}<br>"
                                    "Estimated probability difference: "
                                    "%{x:.4f}<extra></extra>"
                                )
                            )
                        )

                        fig2.update_layout(
                            title="Top 10 Feature Sensitivity Estimates",
                            xaxis_title=(
                                "Change in Predicted Probability"
                            ),
                            yaxis_title="Applicant Features",
                            template="plotly_white",
                            height=500,
                            margin=dict(
                                l=150, r=30, t=70, b=80
                            )
                        )

                        fig2.add_vline(
                            x=0, line_dash="dash"
                        )

                        st.plotly_chart(
                            fig2, use_container_width=True
                        )

                        st.caption(
                            "These values estimate changes in predicted "
                            "probability when individual features are "
                            "replaced with reference values. They are not "
                            "actual SHAP values or causal explanations. "
                            "The demonstration model uses synthetic data."
                        )
                    else:
                        st.info(
                            "No feature sensitivity data is available."
                        )

                else:
                    st.error(
                        f"API Error: {response.status_code} - "
                        f"{response.text}"
                    )

            except requests.exceptions.RequestException as error:
                st.error(
                    "Connection failed. Make sure the FastAPI server "
                    f"is running. Details: {error}"
                )

    with col2:
        st.subheader("Applicant Comparison")

        avg_data = {
            "FICO Score": {
                "Applicant": user_data["fico_range_low"],
                "Average": 700
            },
            "Annual Income ($)": {
                "Applicant": user_data["annual_inc"],
                "Average": 75000
            },
            "DTI (%)": {
                "Applicant": user_data["dti"],
                "Average": 18.0
            },
            "Loan Amount ($)": {
                "Applicant": user_data["loan_amnt"],
                "Average": 15000
            },
            "Revolving Util (%)": {
                "Applicant": user_data["revol_util"],
                "Average": 50.0
            }
        }

        df_comp = pd.DataFrame(avg_data).T

        st.dataframe(
            df_comp.style.highlight_max(axis=1),
            use_container_width=True
        )

# Batch Prediction
with tab2:
    st.subheader("Batch Prediction")
    st.markdown(
        "Upload a CSV containing applicant data for batch prediction."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv"
    )

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df_batch)} applicants.")

            if st.button("Run Batch Prediction"):
                if df_batch.empty:
                    st.warning("The uploaded CSV contains no rows.")
                else:
                    required_columns = [
                        "loan_amnt", "int_rate", "installment",
                        "annual_inc", "dti", "delinq_2yrs",
                        "fico_range_low", "fico_range_high",
                        "open_acc", "pub_rec", "revol_bal",
                        "revol_util", "total_acc", "grade",
                        "sub_grade", "emp_length", "home_ownership",
                        "verification_status", "purpose"
                    ]

                    missing_columns = [
                        column for column in required_columns
                        if column not in df_batch.columns
                    ]

                    if missing_columns:
                        st.error(
                            "Missing required CSV columns: "
                            + ", ".join(missing_columns)
                        )
                    else:
                        results = []
                        limit = min(len(df_batch), 100)
                        progress_bar = st.progress(0)

                        if len(df_batch) > 100:
                            st.warning(
                                "Only the first 100 rows will be processed."
                            )

                        for i in range(limit):
                            row_dict = df_batch.iloc[i][
                                required_columns
                            ].to_dict()

                            try:
                                resp = requests.post(
                                    f"{API_URL}/predict",
                                    json=row_dict,
                                    timeout=30
                                )

                                if resp.status_code == 200:
                                    prediction = resp.json()
                                    row_dict["Prediction_Prob"] = (
                                        prediction["default_probability"]
                                    )
                                    row_dict["Risk_Level"] = (
                                        prediction["risk_level"]
                                    )
                                else:
                                    row_dict["Prediction_Prob"] = None
                                    row_dict["Risk_Level"] = (
                                        f"Error {resp.status_code}"
                                    )

                            except requests.exceptions.RequestException:
                                row_dict["Prediction_Prob"] = None
                                row_dict["Risk_Level"] = "Connection Error"

                            results.append(row_dict)
                            progress_bar.progress((i + 1) / limit)

                        df_results = pd.DataFrame(results)

                        st.success("Batch Prediction Complete!")
                        st.dataframe(
                            df_results, use_container_width=True
                        )

                        csv_output = df_results.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_output,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as error:
            st.error(f"Could not read the CSV file: {error}")
