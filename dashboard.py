import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import io

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Gravity of Debt", page_icon="💸", layout="wide")

st.title("💸 Gravity of Debt: Credit Risk Prediction Engine")
st.markdown("A complete ML solution for evaluating loan applicants.")

tab1, tab2 = st.tabs(["Single Applicant", "Batch Prediction"])

# --- Sidebar Form ---
st.sidebar.header("Applicant Profile")

def get_user_input():
    # Use realistic default values
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=1000.0, max_value=40000.0, value=15000.0)
    int_rate = st.sidebar.slider("Interest Rate (%)", min_value=5.0, max_value=35.0, value=12.0)
    installment = st.sidebar.number_input("Monthly Installment ($)", min_value=30.0, max_value=1500.0, value=400.0)
    
    grade = st.sidebar.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=1)
    sub_grade = st.sidebar.selectbox("Sub-Grade", [f"{grade}{i}" for i in range(1, 6)], index=2)
    
    emp_length = st.sidebar.selectbox("Employment Length", 
        ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', 
         '6 years', '7 years', '8 years', '9 years', '10+ years'], index=10)
         
    home_ownership = st.sidebar.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'ANY'], index=2)
    annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=5000.0, max_value=1000000.0, value=75000.0)
    verification_status = st.sidebar.selectbox("Verification Status", ['Not Verified', 'Source Verified', 'Verified'], index=1)
    
    purpose = st.sidebar.selectbox("Purpose", 
        ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 
         'small_business', 'car', 'medical', 'other'], index=0)
         
    dti = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 18.0)
    delinq_2yrs = st.sidebar.number_input("Delinquencies (Past 2 yrs)", 0.0, 20.0, 0.0)
    
    fico_base = st.sidebar.slider("FICO Score", 600, 850, 700)
    fico_range_low = float(fico_base)
    fico_range_high = float(fico_base + 4)
    
    open_acc = st.sidebar.number_input("Open Accounts", 2.0, 50.0, 10.0)
    pub_rec = st.sidebar.number_input("Public Records", 0.0, 10.0, 0.0)
    revol_bal = st.sidebar.number_input("Revolving Balance ($)", 0.0, 100000.0, 15000.0)
    revol_util = st.sidebar.slider("Revolving Utilization (%)", 0.0, 150.0, 50.0)
    total_acc = st.sidebar.number_input("Total Accounts", 2.0, 100.0, 20.0)
    
    return {
        "loan_amnt": loan_amnt, "int_rate": int_rate, "installment": installment,
        "grade": grade, "sub_grade": sub_grade, "emp_length": emp_length,
        "home_ownership": home_ownership, "annual_inc": annual_inc,
        "verification_status": verification_status, "purpose": purpose,
        "dti": dti, "delinq_2yrs": delinq_2yrs, "fico_range_low": fico_range_low,
        "fico_range_high": fico_range_high, "open_acc": open_acc,
        "pub_rec": pub_rec, "revol_bal": revol_bal, "revol_util": revol_util,
        "total_acc": total_acc
    }

user_data = get_user_input()

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Predict Risk")
        if st.button("Evaluate Applicant", type="primary", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/predict", json=user_data)
                if response.status_code == 200:
                    result = response.json()
                    
                    prob = result['default_probability']
                    risk = result['risk_level']
                    
                    # Gauge Chart
                    color = "green" if risk == "LOW" else "orange" if risk == "MEDIUM" else "red"
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Default Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgreen"},
                                {'range': [20, 50], 'color': "lightyellow"},
                                {'range': [50, 100], 'color': "lightpink"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"### Risk Level: **{risk}**")
                    
                    st.write("**Top Reasons:**")
                    for r in result['top_reasons']:
                        st.write(f"- {r}")
                        
                    # SHAP Chart
                    st.subheader("Feature Contributions (SHAP)")
                    shap_vals = result['shap_values']
                    # Sort by absolute magnitude
                    sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=False)
                    # Get top 10 for plot
                    sorted_shap = sorted_shap[-10:]
                    
                    features = [x[0] for x in sorted_shap]
                    values = [x[1] for x in sorted_shap]
                    colors = ['red' if v > 0 else 'green' for v in values]
                    
                    fig2 = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker_color=colors
                    ))
                    fig2.update_layout(title="Top 10 Most Impactful Features", xaxis_title="SHAP Value (Impact on Risk)")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection failed. Is the API running? Error: {e}")

    with col2:
        st.subheader("Applicant Comparison")
        # Dummy comparison data for illustration
        avg_data = {
            "FICO Score": {"Applicant": user_data['fico_range_low'], "Average": 700},
            "Annual Income ($)": {"Applicant": user_data['annual_inc'], "Average": 75000},
            "DTI (%)": {"Applicant": user_data['dti'], "Average": 18.0},
            "Loan Amount ($)": {"Applicant": user_data['loan_amnt'], "Average": 15000},
            "Revolving Util (%)": {"Applicant": user_data['revol_util'], "Average": 50.0}
        }
        df_comp = pd.DataFrame(avg_data).T
        st.dataframe(df_comp.style.highlight_max(axis=1), use_container_width=True)

with tab2:
    st.subheader("Batch Prediction")
    st.markdown("Upload a CSV with applicant data to get predictions in bulk.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_batch)} applicants.")
        
        if st.button("Run Batch Prediction"):
            results = []
            progress_bar = st.progress(0)
            
            # This is slow for large files; ideally we'd have a batch endpoint
            # For demonstration, calling line by line up to 100 rows
            limit = min(len(df_batch), 100)
            if len(df_batch) > 100:
                st.warning("Only processing first 100 rows for demonstration.")
                
            for i in range(limit):
                row_dict = df_batch.iloc[i].to_dict()
                try:
                    resp = requests.post(f"{API_URL}/predict", json=row_dict)
                    if resp.status_code == 200:
                        res_json = resp.json()
                        row_dict['Prediction_Prob'] = res_json['default_probability']
                        row_dict['Risk_Level'] = res_json['risk_level']
                    else:
                        row_dict['Prediction_Prob'] = "Error"
                        row_dict['Risk_Level'] = "Error"
            
                except Exception as e:
                    row_dict['Prediction_Prob'] = 0
                    row_dict['Risk_Level'] = "UNKNOWN"
                    print("Error:", e)
                    
                results.append(row_dict)
                progress_bar.progress((i + 1) / limit)
                
            df_results = pd.DataFrame(results)
            st.success("Batch Prediction Complete!")
            cols_to_show = ['Prediction_Prob', 'Risk_Level']

            if 'loan_amnt' in df_results.columns:
              cols_to_show.insert(0, 'loan_amnt')

            st.dataframe(df_results[cols_to_show].head())
            
            # Download button
            csv_output = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_output,
                file_name='batch_predictions.csv',
                mime='text/csv',
            )
